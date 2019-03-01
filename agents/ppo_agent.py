'''
/* Copyright (C) 2019 Gerti Tuzi - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MY license, which unfortunately won't be written
 * because no one would care to read it, and in the best of scenarios
 * it will only be used for ego pampering needs.
 */
'''

import numpy as np
import random
from collections import deque
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from agents.topologies.actor import FCGaussianActorValue
from agents.utils.schedule import LinearSchedule
from agents.utils.buffers import ReplayBuffer

BATCH_SIZE = 4096  # minibatch size
GAMMA = 0.99  # discount factor
LR = 3e-4  # learning rate of the actor
WEIGHT_DECAY = 0.0  # L2 weight decay
N_EPISODES = 1000
ROLLOUT = 1000
BUFFER_SIZE = int(ROLLOUT)
LEARN_REPEAT = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
LOG_EVERY = LEARN_REPEAT // 2
class Agent():
    def __init__(self, state_size, action_size, random_seed,
                 config = None, n_agents = 1, writer=None):
        '''

        :param state_size:
        :param action_size:
        :param random_seed:
        :param n_agents:
        :param writer:
        :param rollout_length:
        '''
        self._init_params(action_size, config, n_agents, random_seed, state_size)
        self.buffer = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size,
                                   seed=random_seed, device = self.device) # TrajectoryBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        self.net = FCGaussianActorValue(state_size, action_size, self.seed).to(self.device)
        self.alpha = LinearSchedule(1., 0.0001, self.n_episodes * self.rollout)

        self.active_logstd = self.exploratory_log_sigma_start
        self.exploratory_logstd_schedule = LinearSchedule(self.exploratory_log_sigma_start,
                                                          self.exploratory_log_sigma_end,
                                                          self.n_episodes)

        self.optimizer = optim.Adam([p for p in self.net.parameters() if p.requires_grad], lr=self.lr)
        self.writer = writer if writer is not None else SummaryWriter()
        self.total_steps = 0

        self.c1 = 10    # VF weight
        self.c2 = 0.01 # Entropy weight

        self.dtarg = 0.01
        self.eps = 0.2
        self.eps_ = self.eps
        self.gradient_clip = 10.0
        self.lam = 0.95



    def _init_params(self, action_size, config, n_agents, random_seed, state_size):

        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.n_episodes = N_EPISODES if config is None else config.n_episodes
        self.rollout = ROLLOUT if config is None else config.rollout
        self.batch_size = int(BATCH_SIZE) if config is None else int(config.batch_size)
        self.buffer_size = int(BUFFER_SIZE) if config is None else int(config.replay_buffer_size)
        self.gamma = GAMMA if config is None else config.discount
        self.lr = LR if config is None else config.critic_lr
        self.weight_decay = WEIGHT_DECAY if config is None else config.weight_decay
        self.device = DEVICE if ((config is None) or (config.device is None)) else config.device
        self.gradient_clip = None if config is None else config.grad_clip
        self.learn_repeat = int(LEARN_REPEAT) if ((config is None) or (config.learn_repeat is None)) else int(config.learn_repeat)

        self.exploratory_log_sigma_start = -0.7  if ((config is None) or (config.exploratory_log_sigma_start is None)) else config.exploratory_log_sigma_start
        self.exploratory_log_sigma_end = -1.6  if ((config is None) or (config.exploratory_log_sigma_end is None)) else config.exploratory_log_sigma_end

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        mode = self.net.training
        self.net.eval()
        with torch.no_grad():
            self.net.logstd.data.fill_(self.active_logstd)
            action, log_p, v = self.net(state)
        self.net.train(mode)
        return action.cpu().data.numpy(), log_p.cpu().data.numpy(), v.cpu().data.numpy()

    def test_act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        mode = self.net.training
        self.net.eval()
        with torch.no_grad():
            self.net.logstd.data.fill_(-10.0) # Sample with minimal std deviation
            action = self.net(state)[0]
        self.net.train(mode)
        return action.cpu().data.numpy()

    def optimize_policy(self):
        '''
            Given the collected traces, optimize the policy
        :return:
        '''

        for _ in range(self.learn_repeat):
            experiences = self.buffer.sample(attrs = ['s', 'a', 'r', 'sp', 'logp', 'v','A', 'rFuture', 'done'])
            self.learn(experiences)

        #Anneal log std
        self.active_logstd = self.exploratory_logstd_schedule()

    def learn(self, experiences):
        '''
        :param experiences:
        :param gamma:
        :return:
        '''
        # states, actions, rewards, next_states, old_logp, Vold, dones, R, A = experiences

        if self.total_steps % LOG_EVERY == 0:
            self.log_histogram(experiences.a.detach().cpu().numpy(), 'batch_actions')

        loss = self.objective(experiences)

        _alpha = self.alpha()
        self.eps = self.eps_ * _alpha # Linearly anneal the epsilon ... loosen up in time
        self._update_LR(LR * _alpha) # Decay LR

        # Clear out any debris in the gradient variables
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_clip)
        self.optimizer.step()

        if self.total_steps % LOG_EVERY == 0:
            _norm = self.compute_param_grad_norm(self.net.parameters(), norm_type=2)
            self.log_scalar(_norm, 'grad_norm', self.total_steps)

        self.total_steps +=1

    def objective(self, experiences):
        # Obtain the log probability under the policy as it is now
        log_p, entropy, v_expected = self.net.sa_logp_entropy_value(states=experiences.s, actions = experiences.a) # This should require a grad

        Aunnorm = experiences.A
        A = (Aunnorm - Aunnorm.mean()) / (Aunnorm.std() + 1e-8)
        if self.total_steps % LOG_EVERY == 0:
            self.log_histogram(Aunnorm.detach().cpu().numpy(), 'Aunnorm')
            self.log_histogram(A.detach().cpu().numpy(), 'A')
            self.log_scalar(float(Aunnorm.detach().mean()), 'bellman_me')
            self.log_scalar(float(Aunnorm.detach().pow(2).mean()), 'bellman_mse')

        # Entropy loss
        entropy_loss = entropy.mean()
        if self.total_steps % LOG_EVERY == 0:
            self.log_scalar(entropy_loss.detach().cpu().numpy(), 'entropy_loss')

        # ratio = p / p_old
        # objective function is taking the gradient wrt to theta (not theta old)
        # Note: experiences.logp is the old logp
        ratio = (log_p - experiences.logp.detach()).exp()
        pg_loss_unclipped = -A*ratio
        pg_loss_clipped = -A*ratio.clamp(1-self.eps, 1+self.eps)
        pg_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()

        if self.total_steps % LOG_EVERY == 0:
            for ai in range(self.action_size):
                self.log_histogram(ratio[..., ai].detach().cpu().numpy(), 'ratio_{}'.format(ai))
                self.log_histogram(pg_loss_unclipped[..., ai].detach().cpu().numpy(), 'pg_loss_unclipped_{}'.format(ai))
                self.log_histogram(pg_loss_clipped[..., ai].detach().cpu().numpy(), 'pg_loss_clipped_{}'.format(ai))
            self.log_scalar(pg_loss.detach().cpu().numpy(), 'pg_loss')


        v_expected_clipped = experiences.v + (v_expected - experiences.v).clamp(-self.eps, self.eps)
        vf_loss_1 = (experiences.rFuture - v_expected).pow(2)
        vf_loss_2 = (experiences.rFuture - v_expected_clipped).pow(2)
        # vf_loss = 0.5 * (R - v_expected).pow(2).mean()
        vf_loss = 0.5*(torch.max(vf_loss_1, vf_loss_2)).mean()

        if self.total_steps % LOG_EVERY == 0:
            self.log_scalar(vf_loss.detach().cpu().numpy(), 'vf_loss')
            self.log_scalar(vf_loss_1.detach().mean().cpu().numpy(), 'td_mse')

        return pg_loss + self.c1*vf_loss - self.c2*entropy_loss

    def collect_traces(self, traces):
        '''
            Collect simulation traces from the outside.
        :param traces:
        :return:
        '''

        self.buffer.clear()

        for tau in traces:
            _A = [None] * len(tau)
            rFuture_prev = 0.
            adv_prev = 0.

            # Slide backwards along the trace
            for i in reversed(range(len(tau))):
                v = tau[i].v
                vp = np.zeros_like(v) if (i + 1) == len(tau) else tau[i + 1].v
                r = tau[i].r
                _done = tau[i].done.astype(np.int)

                # Vtarget = r + y*V(s')
                _vtarget = r + GAMMA * vp * (1. - _done)

                # Advantage: V_target(s) - V(s) = [r + y*V(s')] - V(s)
                delta_t = _vtarget - v
                tau[i].A = delta_t + self.lam*GAMMA*adv_prev # A generalized form
                tau[i].rFuture = r + GAMMA*rFuture_prev

                adv_prev = tau[i].A
                rFuture_prev = tau[i].rFuture

        # Add samples to buffer
        for tau in traces:
            for i in range(len(tau)):
                self.buffer.add(tau[i], ['s', 'a', 'r', 'sp', 'logp','v','A', 'rFuture', 'done'])

    def _update_LR(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def log_scalar(self, val, tag='scalar', step = None):
        if step is None:
            step = self.total_steps
        self.writer.add_scalar(tag, val, step)

    def log_histogram(self, vals, tag='scalar', step = None):
        if step is None:
            step = self.total_steps
        self.writer.add_histogram(tag, vals, step)


    def compute_param_grad_norm(self, params, norm_type = 2):
        total_norm = 0
        parameters = list(filter(lambda p: p.grad is not None, params))
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def get_param_grads(self, params):
        parameters = list(filter(lambda p: p.grad is not None, params))
        pout = []
        for p in parameters:
            pout += np.copy(p.grad.data.cpu().numpy()).reshape(-1).tolist()
        return pout

# class Experience:
#     def __init__(self, **kwargs):
#         self._populate_(**kwargs)
#
#     def _populate_(self, **kwargs):
#         # Minimal experience parameters: sarsa and done
#         assert 's' in kwargs
#         assert 'a' in kwargs
#         assert 'r' in kwargs
#         assert 'sp' in kwargs
#         assert 'done' in kwargs
#
#         for k,v in kwargs.items():
#             if k == 's':
#                 self.s = v
#             elif k == 'a':
#                 self.a = v
#             elif k == 'r':
#                 self.r = v
#             elif k == 'sp':
#                 self.sp = v
#             elif k == 'logp':
#                 self.logp = v
#             elif k == 'v':
#                 self.v = v
#             elif k == 'A':
#                 self.A = v
#             elif k == 'done':
#                 self.done = v
#             elif k == 'rFuture':
#                 self.rFuture = v
#             elif k == 'entropy':
#                 self.entropy = v


# class TrajectoryBuffer:
#     """Fixed-size buffer to store experience tuples."""
#
#     def __init__(self, action_size, buffer_size, batch_size, seed, **kwargs):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#         """
#         self.device = DEVICE if 'device' not in kwargs else kwargs['device']
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)  # internal trajectory (deque)
#         self.batch_size = batch_size
#         self.experience = Experience
#         self.seed = random.seed(seed)
#
#     def add(self, experience):
#         """Add a new experience to trajectory."""
#         n_agents = experience.s.shape[0]
#         if n_agents > 1:
#             state = np.vsplit(experience.s, n_agents)
#             action = np.vsplit(experience.a, n_agents)
#             reward = np.vsplit(experience.r, n_agents)
#             next_state = np.vsplit(experience.sp, n_agents)
#             log_p = np.vsplit(experience.logp, n_agents)
#             done = np.vsplit(experience.done, n_agents)
#             Rfuture = np.vsplit(experience.rFuture, n_agents)
#             V = np.vsplit(experience.v, n_agents)
#             A = np.vsplit(experience.A, n_agents)
#             for i in range(n_agents):
#                 e = self.experience(s=state[i], a=action[i], r=reward[i],
#                                     sp=next_state[i], logp=log_p[i], v=V[i],
#                                     done=done[i], rFuture=Rfuture[i], A=A[i])
#
#                 self.memory.append(e)
#         else:
#             self.memory.append(experience)
#
#     def sample(self):
#         """Randomly sample_set a batch of experiences from trajectory."""
#         experiences = random.sample(self.memory, k=self.batch_size)
#
#         states = torch.from_numpy(np.vstack([e.s for e in experiences if e is not None])).float().to(
#             self.device)
#         actions = torch.from_numpy(np.vstack([e.a for e in experiences if e is not None])).float().to(
#             self.device)
#         rewards = torch.from_numpy(np.vstack([e.r for e in experiences if e is not None])).float().to(
#             self.device)
#         next_states = torch.from_numpy(np.vstack([e.sp for e in experiences if e is not None])).float().to(
#             self.device)
#         logp = torch.from_numpy(np.vstack([e.logp for e in experiences if e is not None])).float().to(
#             self.device)
#         v = torch.from_numpy(np.vstack([e.v for e in experiences if e is not None])).float().to(
#             self.device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
#             self.device)
#         rfuture = torch.from_numpy(np.vstack([e.rFuture for e in experiences if e is not None])).float().to(
#             self.device)
#         advantage = torch.from_numpy(np.vstack([e.A for e in experiences if e is not None])).float().to(
#             self.device)
#         return (states, actions, rewards, next_states, logp, v, dones, rfuture, advantage)
#
#     def __len__(self):
#         """Return the current size of internal trajectory."""
#         return len(self.memory)
#
#     def clear(self):
#         self.memory.clear()


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


import torch.nn as nn
from torch.distributions.normal import Normal

class TruncatedNormalSamp():
    def __init__(self, mu, sig, a, b):
        self.varsize = mu.size()
        self.mu = mu.view(-1)
        self.sig = sig.view(-1)
        self.sample_set = torch.zeros(self.mu.size())
        self.sample_set_log_prob = torch.zeros(self.mu.size())
        self.a = a
        self.b = b

    def sample(self):
        self._sample()
        return self.sample_set.view(self.varsize)


    def _sample(self):
        dist = Normal(self.mu, self.sig)
        _idx_resamp = torch.ones(self.mu.size()).byte() # Keep track of what needs to be resampled
        i = 0
        while torch.sum(_idx_resamp) > 0:
            _idx_keep = torch.zeros(self.mu.size()).byte() # Mark what needs to be kept in this sample
            _samp = dist.sample()
            _idx_keep[_idx_resamp] |= (_samp[_idx_resamp] >= self.a) & (_samp[_idx_resamp] <= self.b)
            _idx_resamp &= (_samp < self.a) | (_samp > self.b)
            self.sample_set[_idx_keep] = _samp[_idx_keep]
            i += 1
            # if i % 100 == 0:
            #     print('Resample attempt: {}'.format(i))

