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
from tensorboardX import SummaryWriter
from agents.topologies.actor import FCActor
from agents.topologies.critic import FCCritic
from agents.utils.schedule import LinearSchedule
from agents.utils.buffers import ReplayBuffer
from agents.utils.random_processes import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim


# Defaults
LOG_EVERY = 100
BUFFER_SIZE = int(1e6)          # replay buffer size
BATCH_SIZE = 64                 # minibatch size
GAMMA = 0.99                    # discount factor
TAU_PI = 5e-3                   # soft update parameter for the actor
TAU_Q = 5e-3                    # soft update parameter for the critic
LR_ACTOR = 1e-4                 # learning rate of the actor
LR_CRITIC = 1e-4                # learning rate of the critic
WEIGHT_DECAY = 0.0              # L2 weight decay

UPDATE_POLICY_EVERY = 3         # Delay policy update by ...
EXPLORATORY_N = 1000            # Pure exploratory steps.
LEARN_REPEAT = 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def assertTensorSize(x, shape):
    assert tuple(x.size()) == shape

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, n_agents = 1, config = None, writer = None):
        '''
            Constructor
        :param state_size:
        :param action_size:
        :param random_seed: seed sets random, numpy, and torch
        :param n_agents:
        :param config:
        :param writer:
        '''

        self._init_params(action_size, config, n_agents, random_seed, state_size)
        self._init_ACs()
        self._init_exploratory_process(config)

        # Replay trajectory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, random_seed, device=self.device)

        self.writer = writer if writer is not None else SummaryWriter()
        self.total_steps = 0
        self.episodes = 0

    def _init_ACs(self):
        '''
            Start Actor-Critics
        :param random_seed:
        :return:
        '''
        # Actor Network (w/ Target Network)
        self.actor_local = FCActor(self.state_size, self.action_size, self.seed).to(self.device)
        self.actor_target = FCActor(self.state_size, self.action_size, self.seed).to(self.device)
        self.hard_update(self.actor_local, self.actor_target)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.actor_lr,
                                          weight_decay=self.actor_weight_decay)

        # Critic Network (w/ Target Network)
        self.critic_local = FCCritic(self.state_size, self.action_size, self.seed).to(self.device)
        self.critic_target = FCCritic(self.state_size, self.action_size, self.seed).to(self.device)
        self.hard_update(self.critic_local, self.critic_target)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.critic_lr,
                                           weight_decay=self.critic_weight_decay)


    def _init_params(self, action_size, config, n_agents, random_seed, state_size):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.batch_size = int(BATCH_SIZE) if config is None else int(config.batch_size)
        self.buffer_size = int(BUFFER_SIZE) if config is None else int(config.replay_buffer_size)
        self.gamma = GAMMA if config is None else config.discount
        self.critic_lr = LR_CRITIC if config is None else config.critic_lr
        self.critic_weight_decay = WEIGHT_DECAY if config is None else config.critic_weight_decay
        self.actor_lr = LR_ACTOR if config is None else config.actor_lr
        self.actor_weight_decay = 0.0 if config is None else config.actor_weight_decay
        self.exploration_steps = EXPLORATORY_N if config is None else config.exploration_steps
        self.tau_pi = TAU_PI if config is None else config.tau_pi
        self.tau_q = TAU_Q if config is None else config.tau_q
        self.device = DEVICE if ((config is None) or (config.device is None)) else config.device

        self.update_critic_interval = 1 if (
                (config is None) or (config.update_critic_interval is None)) else config.update_critic_interval
        self.update_critic_repeat = 1 if (
                (config is None) or (config.update_critic_repeat is None)) else config.update_critic_repeat

        self.update_actor_interval = UPDATE_POLICY_EVERY if (
                (config is None) or (config.update_actor_interval is None)) else config.update_actor_interval
        self.update_actor_repeat = 1 if (
                (config is None) or (config.update_actor_repeat is None)) else config.update_actor_repeat

        self.learn_repeat = int(LEARN_REPEAT) if (
                (config is None) or (config.learn_repeat is None)) else int(config.learn_repeat)
        self.actor_grad_clip = None if config is None else config.actor_grad_clip
        self.critic_grad_clip = None if config is None else config.critic_grad_clip

    def _init_exploratory_process(self, config):
        '''
            Set up the random exploratory procesess for each agent
        :param config:
        :return:
        '''
        # Setup exploratory process
        _start_sig = 0.2 if (config is None) or (config.exploratory_sigma_start < 0.) else config.exploratory_sigma_start
        _end_sig = 0.2 if (config is None) or (config.exploratory_sigma_end) else config.exploratory_sigma_end
        _sig_decay_steps = 1e5 if config is None else config.sigma_decay_steps
        self.noise = OUNoise(size = (self.n_agents, self.action_size),
                             seed=self.seed,
                             sigma=LinearSchedule(_start_sig, _end_sig,
                                                  steps=int(_sig_decay_steps)))

    def step(self, experience):
        self.memory.add(experience)
        self.total_steps += 1
        if (len(self.memory) > self.batch_size) and (self.total_steps > self.exploration_steps):
            for _ in range(self.learn_repeat):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        '''
            Obtain action from behavioral policy.
            If we're purely exploring (exploration_steps), actions
            will be taken randomly from the noise process.
        :param state: input state to the actor
        :param add_noise: if true, we're a training agent, otherwise strictly an evaluation action
        :return: action
        '''
        if add_noise and (self.total_steps < self.exploration_steps):
            action = self.noise.sample()
        else:
            state = torch.from_numpy(state).float().to(self.device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            if add_noise:
                action += np.array(self.noise.sample())
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()


    def learn(self, experiences, gamma):
        """
            Optimization step: Update actors and critics
        """

        if self.total_steps % self.update_critic_interval == 0:
            for _ in range(self.update_critic_repeat):
                self._update_critic(experiences, gamma)
            self.soft_update(self.critic_local, self.critic_target, self.tau_q)

        if self.total_steps % self.update_actor_interval == 0:
            for _ in range(self.update_actor_repeat):
                self._update_actor(experiences)
            self.soft_update(self.actor_local, self.actor_target, self.tau_pi)


    def _update_actor(self, experiences):
        # Compute actor objective and maximize (gradient ascend = negative grad descend)
        a_pred = self.actor_local(experiences.s)
        assertTensorSize(a_pred, (self.batch_size, self.action_size))

        actor_loss = -self.critic_local(experiences.s, a_pred).mean()
        # Clear out gradients
        self.actor_optimizer.zero_grad()
        # Compute gradients wrt loss
        actor_loss.backward()
        # Clip gradients
        if self.actor_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        if self.total_steps % LOG_EVERY == 0:
            _grads = self.get_param_grads(self.actor_local.parameters())
            _norm = self.compute_param_grad_norm(self.actor_local.parameters(), norm_type=2)
            self.log_scalar(_norm, 'actor_grad_norm', self.total_steps)

    def _update_critic(self, experiences, gamma):
        '''
            Critic learning step
        :param experiences:
        :param gamma:
        :return:
        '''
        # Target optimal return: yi = r + 𝛾Q'(s', μ'(s'))
        with torch.no_grad():
            a_up = self.actor_target(experiences.sp)
            qp = self.critic_target(experiences.sp, a_up)
            if len(tuple(qp.size())) == 1:
                qp = qp.unsqueeze(1)
            y = experiences.r + gamma * qp * (1 - experiences.done)
            assertTensorSize(a_up, (self.batch_size, self.action_size))
            assertTensorSize(qp, (self.batch_size, 1))
            assertTensorSize(y, (self.batch_size, 1))

        # Compute loss
        # y_hat = Q(s, a)
        y_hat = self.critic_local(experiences.s, experiences.a)
        assertTensorSize(y_hat, (self.batch_size, 1))
        critic_loss = F.mse_loss(y_hat, y)
        # Clear out gradients
        self.critic_optimizer.zero_grad()
        # Compute gradients
        critic_loss.backward()
        # Clip gradients
        if self.critic_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.critic_grad_clip)
        # Apply gradients
        self.critic_optimizer.step()

        if self.total_steps % LOG_EVERY == 0:
            _grads = self.get_param_grads(self.critic_local.parameters())
            _norm = self.compute_param_grad_norm(self.critic_local.parameters(), norm_type=2)
            self.log_scalar(_norm, 'critic_grad_norm', self.total_steps)
            self.log_scalar(float(critic_loss), 'bellman_mse')

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

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

    def get_net_weights(self, net):
        params = list(net.parameters())
        W = []
        B = []

        for li in range(0, len(params), 2):
            w = params[li].data.cpu().numpy().reshape(-1)
            b = params[li + 1].data.cpu().numpy().reshape(-1)
            W.append(w)
            B.append(b)
        return W, B

    def log_scalar(self, val, tag='scalar', step = None):
        if step is None:
            step = self.total_steps
        self.writer.add_scalar(tag, val, step)

    def log_histogram(self, vals, tag='scalar', step = None):
        if step is None:
            step = self.total_steps
        self.writer.add_histogram(tag, vals, step)
