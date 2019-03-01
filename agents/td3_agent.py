'''
 Author: Gerti Tuzi
 Jan 26, 2018
'''

import numpy as np
import random
from tensorboardX import SummaryWriter
from agents.topologies.actor import FCActor
from agents.topologies.critic import FCCritic
import torch
import torch.nn.functional as F
import torch.optim as optim
from agents.utils.schedule import LinearSchedule, ConstantSchedule
from agents.utils.buffers import ReplayBuffer
from agents.utils.random_processes import GaussianProcess


# Defaults
LOG_EVERY = 100
BUFFER_SIZE = int(1e6)          # replay buffer size
BATCH_SIZE = 128                # minibatch size
GAMMA = 0.99                    # discount factor
TAU_PI = 5e-3                   # soft update parameter for the actor
TAU_Q = 5e-3                    # soft update parameter for the critic
LR_ACTOR = 1e-3                 # learning rate of the actor
LR_CRITIC = 1e-3                # learning rate of the critic
WEIGHT_DECAY = 0.0              # L2 weight decay
UPDATE_POLICY_EVERY = 3
EXPLORATORY_N = 1000
LEARN_REPEAT = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def assertTensorSize(x, shape):
    assert tuple(x.size()) == shape

class Agent:
    def __init__(self, state_size, action_size, random_seed, n_agents = 1, config = None, writer = None):

        # Parameters
        self._init_params(action_size, config, n_agents, random_seed, state_size)
        self._set_ACs(action_size, state_size, random_seed) # Setup actors and critics
        self._set_exploratory_process(config)

        self.reward_regularization_noise = GaussianProcess(size = (self.batch_size, self.action_size),
                                                           std=ConstantSchedule(0.2),
                                                           seed=random_seed)
        # Replay trajectory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, random_seed, self.device)
        self.writer = writer if writer is not None else SummaryWriter()
        self.total_steps = 0
        self.episodes = 0

    def _set_exploratory_process(self, config):
        # Setup exploratory process
        _start_sig = 0.2 if config is None else config.exploratory_sigma_start
        _end_sig = 0.2 if config is None else config.exploratory_sigma_end
        _sig_decay_steps = 1e5 if config is None else config.sigma_decay_steps
        self.noise = GaussianProcess(size = (self.n_agents, self.action_size),
                                     std = LinearSchedule(_start_sig, _end_sig,
                                                          steps=int(_sig_decay_steps)),
                                     seed=self.seed)

    def _set_ACs(self, action_size, state_size, random_seed):
        '''
            Setup actors and critics
        :param action_size: action size
        :param state_size: state size
        :param random_seed: random seed
        :return: None
        '''
        # Actor Networks (behavioral + target)
        self.actor_local = FCActor(state_size, action_size, random_seed).to(self.device)
        self.actor_target = FCActor(state_size, action_size, random_seed).to(self.device)
        self.hard_update(self.actor_local, self.actor_target)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr,
                                          weight_decay=self.actor_weight_decay)

        # Critic Networks (behavioral + target). Q1 + Q2, so 4 networks in total
        self.critics_local = [FCCritic(state_size, action_size, random_seed).to(self.device) for _ in range(2)]
        self.critics_target = [FCCritic(state_size, action_size, random_seed).to(self.device) for _ in range(2)]
        [self.hard_update(self.critics_local[i], self.critics_target[i]) for i in range(2)]
        self.critics_optimizer = [optim.Adam(self.critics_local[i].parameters(),
                                             lr=self.critic_lr,
                                             weight_decay=self.critic_weight_decay)
                                  for i in range(2)]

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
        self.update_actor_interval = UPDATE_POLICY_EVERY if (
                    (config is None) or (config.update_actor_interval is None)) else config.update_actor_interval
        self.learn_repeat = int(LEARN_REPEAT) if (
                (config is None) or (config.learn_repeat is None)) else int(config.learn_repeat)
        self.actor_grad_clip = None if config is None else config.actor_grad_clip
        self.critic_grad_clip = None if config is None else config.critic_grad_clip

    def load_critic(self, dir, i):
        '''
            Load weights for the i-th local critic from the passed in directory
        :param dir:
        :param i:
        :return:
        '''
        self.critics_local[i].load_state_dict(torch.load(dir))

    def load_actor(self, dir):
        '''
            Load actor weights
        :param dir:
        :return:
        '''
        self.actor_local.load_state_dict(torch.load(dir))

    def step(self, experience):
        self.memory.add(experience)
        self.total_steps += 1
        if (len(self.memory) > self.batch_size) and (self.total_steps > self.exploration_steps):
            for _ in range(self.learn_repeat):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        '''
            Obtain action from behavioral policy
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
                action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()


    def learn(self, experiences, gamma):
        '''
            Update policy and value parameters using given batch of experience tuples.
        :param experiences: tuple of (s, a, r, s', done)
        :param gamma: discount factor
        :return:
        '''

        # Update critics
        if (self.total_steps + 1) % self.update_critic_interval == 0:
            self._update_critics(experiences, gamma)

        # Update actor
        if (self.total_steps+1) % self.update_actor_interval == 0:
            self._update_actors(experiences)
            self._soft_update_ACs()

            if self.total_steps % LOG_EVERY == 0:
                self._log_grads()

    def _log_grads(self):
        _norm = self.compute_param_grad_norm(self.critics_local[0].parameters(), norm_type=2)
        self.log_scalar(_norm, 'critic_0_grad_norm', self.total_steps)
        _norm = self.compute_param_grad_norm(self.critics_local[1].parameters(), norm_type=2)
        self.log_scalar(_norm, 'critic_1_grad_norm', self.total_steps)
        _norm = self.compute_param_grad_norm(self.actor_local.parameters(), norm_type=2)
        self.log_scalar(_norm, 'actor_grad_norm', self.total_steps)

    def _soft_update_ACs(self):
        self.soft_update(self.critics_local[0], self.critics_target[0], self.tau_q)
        self.soft_update(self.critics_local[1], self.critics_target[1], self.tau_q)
        self.soft_update(self.actor_local, self.actor_target, self.tau_pi)

    def _update_actors(self, experiences):
        '''
            Update actor using deterministic gradient approach
        :param experiences:
        :return:
        '''

        a_est = self.actor_local(experiences.s)
        assertTensorSize(a_est, (self.batch_size, self.action_size))

        actor_loss = -self.critics_local[0].forward(experiences.s, a_est).mean()
        self.actor_optimizer.zero_grad()  # Clear out gradients
        actor_loss.backward()  # Compute gradients
        # Clip gradients
        if self.actor_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step() # Apply gradients

    def _update_critics(self, experiences, gamma):
        '''
            Critic learning step
        :param experiences:
        :param gamma:
        :return:
        '''

        with torch.no_grad():
            # Regularize rewards
            eps = self.reward_regularization_noise.sample()
            eps = np.clip(eps, -0.5, 0.5)
            eps = torch.from_numpy(eps.astype(np.float32)).to(self.device)
            a_tilde = self.actor_target(experiences.sp) + eps
            a_tilde = torch.clamp(a_tilde, -1, 1.)
            q_0_next = self.critics_target[0].forward(experiences.sp, a_tilde)
            q_1_next = self.critics_target[1].forward(experiences.sp, a_tilde)
            q_target_next = torch.cat((q_0_next, q_1_next), dim=1)
            q_target_next, _ = torch.min(q_target_next, dim=1)
            if len(tuple(q_target_next.size())) == 1:
                q_target_next = q_target_next.unsqueeze(1)
            y = experiences.r + gamma * q_target_next * (1 - experiences.done)

            assertTensorSize(eps, (self.batch_size, self.action_size))
            assertTensorSize(a_tilde, (self.batch_size, self.action_size))
            assertTensorSize(q_0_next, (self.batch_size, 1))
            assertTensorSize(q_1_next, (self.batch_size, 1))
            assertTensorSize(q_target_next, (self.batch_size, 1))
            assertTensorSize(y, (self.batch_size, 1))

        # Compute loss
        y0_hat = self.critics_local[0].forward(experiences.s, experiences.a)
        y1_hat = self.critics_local[1].forward(experiences.s, experiences.a)
        assertTensorSize(y0_hat, (self.batch_size, 1))
        assertTensorSize(y1_hat, (self.batch_size, 1))

        critic_0_loss = F.mse_loss(y0_hat, y)
        critic_1_loss = F.mse_loss(y1_hat, y)
        [self.critics_optimizer[i].zero_grad() for i in range(2)]
        # Compute gradients
        critic_0_loss.backward()
        critic_1_loss.backward()
        # Clip gradients
        if self.critic_grad_clip is not None:
            [torch.nn.utils.clip_grad_norm_(self.critics_local[i].parameters(), self.critic_grad_clip) for i in range(2)]
        # Apply gradients
        [self.critics_optimizer[i].step() for i in range(2)]

        if self.total_steps % LOG_EVERY == 0:
            self.log_scalar(float(critic_0_loss), 'critic_0_bellman_mse')
            self.log_scalar(float(critic_1_loss), 'critic_1_bellman_mse')


    def soft_update(self, local_model, target_model, tau):
        '''
            θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: PyTorch model (weights will be copied from)
        :param target_model: PyTorch model (weights will be copied to)
        :param tau: momentum parameter
        :return:
        '''
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

    def log_scalar(self, val, tag='scalar', step = None):
        if step is None:
            step = self.total_steps
        self.writer.add_scalar(tag, val, step)

    def log_histogram(self, vals, tag='scalar', step = None):
        if step is None:
            step = self.total_steps
        self.writer.add_histogram(tag, vals, step)


