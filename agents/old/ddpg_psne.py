import numpy as np
import random
import copy
from collections import namedtuple, deque
from udacity.model2 import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

LOG_EVERY = 100

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decay

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, writer=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.writer = writer if writer is not None else SummaryWriter()
        self.total_steps = 0
        self.episodes = 0

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Param noise: Initialize the perturbed critic with the same weights as the local critic
        self.perturb_parm_sigma = 0.005
        self.perturb_delta = 0.2
        self.perturb_alpha = 1.01
        self.perturbed_actor = Actor(state_size, action_size, random_seed).to(device)
        self.perturb_the_actor()
        self.target_local_critic_divergence = 0.0

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        self.total_steps += 1
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            if np.all(done):
                self.perturb_the_actor()
                self.perturb_the_actor()

                d = self._compute_sample_perturbed_distance(experiences[0])
                self._adapt_perturbation_sigma(d)
                self.episodes += 1
                self.log_scalar(self.perturb_parm_sigma, 'perturb_sigma', self.total_steps)
                self.log_scalar(self.perturb_delta, 'perturb_delta', self.total_steps)
                self.log_scalar(d, 'perturb_distance', self.total_steps)
            self.learn(experiences, GAMMA)


    def test_act(self, state):
        state = torch.from_numpy(state).float().to(device)
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        self.actor_target.train()
        return action


    def act(self, state, add_noise=True, perturb_params=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        if add_noise and perturb_params:
            self.perturbed_actor.eval()
            with torch.no_grad():
                action = self.perturbed_actor(state).cpu().data.numpy()
            self.perturbed_actor.train()
        else:
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
        # Adding noise on actions
        if add_noise and (not perturb_params):
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        # Track the gradient norm of critic
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.)

        if self.total_steps % LOG_EVERY == 0:
            _grads = self.get_param_grads(self.critic_local.parameters())
            _norm = self.compute_param_grad_norm(self.critic_local.parameters(), norm_type=2)
            self.log_scalar(_norm, 'critic_local_grad_norm', self.total_steps)
            self.log_histogram(np.array(_grads), 'critic_local_grad', self.total_steps)
            _W, _B = self.get_net_weights(self.critic_local)
            for li in range(len(_W)):
                self.log_histogram(_W[li], 'critic_local_w{}'.format(li), self.total_steps)
                self.log_histogram(_B[li], 'critic_local_b{}'.format(li), self.total_steps)

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.total_steps % LOG_EVERY == 0:
            _grads = self.get_param_grads(self.actor_local.parameters())
            _norm = self.compute_param_grad_norm(self.actor_local.parameters(), norm_type=2)
            self.log_scalar(_norm, 'actor_local_grad_norm', self.total_steps)
            self.log_histogram(np.array(_grads), 'actor_local_grad', self.total_steps)
            _W, _B = self.get_net_weights(self.actor_local)
            for li in range(len(_W)):
                self.log_histogram(_W[li], 'actor_local_w{}'.format(li), self.total_steps)
                self.log_histogram(_B[li], 'actor_local_b{}'.format(li), self.total_steps)

        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    ################################ Parameter Perturbation ###############################
    def perturb_the_actor(self):
        '''
            Perturbing the parameters per paper
        :return:
        '''
        # Copy the local actor, it's time to boogie
        self.perturbed_actor.load_state_dict(self.actor_local.state_dict())
        for p in self.perturbed_actor.parameters():
            datasize = p.data.size()
            nels = p.data.numel()
            e = torch.distributions.normal.Normal(torch.zeros(nels),
                                                  self.perturb_parm_sigma * torch.ones(nels)).sample()
            p.data.add_(e.view(datasize))

    def _compute_sample_perturbed_distance(self, states):
        a = self.actor_local(states)
        ap = self.perturbed_actor(states)
        d = (1. / self.action_size) * torch.mean((a - ap) ** 2)
        return torch.sqrt(d)

    def _adapt_perturbation_sigma(self, d):
        if d < self.perturb_delta:
            self.perturb_parm_sigma *= self.perturb_alpha
        else:
            self.perturb_parm_sigma *= 1. / self.perturb_alpha

    ##########################################################################################

    def critic_prediction(self, critic, state, action):
        return critic(state, action)

    def compute_param_grad_norm(self, params, norm_type=2):
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

    def log_scalar(self, val, tag='scalar', step=None):
        if step is None:
            step = self.total_steps
        self.writer.add_scalar(tag, val, step)

    def log_histogram(self, vals, tag='scalar', step=None):
        if step is None:
            step = self.total_steps
        self.writer.add_histogram(tag, vals, step)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        n_agents = state.shape[0]
        if n_agents > 1:
            state = np.vsplit(state, n_agents)
            action = np.vsplit(action, n_agents)
            reward = np.vsplit(reward, n_agents)
            next_state = np.vsplit(next_state, n_agents)
            done = np.vsplit(done, n_agents)
            for s, a, r, ns, d in zip(state, action, reward, next_state, done):
                e = self.experience(s, a, r, ns, d)
                self.memory.append(e)
        else:
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)