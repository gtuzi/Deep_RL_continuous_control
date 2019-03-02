'''
    Author: Gerti Tuzi
'''
import numpy as np
import random
from tensorboardX import SummaryWriter
from agents.topologies.actor import FCActor
from agents.topologies.critic import FCCritic
from agents.utils.schedule import LinearSchedule
from agents.utils.buffers import ReplayBuffer
from agents.utils.random_processes import GaussianProcess

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
    def __init__(self, state_size, action_size, random_seed, n_agents = 1, config = None, writer=None):
        '''
            Constructor
        :param state_size (int): self explanatory
        :param action_size (int):self explanatory
        :param random_seed: will set random, numpy, and torch seed
        :param n_agents (int): self explanatory
        :param config: config file containing parameters of "RunConfig"
        :param writer:
        '''

        self._init_params(action_size, config, n_agents, random_seed, state_size)
        self._init_ACs()
        self._init_exploratory_process(config)

        self.writer = writer if writer is not None else SummaryWriter()
        self.total_steps = 0
        self.episodes = 0

        # Replay trajectory
        self.memory = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=random_seed)


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
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr,
                                          weight_decay=self.actor_weight_decay)

        self.perturbed_actor = FCActor(self.state_size, self.action_size, self.seed).to(self.device)
        self.hard_update(self.actor_local, self.perturbed_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = FCCritic(self.state_size, self.action_size, self.seed).to(self.device)
        self.critic_target = FCCritic(self.state_size, self.action_size, self.seed).to(self.device)
        self.hard_update(self.critic_local, self.critic_target)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr,
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
            Setup exploratory process
            Initialize the perturbed critic with the same weights as the local critic
        :param config:
        :return:
        '''
        _start_sig = 0.2 if (config is None) or (config.exploratory_sigma_start < 0.) else config.exploratory_sigma_start
        _end_sig = 0.2 if (config is None) or (config.exploratory_sigma_end) else config.exploratory_sigma_end
        _sig_decay_steps = 1e5 if config is None else config.sigma_decay_steps
        self.noise = [
            GaussianProcess((self.action_size,), LinearSchedule(_start_sig, _end_sig, steps=int(_sig_decay_steps)))
            for _ in range(self.n_agents)]

        _start_delta = 0.2 if config is None else config.psne_start_delta
        _end_delta = 0.2 if config is None else config.psne_end_delta
        _decay_steps = 1e5 if config is None else config.psne_decay_steps
        self.psne_delta = LinearSchedule(_start_delta, _end_delta, steps=int(_decay_steps))
        self.psne_alpha = 1.01 if config is None else config.psne_alpha
        self.psne_sigma = 0.005 if config is None else config.psne_sigma

        assert not callable(self.psne_sigma), 'PSNE sigma must be a literal value'
        assert not callable(self.psne_alpha), 'PSNE alpha must be a literal value'
        assert callable(self.psne_delta), 'psne delta must be a schedule type'

    def step(self, experience):
        """Save experience in replay trajectory, and use random sample_set from buffer to learn."""
        self.memory.add(experience) # Store experience in trajectory
        self.total_steps += 1

        if (len(self.memory) > self.batch_size) and (self.total_steps > self.exploration_steps):

            # Perturb actor's parameters at the end of the episode (PSNE method)
            if np.all(experience.done):
                experiences = self.memory.sample()
                self.perturb_the_actor(experiences)
                self.episodes += 1

            for _ in range(self.learn_repeat):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        '''
            Determine action. Explore if noise is requested, by perturbing the parameter space of the actor
        :param state: input to actor
        :param add_noise: Explore if true
        :return:
        '''
        if add_noise and (self.total_steps < self.exploration_steps):
            action = np.array([self.noise[i].sample() for i in range(self.n_agents)])
        else:
            state = torch.from_numpy(state).float().to(self.device)
            if add_noise:
                mode = self.perturbed_actor.training
                self.perturbed_actor.eval()
                with torch.no_grad():
                    action = self.perturbed_actor(state).cpu().data.numpy()
                self.perturbed_actor.train(mode)
            else:
                mode = self.actor_local.training
                self.actor_local.eval()
                with torch.no_grad():
                    action = self.actor_local(state).cpu().data.numpy()
                self.actor_local.train(mode)
        return np.clip(action, -1, 1)

    def reset(self):
        [self.noise[i].reset() for i in range(self.n_agents)]

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, policy_target(next_state))
        where:
            policy_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        if self.total_steps % self.update_critic_interval == 0:
            for _ in range(self.update_critic_repeat):
                self._update_critic(experiences, gamma)
            self.soft_update(self.critic_local, self.critic_target, self.tau_q)


        if self.total_steps % self.update_actor_interval == 0:
            for _ in range(self.update_actor_repeat):
                self._update_actor(experiences)
            self.soft_update(self.actor_local, self.actor_target, self.tau_pi)

        if self.total_steps % LOG_EVERY == 0:
            _grads = self.get_param_grads(self.critic_local.parameters())
            _norm = self.compute_param_grad_norm(self.critic_local.parameters(), norm_type=2)
            self.log_scalar(_norm, 'critic_grad_norm', self.total_steps)
            _grads = self.get_param_grads(self.actor_local.parameters())
            _norm = self.compute_param_grad_norm(self.actor_local.parameters(), norm_type=2)
            self.log_scalar(_norm, 'actor_grad_norm', self.total_steps)

    def _update_actor(self, experiences):
        # Compute actor objective and maximize (gradient ascend = negative grad descend)
        a_pred = self.actor_local( experiences.s)
        assertTensorSize(a_pred, (self.batch_size, self.action_size))

        actor_loss = -self.critic_local( experiences.s, a_pred).mean()
        # Clear out gradients
        self.actor_optimizer.zero_grad()
        # Compute gradients wrt loss
        actor_loss.backward()
        # Clip gradients
        if self.actor_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

    def _update_critic(self, experiences, gamma):
        '''
            Critic learning step
        :param experiences:
        :param gamma:
        :return:
        '''

        with torch.no_grad():
            ap = self.actor_target(experiences.sp)
            q_target_next = self.critic_target(experiences.sp, ap)
            if len(tuple(q_target_next.size())) == 1:
                q_target_next = q_target_next.unsqueeze(1)
            y = experiences.r + gamma * q_target_next * (1 - experiences.done)
        assertTensorSize(ap, (self.batch_size, self.action_size))
        assertTensorSize(q_target_next, (self.batch_size, 1))
        assertTensorSize(y, (self.batch_size, 1))

        # Compute loss
        y_hat = self.critic_local(experiences.s, experiences.a); assertTensorSize(y_hat, (self.batch_size, 1))
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

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    #---------------- Exploration methods: PSNE ---------------- #
    def perturb_the_actor(self, experiences):
        '''
            Perturbing the parameters per paper
            Ref: https://arxiv.org/pdf/1706.01905.pdf
        :return:
        '''
        # Copy the local actor, and perturb, it's time to boogie
        self.hard_update(local_model=self.actor_local, target_model=self.perturbed_actor)
        for name, p in dict(self.perturbed_actor.named_parameters()).items():
            if 'norm' not in name: # Don't perturb the normalization layers
                datasize = p.data.size()
                nels = p.data.numel()
                e = torch.distributions.normal.Normal(torch.zeros(nels), self.psne_sigma * torch.ones(nels)).sample().to(self.device)
                p.data.add_(e.view(datasize))

        # Compute the distance between local and perturbed actor
        d = self._compute_sample_perturbed_distance(experiences.s)
        # Adapt the sigma to match target d (moving average)
        self._adapt_perturbation_sigma(d)

        self.log_scalar(self.psne_sigma, 'perturb_sigma', self.total_steps)
        self.log_scalar(self.psne_delta(), 'psne_delta', self.total_steps)
        self.log_scalar(d, 'perturb_distance', self.total_steps)

    def _compute_sample_perturbed_distance(self, states):
        '''
            Compute mean squared distance between local actor and perturbed actor.
            Per https://arxiv.org/pdf/1706.01905.pdf section C2: this is equivalent to std. dev of d(pi, pi+noise)
        :param states:
        :return:
        '''
        a = self.actor_local(states)
        ap = self.perturbed_actor(states)
        d = (1. / self.action_size) * torch.mean((a - ap) ** 2)
        return torch.sqrt(d)

    def _adapt_perturbation_sigma(self, d):
        '''
            Updates variance of actor parameter's noise (i.e. std dev of exploration in parameter space).
        :param d:
        :return:
        '''
        if d < self.psne_delta():
            self.psne_sigma *= self.psne_alpha
        else:
            self.psne_sigma *= 1. / self.psne_alpha

    # ---------- Utilities ------------- #
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
