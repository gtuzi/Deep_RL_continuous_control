'''
/* Copyright (C) 2019 Gerti Tuzi - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MY license, which unfortunately won't be written
 * because no one would care to read it, and in the best of scenarios
 * it will only be used for ego pampering needs.
 */
'''


class RunConfig:
    def __init__(self):
        self.n_episodes = 10
        self.rollout = 1000
        self.batch_size = 128
        self.lr = 1.0e-3
        self.replay_buffer_size = 1e6
        self.discount = 0.995
        self.log_interval = int(1e2)
        self.save_interval = 0
        self.eval_interval = 0          # every ...
        self.eval_episodes = 1          # this number of episodes
        self.exploration_steps = 0      # number of actions
        self.learn_interval = 1         # TD learning: how often to learn
        self.learn_repeat = 1           # learn repeats

        # Actor - Critic (AC) parameters
        self.tau_pi = 1e-3
        self.tau_q = 1e-3
        self.update_critic_interval = 1
        self.update_critic_repeat = 1
        self.update_actor_interval = 1
        self.update_actor_repeat = 1
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3

        # AC Regularization
        self.actor_grad_clip = None
        self.critic_grad_clip = None
        self.actor_weight_decay = 0.0
        self.critic_weight_decay = 0.0


        # Regularization
        self.weight_decay = 0.0
        self.grad_clip = 1.0

        # Directories
        self.device = None
        self.train_agent = None
        self.eval_agent = None
        self.policy_dir  = None
        self.critic_0_log_dir = None
        self.critic_1_log_dir = None
        self.actor_log_dir = None

        # --- Noise parameters ----
        self.exploratory_sigma_start = 0.2 # TD3 default (paper)
        self.exploratory_sigma_end = 0.2   # TD3 default  (paper)
        self.exploratory_log_sigma_start = -0.7 # PPO one example   (paper)
        self.exploratory_log_sigma_end = -1.6   # PPO one example   (paper)
        self.sigma_decay_steps = 1e5

        # Ornstein-Uhlenbeck
        self.OU_theta = 0.15
        self.OU_mu = 0.

        # Parameter space noise exploration params (PSNE)
        self.psne_start_delta = 0.2
        self.psne_end_delta = 0.2
        self.psne_decay_steps = 1e5
        self.psne_alpha = 1.01
        self.psne_sigma = 0.1
