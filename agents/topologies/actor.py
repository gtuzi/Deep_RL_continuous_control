'''
/* Copyright (C) 2019 Gerti Tuzi - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MY license, which unfortunately won't be written
 * because no one would care to read it, and in the best of scenarios
 * it will only be used for ego pampering needs.
 */
'''


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class FCActor(nn.Module):
    """Deterministic actor Model."""

    def __init__(self, state_size, action_size, seed, fc_units=(256, 256)):
        '''

        :param state_size:
        :param action_size:
        :param seed:
        :param fc_units:
        '''

        super(FCActor, self).__init__()
        self.seed = torch.manual_seed(seed) if seed is not None else None
        c_sizes = (state_size, ) + fc_units + (action_size,)
        self.n_layers = len(c_sizes) - 1

        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], action_size)

        self.fcnorm1 = nn.LayerNorm(fc_units[0])
        self.fcnorm2 = nn.LayerNorm(fc_units[1])
        self.inputnorm = nn.BatchNorm1d(state_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.inputnorm.weight.data.fill_(1.)
        self.inputnorm.bias.data.fill_(0.)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = self.inputnorm(state)
        x = F.relu(self.fcnorm1(self.fc1(state)))
        x = F.relu(self.fcnorm2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x

class FCGaussianActorValue(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=(256, 256)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        self.state_size = state_size
        self.action_size = action_size

        super(FCGaussianActorValue, self).__init__()
        self.seed = torch.manual_seed(seed)
        c_sizes = (state_size, ) + fc_units + (action_size,)
        self.n_layers = len(c_sizes) - 1

        # Body (features)
        self.state_norm = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])

        # Mean head
        self.fc_mu = nn.Linear(fc_units[1], action_size)

        # Std deviation: initialize to one
        self.logstd = nn.Parameter(torch.zeros(action_size), requires_grad=False)

        # Value head
        self.vfn = nn.Linear(fc_units[1], 1)

        self.fcnorm1 = nn.LayerNorm(fc_units[0])
        self.fcnorm2 = nn.LayerNorm(fc_units[1])
        self.reset_parameters()

    def reset_parameters(self):
        self.state_norm.weight.data.fill_(1.)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_mu.weight.data.uniform_(-3e-3, 3e-3) # Initialize mean ~ 0
        self.vfn.weight.data.uniform_(-3e-3, 3e-3)  # Initialize mean ~ 0

    def _latent_features_body(self, state):
        x = self.state_norm(state)
        x = F.relu(self.fcnorm1(self.fc1(x)))
        x = F.relu(self.fcnorm2(self.fc2(x)))
        return x

    def _latent_feature_to_pd(self, feats):
        mu = torch.tanh(self.fc_mu(feats))      # Mean
        # sig = F.softplus(self.std)            # std dev
        sig = torch.exp(self.logstd)
        return Normal(mu, sig)

    def _latent_feature_to_value(self, feats):
        return self.vfn(feats)


    def sa_logp_entropy_value(self, states, actions):
        '''
            Estimate the log prob of sate-actions
        :param states:
        :param actions:
        :return:
        '''
        feats = self._latent_features_body(states)
        pd = self._latent_feature_to_pd(feats)
        v = self._latent_feature_to_value(feats)
        return pd.log_prob(actions), pd.entropy(), v


    def state_entropy(self, states):
        '''
            Entropy of the distribution conditioned on this state
        :param states:
        :return:
        '''
        feats = self._latent_features_body(states)
        pd = self._latent_feature_to_pd(feats)
        return pd.entropy()



    def value(self, state):
        features = self._latent_features_body(state)
        return self._latent_feature_to_value(features)

    def forward(self, state, eval = False):
        '''
            Forward pass through the policy. Sample the action
        :param state:
        :return:
        '''
        features = self._latent_features_body(state)
        if eval:
            return torch.tanh(self.fc_mu(features))
        else:
            pd = self._latent_feature_to_pd(features)
            actions = pd.sample()
            return actions, pd.log_prob(actions), self._latent_feature_to_value(features)

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