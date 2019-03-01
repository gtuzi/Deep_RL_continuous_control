'''
/* Copyright (C) 2019 Gerti Tuzi - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MY license, which unfortunately won't be written
 * because no one would care to read it, and in the best of scenarios
 * it will only be used for ego pampering needs (yes, you can do it better)
 */
'''

import numpy as np
from collections import namedtuple, deque
import random
import torch


class Experience:
    def __init__(self, **kwargs):
        self._populate_(**kwargs)

    def _populate_(self, **kwargs):
        # Minimal experience parameters: sarsa and done
        assert 's' in kwargs
        assert 'a' in kwargs
        assert 'r' in kwargs
        assert 'sp' in kwargs
        assert 'done' in kwargs

        self.s = None
        self.a = None
        self.r = None
        self.sp = None
        self.logp = None
        self.v = None
        self.A = None
        self.done = None
        self.rFuture = None
        self.entropy = None

        for k,v in kwargs.items():
            if k == 's':                # state
                self.s = v
            elif k == 'a':              # action
                self.a = v
            elif k == 'r':              # reward
                self.r = v
            elif k == 'sp':             # next state
                self.sp = v
            elif k == 'logp':           # action log prob
                self.logp = v
            elif k == 'v':              # value
                self.v = v
            elif k == 'A':              # advantage
                self.A = v
            elif k == 'done':           # last episodic sample
                self.done = v
            elif k == 'rFuture':        # future reward Sum(r: t-->T)
                self.rFuture = v
            elif k == 'entropy':        # action entropy = policy(state)
                self.entropy = v


class ReplayBuffer:
    """Fixed-size uniformly sampled replay buffer"""

    def __init__(self, buffer_size, batch_size, seed, device = 'cpu'):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)  #
        self.experience = Experience
        self.seed = random.seed(seed)
        self.device = device


    def add(self, experience, attrs = ['s', 'a', 'r', 'sp', 'done']):
        """Add a new experience to trajectory."""
        n_agents = experience.s.shape[0]

        if n_agents > 1:
            res = {}
            for att in attrs:
                res[att] = np.vsplit(getattr(experience, att), n_agents)
            for i in range(n_agents):
                _res_agent = {_att:_val[i] for _att, _val in res.items()} # Pull each sample for each agent
                e = self.experience(**_res_agent)
                self.memory.append(e)
        else:
            self.memory.append(experience)

    def sample(self, attrs = ['s', 'a', 'r', 'sp', 'done']):
        """Randomly sample_set a batch of experiences from buffer."""
        experiences = random.sample(self.memory, k=self.batch_size)

        def _get_samp_feat(exp, attr_str):
            res = []
            for e in exp:
                if getattr(e, attr_str) is not None:
                    res.append(np.vstack(getattr(e, attr_str)))

            return torch.from_numpy(np.vstack(res)).float().to(self.device)

        out = namedtuple('out', attrs)
        for att in attrs:
            setattr(out, att, _get_samp_feat(experiences, att))

        return out


    def __len__(self):
        """Return the current size of internal trajectory."""
        return len(self.memory)

    def clear(self):
        self.memory.clear()
