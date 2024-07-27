# this file implements the ReplayMemory class

from collections import namedtuple


import random
from collections import deque
Transition = namedtuple('Transition', ('state', 'action1', 'action2', 'next_state', 'reward1', 'reward2'))
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def add(self, state, action1, action2, next_state, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action1, action2, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

