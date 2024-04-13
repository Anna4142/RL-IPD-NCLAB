# this file implements the ReplayMemory class

from collections import namedtuple


import random
from collections import deque
Transition = namedtuple('Transition', ('state', 'action1', 'action2', 'next_state', 'reward1', 'reward2'))
class ReplayBuffer:
    def __init__(self, capacity):
        """ Initialize the replay buffer with a maximum capacity.
        Args:
            capacity (int): The maximum number of tuples to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action1, action2, next_state, reward1, reward2):
        """ Add a new experience to the buffer using the Transition namedtuple.
        Args:
            state, action1, action2, next_state, reward1, reward2: Components of a transition.
        """
        self.buffer.append(Transition(state, action1, action2, next_state, reward1, reward2))

    def sample(self, batch_size):
        """ Sample a batch of experiences from the buffer.
        Args:
            batch_size (int): The number of experiences to sample.
        Returns:
            list: A batch of sampled experiences.
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def size(self):
        """ Return the current size of the buffer.
        Returns:
            int: The number of experiences currently stored in the buffer.
        """
        return len(self.buffer)

