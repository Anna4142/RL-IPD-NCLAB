# this file implements the ReplayMemory class

from collections import namedtuple
import random
import numpy as np
import os
import torch
Transition = namedtuple('Transition', ('state', 'action1', 'action2', 'next_state', 'reward1', 'reward2'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



    def save(self, model, data_buffer):
        """Saves the states and corresponding Q-values using the latest actions and rewards from DataBuffer."""
        # Assuming get_latest_state() retrieves the latest state
        latest_state = data_buffer.get_latest_state()
        latest_action1 = data_buffer.get_latest_action1()
        latest_action2 = data_buffer.get_latest_action2()
        # Assuming a way to combine or choose between action1 and action2 for the model
        # For simplicity, only action1 is used here
        action = torch.tensor([latest_action1], dtype=torch.long)

        # Convert state to tensor and add batch dimension (model expects batch dimension)
        state_tensor = torch.tensor([latest_state], dtype=torch.float32)

        # Compute Q-values for the latest state
        q_values = model(state_tensor).detach().numpy()

        # Save the state and Q-values to disk
        np.savez("latest_state_q_values", state=state_tensor.numpy(), q_values=q_values)
