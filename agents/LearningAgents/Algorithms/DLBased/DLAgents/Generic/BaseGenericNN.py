import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from agents.LearningAgents.Algorithms.DLBased.Networks.GenericNN import FullyConnected
from abc import ABC, abstractmethod

class GenericNNAgent(ABC):
    def __init__(self, env, hidden_layers=None, learning_rate=0.01, agent_type="Deep"):
        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.hidden_layers = hidden_layers if hidden_layers else [128, 128]
        self.network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.agent_type = agent_type  # Specify that this is a 'Deep' agent

    @abstractmethod
    def decide_action(self, state):
        pass

    @abstractmethod
    def learn(self, states, actions, rewards, next_states, dones):
        pass
