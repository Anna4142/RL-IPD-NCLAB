import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from agents.LearningAgents.Algorithms.DLBased.Networks.GenericNN import FullyConnected
from agents.LearningAgents.Algorithms.DLBased.Networks.SpikingNN import FullyConnectedSNN
from abc import ABC, abstractmethod

class GenericNNAgent(ABC):
    def __init__(self, env, use_spiking_nn, hidden_layers=None, learning_rate=0.01, agent_type="Deep"):
        self.env = env
        self.use_spiking_nn = use_spiking_nn
        self.learning_rate = learning_rate
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.hidden_layers = hidden_layers if hidden_layers else [128, 128]

        if self.use_spiking_nn:
            print("Using spiking neural network.")
            self.network = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
        else:
            self.network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.agent_type = agent_type  # Specify that this is a 'Deep' agent

    @abstractmethod
    def decide_action(self, state):
        pass

    @abstractmethod
    def learn(self, states, actions, rewards, next_states, dones):
        pass

    def save_weights(self, filepath):
        torch.save({
            'network_state_dict': self.network.state_dict(),
        }, filepath)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()  # Set the network to evaluation mode
        print(f"Weights loaded from {filepath}")
