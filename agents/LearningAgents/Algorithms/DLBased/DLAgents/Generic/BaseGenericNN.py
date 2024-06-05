import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from agents.LearningAgents.Algorithms.DLBased.Networks.GenericNN import FullyConnected
from agents.LearningAgents.Algorithms.DLBased.Networks.SpikingNN import FullyConnectedSNN
from abc import ABC, abstractmethod

class GenericNNAgent(ABC):
    def __init__(self, env, use_spiking_nn=True, hidden_layers=None, agent_type="Deep", learning_rate=0.01, gamma=0.99, mouse_hist=None):
        self.env = env
        self.use_spiking_nn = use_spiking_nn
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.hidden_layers = hidden_layers if hidden_layers else [128, 128]
        self.mouse_hist = mouse_hist  # List of predetermined actions
        self.current_step = 0  # To track the current index in mouse_hist

        if self.use_spiking_nn:
            print("Using spiking neural network.")
            self.network = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
        else:
            self.network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.agent_type = agent_type  # Specify that this is a 'Deep' agent

    @abstractmethod
    def decide_action(self, state):
        # Check if using mouse history and if the current step is within the bounds of the mouse_hist list
        if self.mouse_hist and self.current_step < len(self.mouse_hist):
            action = self.mouse_hist[self.current_step]
            self.current_step += 1  # Increment to move to the next action in the subsequent call
            return action
        else:
            # Implement specific action decision logic in subclasses
            raise NotImplementedError("Subclasses must implement this method")

    def reset(self):
        # Reset current step for scenarios where the agent is reused without reinitialization
        self.current_step = 0
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
