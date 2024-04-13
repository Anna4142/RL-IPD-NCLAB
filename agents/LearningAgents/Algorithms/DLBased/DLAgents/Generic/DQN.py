# dqn_agent.py
import torch
import torch.optim as optim
import random
import torch.nn.functional as F
from agents.LearningAgents.Algorithms.DLBased.Networks.GenericNN import FullyConnected
from agents.LearningAgents.Algorithms.DLBased.utils.replay import ReplayBuffer, Transition

class DQNAgent:
    def __init__(self, env, buffer_capacity=10000):
        self.state_size = env.state_size
        self.action_size = env.action_size
        # Assuming env also provides a list of hidden layer sizes
        hidden_layers = env.hidden_layers if hasattr(env, 'hidden_layers') else [128, 128]  # default if not provided
        self.q_network = FullyConnected([self.state_size] + hidden_layers + [self.action_size])
        self.target_network = FullyConnected([self.state_size] + hidden_layers + [self.action_size])
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64

    def decide_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.layers[-1].out_features - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state)

        return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, next_action):
        if self.replay_buffer.size() < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.tensor(batch.state, dtype=torch.float32)
        actions = torch.tensor(batch.action1, dtype=torch.long)
        rewards = torch.tensor(batch.reward1, dtype=torch.float32)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32)
        dones = torch.tensor(batch.reward2, dtype=torch.float32)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.mse_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_transition(self, state, action1, action2, next_state, reward1, reward2):
        self.replay_buffer.add(state, action1, action2, next_state, reward1, reward2)
