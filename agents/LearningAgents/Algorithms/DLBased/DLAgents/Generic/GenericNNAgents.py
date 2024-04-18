import torch
import torch.optim as optim
import random
import torch.nn.functional as F
from agents.LearningAgents.Algorithms.DLBased.Networks.GenericNN import FullyConnected
from agents.LearningAgents.Algorithms.DLBased.utils.replay import ReplayBuffer, Transition
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.BaseGenericNN import GenericNNAgent


class DQNAgent:
    def __init__(self, env, buffer_capacity=10000, agent_type="Deep"):
        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.agent_type = agent_type
        hidden_layers = env.hidden_layers if hasattr(env, 'hidden_layers') else [128, 128]
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
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):

        state = torch.tensor([state], dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        current_q_values = self.q_network(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_network(next_state).max(1)[0].detach()
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = F.mse_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_transition(self, state, action1, action2, next_state, reward, done):
        self.replay_buffer.add(state, action1, action2, next_state, reward, done)


class REINFORCEAgent(GenericNNAgent):
    def __init__(self, env, agent_type="Deep", **kwargs):
        super().__init__(env, agent_type=agent_type, **kwargs)
        self.log_probs = []
        self.rewards = []
        self.gamma=0.9

    def decide_action(self, state):
        state_tensor = torch.tensor([state], dtype=torch.float32)
        logits = self.network(state_tensor)
        action_probs = F.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action))
        return action.item()

    def learn(self, state, action, reward, next_state, done):
        if not self.log_probs:  # Check if no actions were taken
            return

        R = 0
        policy_loss = []
        discounted_rewards = []

        # Calculate discounted rewards in reverse
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate policy loss
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        if policy_loss:
            policy_loss = torch.cat(policy_loss).sum()
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
