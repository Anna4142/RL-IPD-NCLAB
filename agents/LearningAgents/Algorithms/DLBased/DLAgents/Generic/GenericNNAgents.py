import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
from agents.LearningAgents.Algorithms.DLBased.Networks.GenericNN import FullyConnected
from agents.LearningAgents.Algorithms.DLBased.utils.replay import ReplayBuffer, Transition
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.BaseGenericNN import GenericNNAgent
from agents.LearningAgents.Algorithms.DLBased.Networks.SpikingNN import FullyConnectedSNN


class DQNAgent:
    def __init__(self, env, use_spiking_nn, hidden_layers=None, lr=0.001, gamma=0.99):
        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.agent_type = "Deep"

        self.hidden_layers = hidden_layers if hidden_layers is not None else [128, 128]
        self.buffer_capacity = 10000
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64

        if use_spiking_nn:
            self.q_network = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            self.target_network = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
        else:
            self.q_network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
            self.target_network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def decide_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):
        state = np.array(state)  # Convert list of numpy arrays to a single numpy array

        state = torch.tensor([state], dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)
        print("op of spiking qn",self.q_network(state))
        current_q_values = self.q_network(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_network(next_state).max(1)[0].detach()
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = F.mse_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        #loss.backward(retain_graph=True)

        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_transition(self, state, action1, action2, next_state, reward, done):
        self.replay_buffer.add(state, action1, action2, next_state, reward, done)

    def save_weights(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
        }, filepath)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.q_network.eval()
        self.target_network.eval()
        print(f"Weights loaded from {filepath}")

class REINFORCEAgent(GenericNNAgent):
    def __init__(self, env, agent_type="Deep", use_spiking_nn=True, hidden_layers=None, lr=0.001, gamma=0.99):
            super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, learning_rate=learning_rate,
                         gamma=gamma)

            self.gamma = gamma
            self.hidden_layers = hidden_layers if hidden_layers is not None else [128, 128]

            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
            self.log_probs = []
            self.rewards = []

            if self.use_spiking_nn:
                print("Using spiking neural network.")
                self.network = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            else:
                self.network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])

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

    def save_weights(self, filepath):
        torch.save({
            'network_state_dict': self.network.state_dict(),
        }, filepath)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()
        print(f"Weights loaded from {filepath}")


class ActorCriticAgent(GenericNNAgent):
    def __init__(self, env, agent_type="Deep", use_spiking_nn=True, hidden_layers=None, learning_rate=0.001,
                 gamma=0.99):
            super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, learning_rate=learning_rate,
                         gamma=gamma)
            self.state_size = env.state_size
            self.action_size = env.action_size
            self.gamma = gamma
            self.hidden_layers = hidden_layers if hidden_layers is not None else [128, 128]
            self.learning_rate = learning_rate

            # Actor Network
            if self.use_spiking_nn:
                print("Using spiking neural network for actor.")
                self.actor = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            else:
                self.actor = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

            # Critic Network
            if self.use_spiking_nn:
                print("Using spiking neural network for critic.")
                self.critic = FullyConnectedSNN([self.state_size] + self.hidden_layers + [1])
            else:
                self.critic = FullyConnected([self.state_size] + self.hidden_layers + [1])
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def decide_action(self, state):
        state_tensor = torch.tensor([state], dtype=torch.float32)
        logits = self.actor(state_tensor)
        action_probs = F.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        self.current_log_prob = distribution.log_prob(action)  # Store log_prob internally
        return action.item()

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor([state], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state).detach()
        td_target = reward + self.gamma * next_value * (1 - done)
        td_error = td_target - value
        loss_critic = td_error.pow(2).mean()
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # Retrieve the stored log_prob
        log_prob = self.current_log_prob

        # Actor loss
        actor_loss = (-log_prob * td_error.detach()).mean()  # Ensure it's a scalar
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    def save_weights(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filepath)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor.eval()
        self.critic.eval()
        print(f"Weights loaded from {filepath}")
