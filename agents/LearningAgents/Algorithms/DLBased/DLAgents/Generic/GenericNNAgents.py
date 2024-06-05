import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
from agents.LearningAgents.Algorithms.DLBased.Networks.GenericNN import FullyConnected
from agents.LearningAgents.Algorithms.DLBased.utils.replay import ReplayBuffer, Transition
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.BaseGenericNN import GenericNNAgent
from agents.LearningAgents.Algorithms.DLBased.Networks.SpikingNN import FullyConnectedSNN
import os

class DQNAgent:

    def __init__(self, env, use_spiking_nn, hidden_layers=None, learning_rate=0.001, gamma=0.99,
                     use_mouse_hist=False, mouse_hist=None):
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
            self.use_mouse_hist = use_mouse_hist
            self.mouse_hist = mouse_hist if mouse_hist is not None else []
            self.current_step = 0
            self.forced_actions = []  # List to store forced actions
            self.expected_actions = []  # List to store expected actions based on Q-values

            if use_spiking_nn:
                self.q_network = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
                self.target_network = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            else:
                self.q_network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
                self.target_network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])

            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            self.replay_buffer = ReplayBuffer(capacity=self.buffer_capacity)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def decide_action(self, state):

            if self.use_mouse_hist and self.current_step < len(self.mouse_hist):
                action = self.mouse_hist[self.current_step]
                self.forced_actions.append(action)  # Store forced action
                if random.random() < self.epsilon:
                    action = random.randint(0, self.action_size - 1)
                else:
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = self.q_network(state)
                    action = q_values.argmax().item()
                self.expected_actions.append(
                    action)  # Store every action decided, whether forced or derived from policy
                print("expected", self.expected_actions)
                print("forced", self.forced_actions)


            else:
                if random.random() < self.epsilon:
                    action = random.randint(0, self.action_size - 1)
                else:
                    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = self.q_network(state)
                    action = q_values.argmax().item()
                self.expected_actions.append(
                    action)  # Store every action decided, whether forced or derived from policy

            return action


    def learn(self, state, action, reward, next_state, done):
        state = np.array(state)  # Convert list of numpy arrays to a single numpy array

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
        loss.backward(retain_graph=True)

        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_transition(self, state, action1, action2, next_state, reward, done):
        self.replay_buffer.add(state, action1, action2, next_state, reward, done)

    def save_weights(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

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

    def __init__(self, env, agent_type="Deep", use_spiking_nn=True, hidden_layers=None, learning_rate=0.001,
                     gamma=0.99, use_mouse_hist=False, mouse_hist=None):
            super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers,
                             learning_rate=learning_rate,
                             gamma=gamma)
            self.gamma = gamma
            self.use_mouse_hist = use_mouse_hist
            self.mouse_hist = mouse_hist if mouse_hist is not None else []
            self.current_step = 0
            self.forced_actions = []
            self.expected_actions = []

            if use_spiking_nn:
                self.network = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
                print("Using spiking neural network.")
            else:
                self.network = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])

            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            self.log_probs = []
            self.rewards = []

    def decide_action(self, state):
            if self.use_mouse_hist and self.current_step < len(self.mouse_hist):
                action = self.mouse_hist[self.current_step]
                self.forced_actions.append(action)
                state_tensor = torch.tensor([state], dtype=torch.float32)
                logits = self.network(state_tensor)
                action_probs = F.softmax(logits, dim=-1)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                self.log_probs.append(distribution.log_prob(action))
                self.expected_actions.append(action.item())  # Track expected action regardless of decision source
                self.current_step += 1
                print("forced",self.forced_actions)
                print("expected",self.expected_actions)
            else:
                state_tensor = torch.tensor([state], dtype=torch.float32)
                logits = self.network(state_tensor)
                action_probs = F.softmax(logits, dim=-1)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                self.log_probs.append(distribution.log_prob(action))
                self.expected_actions.append(action.item())  # Track expected action regardless of decision source

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


    def __init__(self, env, use_spiking_nn=True, hidden_layers=None, agent_type="Deep", learning_rate=0.01, gamma=0.99, use_mouse_hist=False,mouse_hist=None):
        super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, agent_type=agent_type,
                         learning_rate=learning_rate, gamma=gamma)
        self.gamma = gamma
        self.use_mouse_hist = use_mouse_hist
        self.mouse_hist = mouse_hist if mouse_hist is not None else []
        self.current_step = 0
        self.forced_actions = []
        self.expected_actions = []

        self.log_probs = []
        self.rewards = []

        # Actor and Critic Networks
        if use_spiking_nn:
            print("Using spiking neural network for both actor and critic.")
            self.actor = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            self.critic = FullyConnectedSNN([self.state_size] + self.hidden_layers + [1])
        else:
            self.actor = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
            self.critic = FullyConnected([self.state_size] + self.hidden_layers + [1])

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def decide_action(self, state):
            if self.use_mouse_hist and self.current_step < len(self.mouse_hist):
                action = self.mouse_hist[self.current_step]
                self.forced_actions.append(action)
                state_tensor = torch.tensor([state], dtype=torch.float32)
                logits = self.actor(state_tensor)
                action_probs = F.softmax(logits, dim=-1)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                self.current_log_prob = distribution.log_prob(action)  # Store log_prob internally
                self.expected_actions.append(action.item())  # Track expected action regardless of decision source
                self.current_step += 1
            else:
                state_tensor = torch.tensor([state], dtype=torch.float32)
                logits = self.actor(state_tensor)
                action_probs = F.softmax(logits, dim=-1)
                distribution = torch.distributions.Categorical(action_probs)
                action = distribution.sample()
                self.current_log_prob = distribution.log_prob(action)  # Store log_prob internally
                self.expected_actions.append(action.item())  # Track expected action regardless of decision source

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
        #self.optimizer_critic.step()

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



class TOMActorCriticAgent(GenericNNAgent):
    def __init__(self, env, use_spiking_nn=True, hidden_layers=None, agent_type="Deep", learning_rate=0.01, gamma=0.99, use_mouse_hist=False, mouse_hist=None):
        super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, agent_type=agent_type,
                         learning_rate=learning_rate, gamma=gamma)
        self.gamma = gamma
        self.use_mouse_hist = use_mouse_hist
        self.mouse_hist = mouse_hist if mouse_hist is not None else []
        self.current_step = 0
        self.forced_actions = []
        self.expected_actions = []
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []
        self.current_log_prob = None


        # TOM Actor and Critic Networks
        if use_spiking_nn:
            print("Using spiking neural network for both actor and critic.")
            self.self_critic = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            self.other_critic = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            self.actor = FullyConnectedSNN([self.state_size + self.state_size] + self.hidden_layers + [self.action_size])
        else:
            self.self_critic = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
            self.other_critic = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
            self.actor = FullyConnected([self.state_size + self.state_size] + self.hidden_layers + [self.action_size])

        self.optimizer_self_critic = optim.Adam(self.self_critic.parameters(), lr=learning_rate)
        self.optimizer_other_critic = optim.Adam(self.other_critic.parameters(), lr=learning_rate)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)

    def one_hot_encode_action(self, action_index, action_space_size):
        """
        Encodes an action index into a one-hot vector.

        Parameters:
            action_index (int): The index of the action to encode.
            action_space_size (int): Total number of possible actions.

        Returns:
            torch.Tensor: A one-hot encoded tensor representing the action.
        """
        # Create a zero-filled numpy array with length equal to the number of actions
        one_hot_vector = np.zeros(action_space_size, dtype=np.float32)

        # Set the position corresponding to the action index to 1
        one_hot_vector[action_index] = 1

        # Convert the numpy array to a torch tensor
        return one_hot_vector
    def decide_action(self, state, state_others):
        # Convert states to tensors
        state_tensor = torch.tensor([state], dtype=torch.float32)


        state_others_tensor = torch.tensor([state_others], dtype=torch.float32)

        # Combine state tensors for input to the neural network
        combined_state = torch.cat([state_tensor, state_others_tensor], dim=-1)

        # Pass the combined state through the actor network to get logits
        logits = self.actor(combined_state)
        action_probs = F.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(action_probs)

        if self.use_mouse_hist and self.current_step < len(self.mouse_hist):
            # Forced action from historical mouse data
            action = self.mouse_hist[self.current_step]
            # Compute log probability for the forced action
            # It's crucial that the forced action is within the range of possible actions as per the distribution
            self.current_log_prob = distribution.log_prob(torch.tensor([action]))  # Ensure this is a tensor
        else:
            # Sample an action according to the policy distribution
            action = distribution.sample()
            # Store the log probability of the sampled action
            self.current_log_prob = distribution.log_prob(action)

        # Append log probability and the chosen action to their respective lists for tracking
        self.log_probs.append(self.current_log_prob)
        self.expected_actions.append(action.item())  # Record the action as an integer
        self.current_step += 1  # Increment the step counter

        return action.item()

    def learn(self, state, state_others, action, reward, next_state, next_state_others, done):
        state = torch.tensor([state], dtype=torch.float32)
        action_space_size=len(state)



        state_others = torch.tensor([state_others], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)

        next_state_others= torch.tensor([next_state_others], dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        print("Shape of state_tensor:", state.shape)
        print("Shape of state_others_tensor:", state_others.shape)

        done = torch.tensor([done], dtype=torch.float32)


        # Self Critic update

        self_value = self.self_critic(state)
        next_self_value = self.self_critic(next_state)

        self_td_target = reward + self.gamma * next_self_value * (1 - done)
        self_td_error =F.mse_loss(self_td_target, self_value)
        loss_self_critic = self_td_error
        self.optimizer_self_critic.zero_grad()
        loss_self_critic.backward(retain_graph=True)
        self.optimizer_self_critic.step()

        # Other Critic update
        other_value = self.other_critic(state_others)
        next_other_value = self.other_critic(next_state_others).detach()
        other_td_target = reward + self.gamma * next_other_value * (1 - done)
        other_td_error = other_td_target - other_value
        loss_other_critic = other_td_error.pow(2).mean()

        # Internal reward for Other Critic based on matching actions
        #other_action_probs = F.softmax(other_value, dim=-1)
        #other_action_dist = torch.distributions.Categorical(other_action_probs)
        #other_predicted_action = other_action_dist.sample()
        #internal_reward = torch.tensor([1.0 if other_predicted_action == other_action else -1.0], dtype=torch.float32)
        #loss_other_critic += (internal_reward.pow(2)).mean()

        self.optimizer_other_critic.zero_grad()
        loss_other_critic.backward(retain_graph=True)
        self.optimizer_other_critic.step()

        combined_td_error = self_td_error + other_td_error  # Summing both TD errors for the actor's update

        # Retrieve the stored log_prob
        log_prob = self.current_log_prob

        # Actor loss using combined TD error
        actor_loss = (-log_prob * combined_td_error.detach()).mean()  # Ensure it's a scalar
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
    def save_weights(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'self_critic_state_dict': self.self_critic.state_dict(),
            'other_critic_state_dict': self.other_critic.state_dict(),
        }, filepath)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.self_critic.load_state_dict(checkpoint['self_critic_state_dict'])
        self.other_critic.load_state_dict(checkpoint['other_critic_state_dict'])
        self.actor.eval()
        self.self_critic.eval()
        self.other_critic.eval()
        print(f"Weights loaded from {filepath}")