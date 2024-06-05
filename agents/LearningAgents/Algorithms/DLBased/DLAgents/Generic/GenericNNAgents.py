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
from torch.distributions import Categorical

class DQNAgent:

    def __init__(self, env, use_spiking_nn, hidden_layers=None, learning_rate=0.001, gamma=0.99,
                      mouse_hist=None,use_mouse_hist=False,human_hist=False, use_human_hist=None):
            self.env = env
            self.state_size = env.state_size
            self.action_size = env.action_size
            self.agent_type = "Deep"
            self.use_human=use_human_hist
            self.human_actions=human_hist
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
        if (self.use_human and self.current_step < len(self.human_actions)) or (self.use_mouse_hist and self.current_step < len(self.mouse_hist)):
            if self.use_human and self.current_step < len(self.human_actions):
                action = self.human_actions[self.current_step]
            else:
                action = self.mouse_hist[self.current_step]

            self.forced_actions.append(action)  # Store forced action

            if random.random() < self.epsilon:
                action = random.randint(0, self.action_size - 1)
            else:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state)
                action = q_values.argmax().item()

            self.expected_actions.append(action)
            print("expected", self.expected_actions)
            print("forced", self.forced_actions)

        else:
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_size - 1)
            else:
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state)
                action = q_values.argmax().item()
            self.expected_actions.append(action)

        self.current_step += 1
        return action

    def learn(self, state, action, reward, next_state, done):
        # Store the new experience in the replay buffer
        self.store_transition(state, action, action, next_state, reward, done)

        # Only start learning when we have enough samples for a batch
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        print(f"Batch type: {type(batch)}")
        print(f"Batch length: {len(batch)}")
        print(f"First item in batch: {batch[0]}")

        # Separate the batch into its components
        batch_state, batch_action1, batch_action2, batch_next_state, batch_reward, batch_done = zip(*batch)

        print(f"batch_state type: {type(batch_state)}")
        print(f"First state in batch: {batch_state[0]}")

        # Convert batch data to tensors, handling potential inconsistencies
        try:
            batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32)
            batch_action = torch.tensor(np.array(batch_action1), dtype=torch.long)
            batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32)
            batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32)
            # Convert 'done' to a consistent format (use 1 for done, 0 for not done)
            batch_done = torch.tensor([1 if isinstance(d, tuple) else d for d in batch_done], dtype=torch.float32)
        except Exception as e:
            print(f"Error converting to tensor: {e}")
            return

        print(f"Tensor shapes: state {batch_state.shape}, action {batch_action.shape}, "
              f"reward {batch_reward.shape}, next_state {batch_next_state.shape}, done {batch_done.shape}")

        # Compute Q values
        current_q_values = self.q_network(batch_state).gather(1, batch_action.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_network(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + self.gamma * next_q_values * (1 - batch_done)

        # Compute loss
        loss = F.mse_loss(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(f"Loss: {loss.item()}")

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

class REINFORCEAgent:
    def __init__(self, env, use_spiking_nn=True, hidden_layers=None, learning_rate=0.001,
                     gamma=0.99, mouse_hist=None,use_mouse_hist=False,use_human_hist=False, human_hist=None):
            self.env = env
            self.mouse_hist = mouse_hist
            self.use_mouse_hist = use_mouse_hist
            self.state_size = env.state_size
            self.action_size = env.action_size
            self.agent_type = "Deep"
            self.use_human_hist = use_human_hist
            self.human_hist = human_hist if human_hist is not None else []
            self.hidden_layers = hidden_layers if hidden_layers is not None else [128, 128]
            self.gamma = gamma
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

    def decide_action(self, state, human_decision=None):
            state_tensor = torch.tensor([state], dtype=torch.float32)
            logits = self.network(state_tensor)
            action_probs = F.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(action_probs)

            if human_decision is not None:
                action = torch.tensor([human_decision])
                self.forced_actions.append(human_decision)
            elif self.use_human_hist and self.current_step < len(self.human_hist):
                action = torch.tensor([self.human_hist[self.current_step]])
                self.forced_actions.append(self.human_hist[self.current_step])
            else:
                action = distribution.sample()

            self.log_probs.append(distribution.log_prob(action))
            self.expected_actions.append(action.item())
            self.current_step += 1

            print("forced", self.forced_actions)
            print("expected", self.expected_actions)

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
    def __init__(self, env, use_spiking_nn=True, hidden_layers=None, agent_type="Deep",
                     learning_rate=0.01, gamma=0.99,mouse_hist=None,use_mouse_hist=False, use_human_hist=False, human_hist=None):
            super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers,
                             agent_type=agent_type, learning_rate=learning_rate, gamma=gamma)
            self.gamma = gamma
            self.mouse_hist=mouse_hist
            self.use_mouse_hist=use_mouse_hist
            self.use_human_hist = use_human_hist
            self.human_hist = human_hist if human_hist is not None else []
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

    def decide_action(self, state, human_decision=None):
            state_tensor = torch.tensor([state], dtype=torch.float32)
            logits = self.actor(state_tensor)
            action_probs = F.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(action_probs)

            if human_decision is not None:
                action = torch.tensor([human_decision])
                self.forced_actions.append(human_decision)
            elif self.use_human_hist and self.current_step < len(self.human_hist):
                action = torch.tensor([self.human_hist[self.current_step]])
                self.forced_actions.append(self.human_hist[self.current_step])
            else:
                action = distribution.sample()

            self.current_log_prob = distribution.log_prob(action)  # Store log_prob internally
            self.expected_actions.append(action.item())  # Track expected action regardless of decision source
            self.current_step += 1

            print("forced", self.forced_actions)
            print("expected", self.expected_actions)

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

class TOMActorCriticAgent(GenericNNAgent):
    def __init__(self, env, use_spiking_nn=False, hidden_layers=None, agent_type="Deep", learning_rate=0.01,
                     gamma=0.99,
                     use_mouse_hist=False, mouse_hist=None, use_human_hist=False, human_hist=None):
            super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, agent_type=agent_type,
                             learning_rate=learning_rate, gamma=gamma)
            self.gamma = gamma
            self.use_mouse_hist = use_mouse_hist
            self.mouse_hist = mouse_hist if mouse_hist is not None else []
            self.use_human_hist = use_human_hist
            self.human_hist = human_hist if human_hist is not None else []
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
                self.actor = FullyConnectedSNN(
                    [self.state_size + self.state_size] + self.hidden_layers + [self.action_size])
            else:
                self.self_critic = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
                self.other_critic = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
                self.actor = FullyConnected(
                    [self.state_size + self.state_size] + self.hidden_layers + [self.action_size])

            self.optimizer_self_critic = optim.Adam(self.self_critic.parameters(), lr=learning_rate)
            self.optimizer_other_critic = optim.Adam(self.other_critic.parameters(), lr=learning_rate)
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)

    def decide_action(self, state, state_others, human_decision=None):
            state_tensor = torch.tensor([state], dtype=torch.float32)
            state_others_tensor = torch.tensor([state_others], dtype=torch.float32)
            combined_state = torch.cat([state_tensor, state_others_tensor], dim=-1)
            logits = self.actor(combined_state)
            action_probs = F.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(action_probs)

            if human_decision is not None:
                action = human_decision
            elif self.use_human_hist and self.current_step < len(self.human_hist):
                action = self.human_hist[self.current_step]
            elif self.use_mouse_hist and self.current_step < len(self.mouse_hist):
                action = self.mouse_hist[self.current_step]
            else:
                action = distribution.sample().item()

            self.forced_actions.append(action)
            self.current_log_prob = distribution.log_prob(torch.tensor([action]))
            self.log_probs.append(self.current_log_prob)
            self.expected_actions.append(action)
            self.current_step += 1

            return action
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
    '''''
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
    '''''
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



class SoftActorCriticAgent:
    def __init__(self, env, use_spiking_nn=False, hidden_layers=None, learning_rate=0.01, gamma=0.99, tau=0.005, alpha=0.2, use_mouse_hist=False, mouse_hist=None):
        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.agent_type = "Deep"

        self.hidden_layers = hidden_layers if hidden_layers is not None else [128, 128]
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.use_mouse_hist = use_mouse_hist
        self.mouse_hist = mouse_hist if mouse_hist is not None else []
        self.current_step = 0
        self.forced_actions = []
        self.expected_actions = []

        input_size = self.state_size + 1  # State size + Action size

        if use_spiking_nn:
            print("Using spiking neural network for both actor and critic.")
            self.actor = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            self.self_critic = FullyConnectedSNN([input_size] + self.hidden_layers + [1])  # Adjusted to single output unit
            self.target_critic = FullyConnectedSNN([input_size] + self.hidden_layers + [1])  # Adjusted to single output unit
        else:
            self.actor = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
            self.self_critic = FullyConnected([input_size] + self.hidden_layers + [1])  # Adjusted to single output unit
            self.target_critic = FullyConnected([input_size] + self.hidden_layers + [1])  # Adjusted to single output unit

        self.target_critic.load_state_dict(self.self_critic.state_dict())
        self.target_critic.eval()
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.self_critic.parameters(), lr=learning_rate)

    def decide_action(self, state):
        if self.use_mouse_hist and self.current_step < len(self.mouse_hist):
            action = self.mouse_hist[self.current_step]
            self.forced_actions.append(action)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = self.actor(state)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample().item()

        self.expected_actions.append(action)
        self.current_step += 1

        return action

    def learn(self, state, action1, reward1, next_state, done):
        # Convert inputs to tensors
        state = torch.tensor([state], dtype=torch.float32)
        action1 = torch.tensor([action1], dtype=torch.long)
        reward1 = torch.tensor([reward1], dtype=torch.float32)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        # Print tensor shapes for debugging
        print(f"State shape: {state.shape}")
        print(f"Action1 shape: {action1.shape}")
        print(f"Next state shape: {next_state.shape}")

        # Update critic
        with torch.no_grad():
            next_logits = self.actor(next_state)
            next_action_dists = Categorical(logits=next_logits)
            next_actions = next_action_dists.sample().unsqueeze(1)
            next_log_probs = next_action_dists.log_prob(next_actions.squeeze(1)).unsqueeze(1)
            print(f"Next actions shape: {next_actions.shape}")
            print(f"Next log_probs shape: {next_log_probs.shape}")
            concatenated_next = torch.cat([next_state, next_actions.float()], dim=1)
            print(f"Concatenated next state and next actions shape: {concatenated_next.shape}")
            next_q_values = self.target_critic(concatenated_next).squeeze(-1)
            print(f"Next q_values shape: {next_q_values.shape}")
            expected_q_values = reward1 + self.gamma * (1 - done) * (next_q_values - self.alpha * next_log_probs.squeeze(-1))
            print(f"Expected q_values shape: {expected_q_values.shape}")

        concatenated_state_action = torch.cat([state, action1.float().unsqueeze(1)], dim=1)
        print(f"Concatenated state and action1 shape: {concatenated_state_action.shape}")
        current_q_values = self.self_critic(concatenated_state_action).squeeze(-1)
        print(f"Current q_values shape: {current_q_values.shape}")

        critic_loss = F.mse_loss(current_q_values, expected_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        logits = self.actor(state)
        action_dists = Categorical(logits=logits)
        actions = action_dists.sample().unsqueeze(1)
        log_probs = action_dists.log_prob(actions.squeeze(1)).unsqueeze(1)
        concatenated_state_action_actor = torch.cat([state, actions.float()], dim=1)
        print(f"Concatenated state and actions for actor shape: {concatenated_state_action_actor.shape}")
        current_q_values = self.self_critic(concatenated_state_action_actor).squeeze(-1)
        print(f"Current q_values for actor shape: {current_q_values.shape}")
        actor_loss = (self.alpha * log_probs.squeeze(-1) - current_q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_critic.parameters(), self.self_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action1, action2, reward1, reward2, next_state, done):
        self.replay_buffer.add(state, action1, action2, next_state, reward1, reward2)

    def save_weights(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'self_critic_state_dict': self.self_critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
        }, filepath)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.self_critic.load_state_dict(checkpoint['self_critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.eval()
        self.self_critic.eval()
        self.target_critic.eval()
        print(f"Weights loaded from {filepath}")

class A2CAgent(GenericNNAgent):
    def __init__(self, env, use_spiking_nn=False, hidden_layers=None, agent_type="Deep", learning_rate=0.001,
                     gamma=0.99, entropy_coef=0.01, value_loss_coef=0.5, use_mouse_hist=False, mouse_hist=None,
                     use_human_hist=False, human_hist=None):
            super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, agent_type=agent_type,
                             learning_rate=learning_rate, gamma=gamma)
            self.entropy_coef = entropy_coef
            self.value_loss_coef = value_loss_coef
            self.use_mouse_hist = use_mouse_hist
            self.mouse_hist = mouse_hist if mouse_hist is not None else []
            self.use_human = use_human_hist
            self.human_actions = human_hist if human_hist is not None else []
            self.current_step = 0
            self.forced_actions = []
            self.expected_actions = []

            # Actor and Critic Networks
            if use_spiking_nn:
                print("Using spiking neural network for both actor and critic.")
                self.actor = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
                self.critic = FullyConnectedSNN([self.state_size] + self.hidden_layers + [1])
            else:
                self.actor = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
                self.critic = FullyConnected([self.state_size] + self.hidden_layers + [1])

            self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                        lr=learning_rate)

            # Initialize memory
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.values = []

    def decide_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs = F.softmax(self.actor(state), dim=-1)
            value = self.critic(state)
            dist = Categorical(action_probs)

            if (self.use_human and self.current_step < len(self.human_actions)) or (
                    self.use_mouse_hist and self.current_step < len(self.mouse_hist)):
                if self.use_human and self.current_step < len(self.human_actions):
                    action = torch.tensor([self.human_actions[self.current_step]])
                else:
                    action = torch.tensor([self.mouse_hist[self.current_step]])
                self.forced_actions.append(action.item())
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.expected_actions.append(action.item())
            self.current_step += 1

            return action.item()

    def learn(self, state, action, reward, next_state, done):
        self.rewards.append(torch.tensor([reward], dtype=torch.float))

        if done:
            self.update()

    def update(self):
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        rewards = torch.cat(self.rewards)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze(-1)

        # Compute returns and advantages
        returns = self.compute_returns(rewards)
        advantages = returns - values

        # Compute actor (policy) loss
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Compute critic (value) loss
        critic_loss = F.mse_loss(values, returns)

        # Compute entropy to encourage exploration
        action_probs = F.softmax(self.actor(states), dim=-1)
        dist = Categorical(action_probs)
        entropy = dist.entropy().mean()

        # Total loss
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

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

class PPOAgent(GenericNNAgent):
    def __init__(self, env, use_spiking_nn=False, hidden_layers=None, agent_type="Deep", learning_rate=0.001,
                     gamma=0.99, clip_epsilon=0.2, use_mouse_hist=False, mouse_hist=None,
                     use_human_hist=False, human_hist=None):
            super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, agent_type=agent_type,
                             learning_rate=learning_rate, gamma=gamma)
            self.clip_epsilon = clip_epsilon
            self.use_mouse_hist = use_mouse_hist
            self.mouse_hist = mouse_hist if mouse_hist is not None else []
            self.use_human_hist = use_human_hist
            self.human_hist = human_hist if human_hist is not None else []
            self.current_step = 0
            self.forced_actions = []
            self.expected_actions = []

            # Actor and Critic Networks
            if use_spiking_nn:
                print("Using spiking neural network for both actor and critic.")
                self.actor = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
                self.critic = FullyConnectedSNN([self.state_size] + self.hidden_layers + [1])
            else:
                self.actor = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
                self.critic = FullyConnected([self.state_size] + self.hidden_layers + [1])

            self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                        lr=learning_rate)

    def decide_action(self, state, human_decision=None):
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs = F.softmax(self.actor(state), dim=-1)
            dist = Categorical(action_probs)

            if human_decision is not None:
                action = human_decision
                self.forced_actions.append(action)
            elif self.use_human_hist and self.current_step < len(self.human_hist):
                action = self.human_hist[self.current_step]
                self.forced_actions.append(action)
            elif self.use_mouse_hist and self.current_step < len(self.mouse_hist):
                action = self.mouse_hist[self.current_step]
                self.forced_actions.append(action)
            else:
                action = dist.sample().item()

            self.last_state = state
            self.last_action = torch.tensor([action])
            self.last_log_prob = dist.log_prob(self.last_action)
            self.expected_actions.append(action)
            self.current_step += 1

            return action

    def learn(self, state, action, reward, next_state, done):
        state = self.last_state
        action = self.last_action
        old_log_prob = self.last_log_prob

        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        if isinstance(done, tuple):
            done = done[0]  # Assume the first element of the tuple is the done flag
        done = torch.FloatTensor([float(done)])

        # Compute values
        value = self.critic(state)
        next_value = self.critic(next_state)

        # Compute returns and advantages
        returns = reward + self.gamma * next_value * (1 - done)
        advantage = returns - value

        # Compute new action probabilities and values
        new_action_probs = F.softmax(self.actor(state), dim=-1)
        new_dist = Categorical(new_action_probs)
        new_log_prob = new_dist.log_prob(action)
        entropy = new_dist.entropy().mean()

        # Compute ratio and surrogate losses
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        # Compute critic loss
        critic_loss = F.mse_loss(value, returns.detach())

        # Compute total loss and perform optimization step
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_weights(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filepath)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.state_dict(checkpoint['critic_state_dict'])
        self.actor.eval()
        self.critic.eval()
        print(f"Weights loaded from {filepath}")


class ACERAgent(GenericNNAgent):
    def __init__(self, env, use_spiking_nn=False, hidden_layers=None, agent_type="Deep", learning_rate=0.001,
                 gamma=0.99,
                 buffer_size=10000, batch_size=64, trust_region_delta=1, c=10.0, use_mouse_hist=False, mouse_hist=None):
        super().__init__(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, agent_type=agent_type,
                         learning_rate=learning_rate, gamma=gamma)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.trust_region_delta = trust_region_delta
        self.c = c
        self.use_mouse_hist = use_mouse_hist
        self.mouse_hist = mouse_hist if mouse_hist is not None else []
        self.current_step = 0
        self.forced_actions = []
        self.expected_actions = []

        # Actor and Critic Networks
        if use_spiking_nn:
            print("Using spiking neural network for both actor and critic.")
            self.actor = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
            self.critic = FullyConnectedSNN([self.state_size] + self.hidden_layers + [self.action_size])
        else:
            self.actor = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])
            self.critic = FullyConnected([self.state_size] + self.hidden_layers + [self.action_size])

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def decide_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = F.softmax(self.actor(state), dim=-1)
        dist = Categorical(action_probs)

        if self.use_mouse_hist and self.current_step < len(self.mouse_hist):
            action = self.mouse_hist[self.current_step]
            self.forced_actions.append(action)
        else:
            action = dist.sample()

        self.last_state = state
        self.last_action = action
        self.last_log_prob = dist.log_prob(action)
        self.expected_actions.append(action.item())
        self.current_step += 1

        return action.item()

    def learn(self, state, action1, action2, next_state, reward1):
        # Store the transition in replay buffer
        self.replay_buffer.add(state, action1, action2, next_state, reward1,0)

        # Sample a batch from the replay buffer
        if self.replay_buffer.size() < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state))
        action1_batch = torch.LongTensor(batch.action1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        reward1_batch = torch.FloatTensor(batch.reward1)
        reward1_batch = torch.FloatTensor(batch.reward1)
        actual_batch_size = reward1_batch.size(0)
        print(f"Actual batch size: {actual_batch_size}")
        print(f"Expected batch size: {self.batch_size}")

        if actual_batch_size != self.batch_size:
            print(
                f"Warning: Actual batch size ({actual_batch_size}) does not match expected batch size ({self.batch_size})")
            # Adjust other tensors to match the actual batch size
            state_batch = state_batch[:actual_batch_size]
            action1_batch = action1_batch[:actual_batch_size]
            next_state_batch = next_state_batch[:actual_batch_size]



        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.critic(state_batch).gather(1, action1_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states
        next_state_values = self.critic(next_state_batch).detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward1_batch

        # Compute actor loss
        log_probs = F.log_softmax(self.actor(state_batch), dim=-1)
        action_log_probs = log_probs.gather(1, action1_batch.unsqueeze(1))
        advantages = expected_state_action_values.unsqueeze(1) - state_action_values
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        # Compute critic loss
        critic_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Compute importance weights
        with torch.no_grad():
            old_log_probs = F.log_softmax(self.actor(state_batch), dim=-1).gather(1, action1_batch.unsqueeze(1))
        importance_weights = torch.exp(action_log_probs - old_log_probs)

        # Compute trust region loss
        trust_region_loss = self.c * F.mse_loss(importance_weights, torch.ones_like(importance_weights))

        # Combine losses
        loss = actor_loss + critic_loss + trust_region_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

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