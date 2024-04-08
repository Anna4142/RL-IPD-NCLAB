import numpy as np
from envs.one_d_world.game import CustomEnv

class TDLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99,
                 min_exploration_rate=0.01):

        self.state_size = env.state_size  # Number of states in the environment
        self.env=env
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.value_table = np.zeros(self.state_size)  # Value table for storing the value of each state

    def calculate_state_size(self, env):
        # Assuming history is a list of tuples (action1, action2) and each action can be 0 or 1
        # Calculate the total number of possible states based on history length
        return 2 ** (env.history_length * 2)  # Each position in history can be one of 4 states

    def encode_state(self, history):
        # Convert the history of actions into a single integer for state representation
        encoded = 0
        for action1, action2 in history:
            encoded = encoded * 4 + (action1 + 1) * 2 + (action2 + 1)  # Shift and add new action pair
        return encoded

    def decide_action(self, state):
        # Simplified action decision for demonstration; actual decision logic depends on the policy
        return np.random.randint(self.env.action_size)  # Random action

    def learn(self, state, reward, next_state):
        # Update value table using the TD(0) update rule
        td_target = reward + self.discount_rate * self.value_table[next_state]
        td_error = td_target - self.value_table[state]
        self.value_table[state] += self.learning_rate * td_error

        # Optionally, decay epsilon here if your policy depends on exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
