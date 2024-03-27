import numpy as np
from envs.one_d_world.game import CustomEnv

class TDGammaAgent:
    def __init__(self, env, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.state_size = env.state_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate  # This is gamma in the TD update formula
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.value_table = np.zeros(self.state_size)  # Value table for storing the value of each state

    def encode_state(self, history):
        # Assuming history is a list of tuples (action1, action2), and each action can be 0 or 1
        # Convert the history of actions into a single integer for state representation
        encoded = 0
        for action1, action2 in history:
            encoded = encoded * 4 + (action1 * 2) + action2  # Example encoding strategy
        return encoded

    def decide_action(self, state):
        # Simplified action decision for demonstration; actual decision logic depends on the policy
        return np.random.randint(2)  # Assuming 2 actions, for simplicity

    def learn(self, state, reward, next_state):
        # Update value table using the TD(0) update rule
        td_target = reward + self.discount_rate * self.value_table[next_state]  # gamma is used here
        td_error = td_target - self.value_table[state]
        self.value_table[state] += self.learning_rate * td_error

        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)