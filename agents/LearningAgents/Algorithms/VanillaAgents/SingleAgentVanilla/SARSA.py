import numpy as np

class SARSAgent:
    def __init__(self, env, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99,
                 min_exploration_rate=0.01):

        self.state_size = env.state_size
        self.action_size = env.action_size

        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((self.state_size, self.action_size))

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
        # Epsilon-greedy action selection
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, next_action):
        # Update Q-table using the SARSA update rule
        next_estimate = self.q_table[next_state, next_action]
        self.q_table[state, action] = self.q_table[state, action] + \
                                      self.learning_rate * (
                                          reward + self.discount_rate * next_estimate - self.q_table[state, action])

        # Decay epsilon
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)