import numpy as np
from envs.one_d_world.game import CustomEnv  # Assuming this is a multi-agent version of the environment

class JALAgent:
    def __init__(self, env, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        self.num_agents = 2
        self.state_size = env.state_size
        self.action_size = env.action_size ** self.num_agents  # Joint action space size
        self.env=env
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((self.state_size, self.action_size))

    def encode_state(self, history):
        # Convert the history of actions into a single integer for state representation
        # This method needs to be defined based on the specific environment's history format
        encoded = 0
        for action_pair in history:
            encoded = encoded * 4 + (action_pair[0] * 2) + action_pair[1]  # Example for a specific encoding
        return encoded

    def decode_joint_action(self, joint_action):
        # Decode the joint action index into individual actions for each agent
        actions = []
        for _ in range(self.num_agents):
            actions.append(joint_action % self.env.action_size)
            joint_action //= self.env.action_size
        return actions[::-1]  # Reverse to get the correct order

    def decide_action(self, state):
        # Choose a joint action based on epsilon-greedy policy
        if np.random.rand() < self.exploration_rate:
            joint_action = np.random.randint(self.action_size)  # Random joint action
        else:
            joint_action = np.argmax(self.q_table[state])  # Best known joint action from Q-table

        # Decode the chosen joint action into individual actions for each agent
        action1, action2 = self.decode_joint_action(joint_action)
        return action1, action2  # Return a tuple of actions

    def learn(self, state, joint_action, reward, next_state):
        # Update Q-table using the Q-learning update rule for the joint action
        next_max = np.max(self.q_table[next_state])
        td_target = reward + self.discount_rate * next_max
        td_error = td_target - self.q_table[state, joint_action]
        self.q_table[state, joint_action] += self.learning_rate * td_error

        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

# Example initialization
# env = CustomEnv(...)  # Initialize your multi-agent environment
# num_agents = 2  # For example, assuming there are two agents
# jal_agent = JALAgent(env, num_agents)
