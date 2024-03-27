import numpy as np
from envs.one_d_world.game import CustomEnv

class NashQAgent:
    def __init__(self,env,num_states, num_actions, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0,
                 exploration_decay=0.99, min_exploration_rate=0.01):
        self.env=env
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Initialize Q-tables for each agent
        self.q_table1 = np.zeros((num_states, num_actions, num_actions))  # Agent 1's Q-table
        self.q_table2 = np.zeros((num_states, num_actions, num_actions))  # Agent 2's Q-table

    def decide_actions(self, state):
        if np.random.rand() < self.exploration_rate:
            # Explore: select random actions for both agents
            action1 = np.random.randint(self.num_actions)
            action2 = np.random.randint(self.num_actions)
        else:
            # Exploit: select actions based on Nash equilibrium calculation
            action1, action2 = self.calculate_nash_equilibrium(state)
        return action1, action2

    def calculate_nash_equilibrium(self, state):
        # Assuming a 2x2 game with actions 0 and 1 for both agents
        num_actions = 2

        # Placeholder for the utility/payoff matrix for each action combination
        # For a real scenario, these would be derived from the environment's dynamics or the Q-values
        utility_matrix_1 = np.array(self.env.payout_matrix)[:, :, 0]  # Utility for agent 1
        utility_matrix_2 = np.array(self.env.payout_matrix)[:, :, 1]  # Utility for agent 2

        best_response_1 = np.zeros(num_actions)
        best_response_2 = np.zeros(num_actions)

        # Calculate best response for Agent 1 given Agent 2's actions
        for a2 in range(num_actions):
            best_response_1[a2] = np.argmax(utility_matrix_1[:, a2])

        # Calculate best response for Agent 2 given Agent 1's actions
        for a1 in range(num_actions):
            best_response_2[a1] = np.argmax(utility_matrix_2[a1, :])

        # Find the action pair where both agents' responses align
        for a1 in range(num_actions):
            for a2 in range(num_actions):
                if best_response_1[a2] == a1 and best_response_2[a1] == a2:
                    # This (a1, a2) is a Nash equilibrium (simplified assumption)
                    return (int(a1), int(a2))

        # Default to random actions if no equilibrium is found (highly unlikely in this simplified scenario)
        return (np.random.randint(num_actions), np.random.randint(num_actions))

    def learn(self, state, action1, action2, reward1, reward2, next_state):
        # Best response for Agent 1 given Agent 2's action
        best_response_1 = np.argmax(self.q_table1[state, :, action2])
        # Best response for Agent 2 given Agent 1's action
        best_response_2 = np.argmax(self.q_table2[state, action1, :])

        # Update Q-table for Agent 1
        self.q_table1[state, action1, action2] += self.learning_rate * (
                    reward1 + self.discount_factor * self.q_table1[next_state, best_response_1, action2] -
                    self.q_table1[state, action1, action2])

        # Update Q-table for Agent 2
        self.q_table2[state, action1, action2] += self.learning_rate * (
                    reward2 + self.discount_factor * self.q_table2[next_state, action1, best_response_2] -
                    self.q_table2[state, action1, action2])

        # Update exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

# Example initialization
# Assume some environment with env.num_states and env.num_actions per agent
# env = SomeEnvironment()
# agent = NashQAgent(env.num_states, env.num_actions)
