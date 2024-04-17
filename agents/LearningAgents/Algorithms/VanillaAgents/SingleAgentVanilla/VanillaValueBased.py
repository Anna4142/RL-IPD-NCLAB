# derived_agents.py
import numpy as np
from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.BaseVanillaValueBased import BaseAgent

class SARSAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.q_table = np.zeros((self.state_size, self.action_size))


    def decide_action(self, state):

        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state,done):


        next_estimate = self.q_table[next_state, action]
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_rate * next_estimate - self.q_table[state, action])
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

class TDLearningAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.state_size = env.state_size
        self.value_table = np.zeros(self.state_size)

    def decide_action(self, state):
        return np.random.randint(self.env.action_size)  # Simplified for demonstration

    def learn(self, state, action, reward, next_state, done):
        td_target = reward + self.discount_rate * self.value_table[next_state]
        td_error = td_target - self.value_table[state]
        self.value_table[state] += self.learning_rate * td_error

class TDGammaAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.state_size = env.state_size
        self.value_table = np.zeros(self.state_size)

    def decide_action(self, state):
        return np.random.randint(2)  # Assuming 2 actions for simplicity

    def learn(self, state, action, reward, next_state, done):
        td_target = reward + self.discount_rate * self.value_table[next_state]
        td_error = td_target - self.value_table[state]
        self.value_table[state] += self.learning_rate * td_error

class QLearningAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.q_table = np.zeros((self.state_size, self.action_size))

    def decide_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_rate * next_max - self.q_table[state, action])


# Additional derived agents (TDLearningAgent, TDGammaAgent, QLearningAgent) would be defined here in a similar manner.

