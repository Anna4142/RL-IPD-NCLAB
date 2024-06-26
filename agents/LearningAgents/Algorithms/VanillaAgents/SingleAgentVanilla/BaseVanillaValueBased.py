# base_agent.py
class BaseAgent:
    def __init__(self, env, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01, agent_type="Vanilla"):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.agent_type = agent_type  # Identifies the agent type

    def calculate_state_size(self, env):
        return 2 ** (env.history_length * 2)  # Encoding state size based on history length

    def encode_state(self, history):
        encoded = 0
        for action1, action2 in history:
            encoded = encoded * 4 + (action1 * 2) + action2
        return encoded


    def decide_action(self, state):
        pass


    def learn(self, state, action, reward, next_state, done):
        pass

