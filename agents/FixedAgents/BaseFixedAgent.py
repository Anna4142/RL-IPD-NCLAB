from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, env, agent_type="Unknown"):
        self.env = env
        self.agent_type = agent_type
        self.history = [(-1, -1) for _ in range(env.history_length)]

    @abstractmethod
    def decide_action(self, opponent_last_action):
        pass

    def update_history(self, action, opponent_action):
        self.history.pop(0)
        self.history.append((action, opponent_action))

