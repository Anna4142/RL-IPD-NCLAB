import random
from agents.FixedAgents.BaseFixedAgent import BaseAgent
# Basic definitions for the Prisoner's Dilemma actions
COOPERATE = 0
DEFECT = 1




class UnconditionalCooperator(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
    def decide_action(self, opponent_last_action):
        return COOPERATE

class UnconditionalDefector(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
    def decide_action(self, opponent_last_action):
        return DEFECT

class RandomAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
    def decide_action(self, opponent_last_action):
        return random.choice([COOPERATE, DEFECT])

class Probability25Cooperator(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
        self.p = 0.25
    def decide_action(self, opponent_last_action):
        return COOPERATE if random.random() < self.p else DEFECT

class Probability50Cooperator(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
        self.p = 0.50
    def decide_action(self, opponent_last_action):
        return COOPERATE if random.random() < self.p else DEFECT

class Probability75Cooperator(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
        self.p = 0.75
    def decide_action(self, opponent_last_action):
        return COOPERATE if random.random() < self.p else DEFECT

class TitForTat(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
    def decide_action(self, opponent_last_action):
        return COOPERATE if opponent_last_action == -1 else opponent_last_action

class SuspiciousTitForTat(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
    def decide_action(self, opponent_last_action):
        return DEFECT if opponent_last_action == -1 else opponent_last_action

class GenerousTitForTat(BaseAgent):
    def __init__(self, env, generosity=0.3):
        super().__init__(env, agent_type="Fixed")
        self.generosity = generosity
    def decide_action(self, opponent_last_action):
        if opponent_last_action == COOPERATE or opponent_last_action == -1:
            return COOPERATE
        else:
            return COOPERATE if random.random() < self.generosity else DEFECT

class Pavlov(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
    def decide_action(self, opponent_last_action):
        if len(self.history) > 1 and self.history[-1][0] == self.history[-1][1]:
            return self.history[-1][0]
        else:
            return DEFECT if self.history[-1][0] == COOPERATE else COOPERATE

class GRIM(BaseAgent):
    def __init__(self, env):
        super().__init__(env, agent_type="Fixed")
        self.grim_triggered = False
    def decide_action(self, opponent_last_action):
        if opponent_last_action == DEFECT:
            self.grim_triggered = True
        return DEFECT if self.grim_triggered else COOPERATE

