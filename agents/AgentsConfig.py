# agents_config.py

from agents.FixedAgents.FixedAgents import (
    UnconditionalCooperator, UnconditionalDefector, RandomAgent,
    Probability25Cooperator, Probability50Cooperator, Probability75Cooperator,
    TitForTat, SuspiciousTitForTat, GenerousTitForTat, Pavlov, GRIM
)

from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.VanillaValueBased import (
    SARSAgent, TDLearningAgent, TDGammaAgent, QLearningAgent
)

from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import DQNAgent, REINFORCEAgent,TOMActorCriticAgent, ActorCriticAgent

agent_types = {
    "Fixed": {
        "UnconditionalCooperator": UnconditionalCooperator,
        "UnconditionalDefector": UnconditionalDefector,
        "RandomAgent": RandomAgent,
        "Probability25Cooperator": Probability25Cooperator,
        "Probability50Cooperator": Probability50Cooperator,
        "Probability75Cooperator": Probability75Cooperator,
        "TitForTat": TitForTat,
        "SuspiciousTitForTat": SuspiciousTitForTat,
        "GenerousTitForTat": GenerousTitForTat,
        "Pavlov": Pavlov,
        "GRIM": GRIM
    },
    "Vanilla": {
        "SARSAgent": SARSAgent,
        "TDLearningAgent": TDLearningAgent,
        "TDGammaAgent": TDGammaAgent,
        "QLearningAgent": QLearningAgent
    },
    "Deep": {
        "DQNAgent": DQNAgent,
        "REINFORCEAgent": REINFORCEAgent,
        "ActorCriticAgent": ActorCriticAgent,
        'TOMAC':TOMActorCriticAgent
    }
}
