from envs.one_d_world.game import CustomEnv

from agents.FixedAgents.FixedAgents import UnconditionalCooperator, UnconditionalDefector, RandomAgent, Probability25Cooperator,Probability50Cooperator,Probability75Cooperator, TitForTat, SuspiciousTitForTat, GenerousTitForTat, Pavlov, GRIM

from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.VanillaValueBased import SARSAgent, TDLearningAgent, TDGammaAgent, QLearningAgent

from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.DQN import DQNAgent
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.REINFORCE import REINFORCEAgent

from agents.LearningAgents.Algorithms.VanillaAgents.MultiAgentVanila import NASHQ
from Evaluation.Visualization import MetricsVisualizer
from Buffer.DataBuffer import DataBuffer
from ExperimentManager import ExperimentManager
env = CustomEnv("prisoners_dilemma")
algorithm_type = env.algorithm_type

if algorithm_type == "MULTI AGENT":
    # Initialize a centralized agent

    agent = NASHQ.NashQAgent(env)
    agent_names = f"{agent.__class__.__name__}"
elif algorithm_type == "SINGLE AGENT":

    agent1 = SARSAgent(env)

    agent2 =DQNAgent(env)

    agent_names = f"{agent1.__class__.__name__}_{agent2.__class__.__name__}"
# Assuming 'agent_names' is set from the above logic
experiment_id = f"experiment_{agent_names}"
experiment_manager = ExperimentManager()
experiment_number = experiment_manager.get_next_experiment_number(experiment_id)
experiment_number = f"experiment_{experiment_number}"
print("experiment number",experiment_number)
data_buffer = DataBuffer()
# Initialize the MetricsVisualizer with the data buffer
visualizer = MetricsVisualizer(data_buffer)
# Simulation loop
action1=0
action2=0
state=(0,[0., 0., 0., 0., 0. ,0. ,0. ,0. ,0., 0. ,0. ,0., 0., 0., 0., 0.])
for _ in range(env.rounds):


    # Agents decide their actions based on their respective states
    action1 = agent1.decide_action(state[0])
    action2 = agent2.decide_action(state[1])
    print("action 1",action1)
    print("action 2",action2)
    # Environment steps forward based on the selected actions
    next_state, rewards, done, info = env.step((action1, action2), agent1, agent2)
    print("next state 1", next_state[0])
    print("next state 2",next_state[1])



    # Store transition for each agent if they are DQNAgents
    if isinstance(agent1, DQNAgent):
        agent1.store_transition(state[0], action1, action2, next_state[0], rewards[0], rewards[1])
    if isinstance(agent2, DQNAgent):
        agent2.store_transition(state[1], action1, action2, next_state[1], rewards[1], rewards[1])

    # Agents learn from the transition
    if hasattr(agent1, 'learn'):
        agent1.learn(state[0], action1, rewards[0], next_state[0], done)
    if hasattr(agent2, 'learn'):
        agent2.learn(state[1], action2, rewards[1], next_state[1], done)

    # Update visualization and render environment
    visualizer.update_metrics(rewards[0], rewards[1], action1, action2)
    position=(action1,action2)
    state=next_state

    env.render(pos=position)




# Save results after simulation
visualizer.save_all_results_and_plots(experiment_id, experiment_number)





