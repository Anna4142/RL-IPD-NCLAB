from envs.one_d_world.game import CustomEnv
from agents.FixedAgents.FixedAgents import UnconditionalCooperator, UnconditionalDefector, RandomAgent, Probability25Cooperator,Probability50Cooperator,Probability75Cooperator, TitForTat, SuspiciousTitForTat, GenerousTitForTat, Pavlov, GRIM

from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.VanillaValueBased import SARSAgent, TDLearningAgent, TDGammaAgent, QLearningAgent

from agents.LearningAgents.Algorithms.VanillaAgents.MultiAgentVanila import NASHQ
from Evaluation.Visualization import MetricsVisualizer
from Buffer.DataBuffer import DataBuffer
from ExperimentManager import ExperimentManager
env = CustomEnv("prisoners_dilemma")
algorithm_type = env.algorithm_type

# Initialize agents based on the algorithm type
# Initialize the agents
if algorithm_type == "MULTI AGENT":
    # Initialize a centralized agent

    agent = NASHQ.NashQAgent(env)
    agent_names = f"{agent.__class__.__name__}"
elif algorithm_type == "SINGLE AGENT":
    agent1 = UnconditionalCooperator(env)
    agent2 = SARSAgent(env)
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
for _ in range(env.rounds):
    # Agents decide their actions
    current_state = env.get_state()

    #current_state=(env.player2_action,env.player1_action)

    if algorithm_type == "MULTI AGENT":
        action1, action2 = agent.decide_action(current_state)
    elif algorithm_type == "SINGLE AGENT":
       action1 = agent1.decide_action(env.player2_action)
       action2 = agent2.decide_action(env.player1_action)
    # Environment steps forward based on the selected actions
    obs, actions, reward, done = env.step((action1, action2))

    reward1=reward[0]
    reward2=reward[1]


    visualizer.update_metrics(reward1,reward2,action1,action2)
    # Optionally, render the current state of the game
    env.render(pos=env.association[env.state_space])

    if done:

        break
# Call the function to run the entire process
visualizer .save_all_results_and_plots( experiment_id,experiment_number)




