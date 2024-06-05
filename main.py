from envs.one_d_world.game import CustomEnv

from agents.FixedAgents.FixedAgents import UnconditionalCooperator, UnconditionalDefector, RandomAgent, Probability25Cooperator,Probability50Cooperator,Probability75Cooperator, TitForTat, SuspiciousTitForTat, GenerousTitForTat, Pavlov, GRIM

from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.VanillaValueBased import SARSAgent, TDLearningAgent, TDGammaAgent, QLearningAgent

from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import DQNAgent
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import REINFORCEAgent
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import ActorCriticAgent
import random
from agents.LearningAgents.Algorithms.VanillaAgents.MultiAgentVanila import NASHQ
from Evaluation.Visualization import MetricsVisualizer
from Buffer.DataBuffer import DataBuffer
from ExperimentManager import ExperimentManager
from agents.AgentsConfig import agent_types
import os
import os
from RunConfig import RunConfig
import json

def create_agent(env, agent_type, agent_name, use_spiking_nn=False, load_saved_weights=False, base_directory=None,
                 experiment_id=None, hidden_layers=None, learning_rate=0.01, gamma=0.99, mouse_hist=None, use_mouse_hist=False):
    """
    Creates an agent with specified configurations. This function can handle 'Deep' agents and potentially other types.

    Args:
    - env: Environment in which the agent operates.
    - agent_type: Type of the agent ('Deep' or others).
    - agent_name: Specific name of the agent class.
    - use_spiking_nn: Flag to use spiking neural network models.
    - load_saved_weights: Flag to load saved weights if available.
    - base_directory: Base directory where weights might be stored.
    - experiment_id: Specific experiment identifier for loading weights.
    - hidden_layers: Configuration of hidden layers for neural networks.
    - learning_rate: Learning rate for the agent's optimizer.
    - gamma: Discount factor for the agent's learning algorithm.
    - mouse_hist: Optional list of predetermined actions for the agent.
    - use_mouse_hist: Boolean to determine whether to use the mouse_hist list or not.

    Returns:
    - agent: The initialized agent.
    """
    agent_class = agent_types[agent_type][agent_name]
    print("LOAD WEIGHTS PATH",load_saved_weights)
    # Initialize the agent based on its type, checking specifically for 'Deep' agents
    if agent_type == "Deep":
        agent = agent_class(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers,
                            learning_rate=learning_rate, gamma=gamma, use_mouse_hist=use_mouse_hist,mouse_hist=mouse_hist)
        if load_saved_weights:
            weights_path = os.path.join(base_directory, experiment_id)
            if os.path.exists(weights_path):
                agent.load_weights(weights_path)
                print("Weights loaded from", weights_path)
            else:
                print("No weights found at specified path.")
    else:
        # Initialize non-deep agents without neural network specific parameters
        agent = agent_class(env)

    return agent

# Define a function to safely extract parameter values with default fallbacks
def get_parameter(agent_config, param_name, default_value):
    # Retrieve parameters dictionary
    params = agent_config[2]  # Since get_agent_details returns a tuple (type, name, parameters)
    # Return the parameter value if available, otherwise return the default value
    return params.get(param_name, default_value)
"""""
env = CustomEnv("prisoners_dilemma")
load_weights_flag = False
use_predefined_weights_id = False  # Boolean flag to choose weights directory
# Configuration or initial settings
agent_type1 = "Fixed"
agent_name1 = "TitForTat"
agent_type2 = "Deep"
agent_name2 = "ActorCriticAgent"
save_directory="weights"
experiment_id = f"experiment_{agent_name1}_{agent_name2}"
"""""
config = RunConfig()

# Setup environment
environment_name = config.get_environment_name()
env = CustomEnv(environment_name)

# Retrieve agent configurations and experiment details
agent_type1, agent_name1, agent_params1 = config.get_agent_details('agent1')
agent_type2, agent_name2, agent_params2 = config.get_agent_details('agent2')
save_directory = config.get_save_directory()
use_predefined_weights = config.use_predefined_weights()
with open('Mouse_choices/converted_data_1774.json', 'r') as file:

        mouse_hist_agent2 = json.load(file)

use_mouse_hist=True
print("mouse hist",mouse_hist_agent2)
# Construct the experiment ID and prepare for managing weights
experiment_id = f"experiment_{agent_name1}_{agent_name2}"
weights_dir = os.path.join(save_directory, experiment_id)
load_weights_flag = False
use_predefined_weights=False
# If using predefined weights, adjust the experiment ID or weights directory as necessary
if use_predefined_weights:
    experiment_id += "_predefined"
    weights_dir = os.path.join(save_directory, experiment_id)
    load_weights_flag = True

if use_mouse_hist:
    experiment_id += "_usingforcedactions"
    weights_dir = os.path.join(save_directory, experiment_id)
    load_weights_flag = False

experiment_manager = ExperimentManager()
experiment_number = experiment_manager.get_next_experiment_number(experiment_id)
experiment_number = f"experiment_{experiment_number}"
print("experiment number",experiment_number)
# Generate experiment_id based on known agent types or other details
if use_predefined_weights:
     experiment_id_loading = f"experiment_UnconditionalCooperator_DQNAgent"
else:
    experiment_id_loading = experiment_id




data_buffer = DataBuffer()
algorithm_type = env.algorithm_type

if algorithm_type == "MULTI AGENT":
    # Initialize a centralized agent

    agent = NASHQ.NashQAgent(env)
    agent_names = f"{agent.__class__.__name__}"
elif algorithm_type == "SINGLE AGENT":

    agent1_config = config.get_agent_details('agent1')
    agent2_config = config.get_agent_details('agent2')
    hidden_layers1 = get_parameter(agent1_config, 'hidden_layers', [128])  # Default to [128] if not specified
    learning_rate1 = get_parameter(agent1_config, 'learning_rate', 0.01)  # Default to 0.01 if not specified
    gamma1 = get_parameter(agent1_config, 'gamma', 0.95)  # Default to 0.95 if not specified

    # Extract specific parameters for Agent 2
    hidden_layers2 = get_parameter(agent2_config, 'hidden_layers', [128])  # Default to [128] if not specified
    learning_rate2 = get_parameter(agent2_config, 'learning_rate', 0.01)  # Default to 0.01 if not specified
    gamma2 = get_parameter(agent2_config, 'gamma', 0.95)  # Default to 0.95 if not specified

    agent1 = create_agent(env, agent_type1, agent_name1, base_directory=save_directory,
                         experiment_id=experiment_id_loading)

    agent2 = create_agent(env, agent_type2, agent_name2, use_spiking_nn=True, load_saved_weights=load_weights_flag,
                          base_directory=save_directory, experiment_id=experiment_id_loading,
                          hidden_layers=hidden_layers2, learning_rate=learning_rate2, gamma=gamma2,
                          mouse_hist=mouse_hist_agent2,use_mouse_hist=use_mouse_hist)
    initial_state1 = env.get_initial_state_for_agent(agent1)
    initial_state2 = env.get_initial_state_for_agent(agent2)
    state = (initial_state1,initial_state2)

    agent_names = f"{agent1.__class__.__name__}_{agent2.__class__.__name__}"


# Initialize the MetricsVisualizer with the data buffer
visualizer = MetricsVisualizer(data_buffer)
# Simulation loop
action1=0
action2=0
mouse_hist_idx = 0
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
    state = next_state



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
    visualizer.update_metrics(rewards[0], rewards[1], action1, action2,mouse_hist_agent2[mouse_hist_idx])
    position=(action1,action2)
    mouse_hist_idx +=1

    env.render(pos=position)
# Save results after simulation
visualizer.save_all_results_and_plots(experiment_id, experiment_number)
if algorithm_type == "SINGLE AGENT":
    save_directory = f'{save_directory}/{experiment_id}/{experiment_number}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Check if the agent_type attribute of each agent is 'Deep' before saving
    if agent1.agent_type == "Deep":
        agent1.save_weights(save_directory)
    if agent2.agent_type == "Deep":
        agent2.save_weights(save_directory)











