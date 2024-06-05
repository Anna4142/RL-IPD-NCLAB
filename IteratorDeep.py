from envs.one_d_world.game import CustomEnv

from agents.FixedAgents.FixedAgents import UnconditionalCooperator, UnconditionalDefector, RandomAgent, Probability25Cooperator,Probability50Cooperator,Probability75Cooperator, TitForTat, SuspiciousTitForTat, GenerousTitForTat, Pavlov, GRIM

from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.VanillaValueBased import SARSAgent, TDLearningAgent, TDGammaAgent, QLearningAgent

from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import DQNAgent
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import REINFORCEAgent
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import ActorCriticAgent
import numpy as np
from agents.LearningAgents.Algorithms.VanillaAgents.MultiAgentVanila import NASHQ
from Evaluation.Visualization import MetricsVisualizer
from Buffer.DataBuffer import DataBuffer
from ExperimentManager import ExperimentManager
from agents.AgentsConfig import agent_types
import os
import os
from RunConfig import RunConfig
def create_agent(env, agent_type, agent_name, use_spiking_nn=False, load_saved_weights=False, base_directory=None,
                 experiment_id=None, hidden_layers=None, learning_rate=0.01, gamma=0.99):
    # Retrieve the class for the agent
    agent_class = agent_types[agent_type][agent_name]

    # Check if the agent type is 'Deep', and initialize accordingly
    if agent_type == "Deep":
        agent = agent_class(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers, learning_rate=learning_rate, gamma=gamma)
        if load_saved_weights:
            weights_path = os.path.join(base_directory, experiment_id)
            print("weights loaded from", weights_path)

            if os.path.exists(weights_path):
                agent.load_weights(weights_path)
            return agent
        else:
            print(f"No weights loaded for {agent_name}, starting from scratch or no weights found.")
            return agent
    else:
        return agent_class(env)

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
# Define parameter ranges
hidden_layers_options = [[512], [256], [32]]
learning_rates = np.logspace(-7, -0.5, num=3)  # 15 values from 1e-07 to 0.3
gammas = np.linspace(0.5, 0.9999, num=3)  # 15 values from 0.5 to 0.9999

config = RunConfig()

# Setup environment
environment_name = config.get_environment_name()
env = CustomEnv(environment_name)

# Retrieve agent configurations and experiment details
agent_type1, agent_name1, agent_params1 = config.get_agent_details('agent1')
agent1 = create_agent(env, agent_type1, agent_name1, **agent_params1)
save_directory = config.get_save_directory()
use_predefined_weights = config.use_predefined_weights()

# Construct the experiment ID and prepare for managing weights
#experiment_id = f"experiment_{agent_name1}_{agent_name2}"
#weights_dir = os.path.join(save_directory, experiment_id)
load_weights_flag = False
# If using predefined weights, adjust the experiment ID or weights directory as necessary
'''''
if use_predefined_weights:
    experiment_id += "_predefined"
    weights_dir = os.path.join(save_directory, experiment_id)
    load_weights_flag = True
'''

#agent2 = create_agent(env, agent_type2, agent_name2, **agent_params2)
experiment_manager = ExperimentManager()
#experiment_number = experiment_manager.get_next_experiment_number(experiment_id)
#experiment_number = f"experiment_{experiment_number}"
#print("experiment number",experiment_number)
# Generate experiment_id based on known agent types or other details





data_buffer = DataBuffer()
algorithm_type = env.algorithm_type
# Initialize the MetricsVisualizer with the data buffer
visualizer = MetricsVisualizer(data_buffer)
# Simulation loop
action1=0
action2=0




for hidden_layers in hidden_layers_options:
        for learning_rate in learning_rates:
          for gamma in gammas:
        # Update parameters for Agent 2
            agent_type2, agent_name2, agent_params2 = config.get_agent_details('agent2')
            agent_params2.update({
            'hidden_layers': hidden_layers,
            'learning_rate': learning_rate,
            'gamma': gamma
          })
            hidden_layers_str = '_'.join(map(str, hidden_layers))

           # Construct the experiment ID dynamically
            experiment_id = f"Experiment_mem3_hist3_{agent_name1}_{agent_name2}_Deep"
            experiment_number = f"HL_{hidden_layers_str}_LR_{learning_rate}_G_{gamma}"
            weights_dir = os.path.join(save_directory, experiment_id,experiment_number)
            if use_predefined_weights:
                experiment_id += "_predefined"
                weights_dir = os.path.join(save_directory, experiment_id)
                load_weights_flag = True
            else:
                load_weights_flag = False

            # Create or reinitialize Agent 2 with updated parameters
            agent2 = create_agent(env, agent_type2, agent_name2, **agent_params2)

            # Setup experiment environment and data recording
            data_buffer = DataBuffer()
            visualizer = MetricsVisualizer(data_buffer)
            state = (env.get_initial_state_for_agent(agent1), env.get_initial_state_for_agent(agent2))

            # Simulation loop
            for _ in range(env.rounds):
                # Agents decide their actions based on their respective states
                action1 = agent1.decide_action(state[0])
                action2 = agent2.decide_action(state[1])
                print("action 1", action1)
                print("action 2", action2)
                # Environment steps forward based on the selected actions
                next_state, rewards, done, info = env.step((action1, action2), agent1, agent2)
                print("next state 1", next_state[0])
                print("next state 2", next_state[1])
                state = next_state

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
                position = (action1, action2)

                env.render(pos=position)
            visualizer.save_all_results_and_plots(experiment_id, experiment_number)
            if algorithm_type == "SINGLE AGENT":
                save_directory_path = f'{save_directory}/{experiment_id}/'
                if not os.path.exists(save_directory_path):
                    os.makedirs(save_directory_path)

                # Generate a unique filename for the weights based on experiment details
                weight_file_name = f"weights_HL_{(hidden_layers)}_LR{learning_rate}_G{gamma}.pth"  # Update with actual parameters

                # Save weights for each deep agent with unique filenames
                if agent1.agent_type == "Deep":
                    agent1_weight_file = os.path.join(save_directory_path, f"{weight_file_name}")
                    agent1.save_weights(agent1_weight_file)

                if agent2.agent_type == "Deep":
                    agent2_weight_file = os.path.join(save_directory_path, f"{weight_file_name}")
                    #agent2.save_weights(agent2_weight_file)


