from envs.one_d_world.game import CustomEnv
from agents.FixedAgents.FixedAgents import UnconditionalCooperator, UnconditionalDefector, RandomAgent, \
    Probability25Cooperator, Probability50Cooperator, Probability75Cooperator, TitForTat, SuspiciousTitForTat, \
    GenerousTitForTat, Pavlov, GRIM
from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.VanillaValueBased import SARSAgent, \
    TDLearningAgent, TDGammaAgent, QLearningAgent
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import DQNAgent, REINFORCEAgent, \
    ActorCriticAgent,TOMActorCriticAgent
import numpy as np
from agents.LearningAgents.Algorithms.VanillaAgents.MultiAgentVanila import NASHQ
from Evaluation.Visualization import MetricsVisualizer
from Buffer.DataBuffer import DataBuffer
from ExperimentManager import ExperimentManager
from agents.AgentsConfig import agent_types
import os
from RunConfig import RunConfig
import json

def create_agent(env, agent_type, agent_name, use_spiking_nn=False, load_saved_weights=False, base_directory=None,
                 experiment_id=None, hidden_layers=None, learning_rate=0.01, gamma=0.99, mouse_hist=None,
                 use_mouse_hist=False):
    agent_class = agent_types[agent_type][agent_name]
    if agent_type == "Deep":
        agent = agent_class(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers,
                            learning_rate=learning_rate, gamma=gamma, mouse_hist=mouse_hist,
                            use_mouse_hist=use_mouse_hist)
        if load_saved_weights:
            weights_path = os.path.join(base_directory, experiment_id)
            if os.path.exists(weights_path):
                agent.load_weights(weights_path)
                print(f"Weights loaded from {weights_path}")
            else:
                print("No weights found at specified path.")
    else:
        agent = agent_class(env)
    return agent


# Define parameter ranges
hidden_layers_options = [[512]]
learning_rates = np.logspace(-7, -0.5, num=1)  # 3 values from 1e-07 to 0.3
gammas = np.linspace(0.5, 0.9999, num=1)  # 3 values from 0.5 to 0.9999

config = RunConfig()

# Setup environment
environment_name = config.get_environment_name()
env = CustomEnv(environment_name)

# Retrieve agent configurations and experiment details
agent_type1, agent_name1, agent_params1 = config.get_agent_details('agent1')
agent1 = create_agent(env, agent_type1, agent_name1, **agent_params1)
save_directory = config.get_save_directory()
use_predefined_weights = config.use_predefined_weights()
use_mouse_hist = config.use_forced_actions()  # Get the use_forced_actions parameter
mouse_hist_agent2 = None
if use_mouse_hist:
    with open('Mouse_choices/converted_data_1774.json', 'r') as file:
        mouse_hist_agent2 = json.load(file)

experiment_manager = ExperimentManager()
data_buffer = DataBuffer()
algorithm_type = env.algorithm_type
visualizer = MetricsVisualizer(data_buffer)
action1 = 0
action2 = 0
mouse_hist_idx = 0
next_state = [np.zeros(env.state_size, dtype=float), np.zeros(env.state_size, dtype=float)]
last_state1 = np.zeros(env.state_size, dtype=float)
last_state2 = np.zeros(env.state_size, dtype=float)
for hidden_layers in hidden_layers_options:
    for learning_rate in learning_rates:
        for gamma in gammas:
            agent_type2, agent_name2, agent_params2 = config.get_agent_details('agent2')
            agent_params2.update({
                'hidden_layers': hidden_layers,
                'learning_rate': learning_rate,
                'gamma': gamma,
                'mouse_hist': mouse_hist_agent2,
                'use_mouse_hist': use_mouse_hist
            })
            hidden_layers_str = '_'.join(map(str, hidden_layers))

            # Construct the experiment ID dynamically
            experiment_id = f"Experiment_mem2_hist2_{agent_name1}_{agent_name2}"
            experiment_number = f"HL_{hidden_layers_str}_LR_{learning_rate}_G_{gamma}"
            weights_dir = os.path.join(save_directory, experiment_id, experiment_number)
            load_weights_flag = use_predefined_weights

            # Create or reinitialize Agent 2 with updated parameters
            agent2 = create_agent(env, agent_type2, agent_name2, **agent_params2)

            # Setup experiment environment and data recording
            data_buffer = DataBuffer()
            visualizer = MetricsVisualizer(data_buffer)
            state = (env.get_initial_state_for_agent(agent1), env.get_initial_state_for_agent(agent2))

            # Simulation loop
            for _ in range(env.rounds):

                # Agents decide their actions based on their respective states
                if isinstance(agent1, TOMActorCriticAgent):
                    # Check if agent2 is of type "Fixed"
                    last_state2= state[0]
                    state_others = last_state2 if agent_type2 == "Fixed" else state[1]
                    action1 = agent1.decide_action(state[0], state_others)
                else:
                    action1 = agent1.decide_action(state[0])  # Non-TOMAC agent uses only its own state

                    # Decision-making for agent2
                if isinstance(agent2, TOMActorCriticAgent):
                    # Check if agent1 is of type "Fixed"
                    last_state1 = state[1]
                    state_others = last_state1 if agent_type1 == "Fixed" else state[0]
                    action2 = agent2.decide_action(state[1], state_others)
                else:
                    action2 = agent2.decide_action(state[1])  # Non-TOMAC agent

                # Agents decide their actions based on their respective states


                print("action 1", action1)
                print("action 2", action2)
                # Environment steps forward based on the selected actions
                next_state, rewards, done, info = env.step((action1, action2), agent1, agent2)
                print("next state 1", next_state[0])
                print("next state 2", next_state[1])
                next_state1 = next_state
                state = next_state1

                if isinstance(agent1, DQNAgent):
                    agent1.store_transition(state[0], action1, action2, next_state[0], rewards[0], rewards[1])
                if isinstance(agent2, DQNAgent):
                    agent2.store_transition(state[1], action1, action2, next_state[1], rewards[1], rewards[1])

                if isinstance(agent1, TOMActorCriticAgent):
                    # Further check if agent2 is of type "Fixed"
                    if agent_type2 == "Fixed":  # This should be a class check, not a string comparison
                        # Use the last state of agent2 as state_others for agent1
                        state_others = last_state1
                        next_state_others = state[0]
                        agent1.learn(state[0], state_others, action1, rewards[0], next_state[0], next_state_others, done)
                    else:
                        # Normal processing if agent2 is not "Fixed"
                        agent1.learn(state[0], state[1], action1, rewards[0], next_state[0], next_state[1], done)
                elif hasattr(agent1, 'learn'):
                    # Standard learning process for non-TOMAC agents
                    agent1.learn(state[0], action1, rewards[0], next_state[0], done)

                # Learning for agent2, checking if it is TOMAC
                if isinstance(agent2, TOMActorCriticAgent):
                    # Further check if agent1 is of type "Fixed"
                    if agent_type1 == "Fixed":
                        state_others = last_state2
                        next_state_others = state[1]
                        agent2.learn(state[1], state_others, action2, rewards[1], next_state[1], next_state_others, done)
                    else:
                        # Normal processing if agent1 is not "Fixed"
                        agent2.learn(state[1], state[0], action2, rewards[1], next_state[1], next_state[0], done)
                elif hasattr(agent2, 'learn'):
                    # Standard learning process for non-TOMAC agents
                    agent2.learn(state[1], action2, rewards[1], next_state[1], done)



                # Update visualization and render environment
                if mouse_hist_agent2 is not None:
                    visualizer.update_metrics(rewards[0], rewards[1], action1, action2,
                                              mouse_hist_agent2[mouse_hist_idx])
                else:
                    visualizer.update_metrics(rewards[0], rewards[1], action1, action2, None)

                position = (action1, action2)
                mouse_hist_idx += 1

                env.render(pos=position)

            visualizer.save_all_results_and_plots(experiment_id, experiment_number)
            if algorithm_type == "SINGLE AGENT":
                save_directory_path = f'{save_directory}/{experiment_id}/'
                if not os.path.exists(save_directory_path):
                    os.makedirs(save_directory_path)

                # Generate a unique filename for the weights based on experiment details
                weight_file_name = f"weights_HL_{hidden_layers_str}_LR{learning_rate}_G{gamma}.pth"

                # Save weights for each deep agent with unique filenames
                if agent1.agent_type == "Deep":
                    agent1_weight_file = os.path.join(save_directory_path, f"{weight_file_name}")
                    agent1.save_weights(agent1_weight_file)

                if agent2.agent_type == "Deep":
                    agent2_weight_file = os.path.join(save_directory_path, f"{weight_file_name}")
                    agent2.save_weights(agent2_weight_file)


