import os
import json
import torch
import numpy as np
import random
from agents.FixedAgents.FixedAgents import UnconditionalCooperator, UnconditionalDefector, RandomAgent, \
    Probability25Cooperator, Probability50Cooperator, Probability75Cooperator, TitForTat, SuspiciousTitForTat, \
    GenerousTitForTat, Pavlov, GRIM
from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.VanillaValueBased import SARSAgent, \
    TDLearningAgent, TDGammaAgent, QLearningAgent
from agents.LearningAgents.Algorithms.DLBased.DLAgents.Generic.GenericNNAgents import DQNAgent, REINFORCEAgent, \
    ActorCriticAgent, TOMActorCriticAgent, SoftActorCriticAgent, A2CAgent, PPOAgent
from Evaluation.Visualization import MetricsVisualizer
from Buffer.DataBuffer import DataBuffer
from ExperimentManager import ExperimentManager
from envs.one_d_world.game import CustomEnv
from agents.AgentsConfig import agent_types
from RunConfig import RunConfig
from envs.GameConfig import GameConfig

def create_agent(env, agent_type, agent_name, use_spiking_nn=False, load_saved_weights=False,
                 base_directory=None, experiment_id=None, hidden_layers=None, learning_rate=0.01,
                 gamma=0.99, mouse_hist=None, use_mouse_hist=False, human_hist=None, use_human_hist=False):
    agent_class = agent_types[agent_type][agent_name]
    if agent_type == "Deep":
        agent = agent_class(env, use_spiking_nn=use_spiking_nn, hidden_layers=hidden_layers,
                            learning_rate=learning_rate, gamma=gamma, mouse_hist=mouse_hist,
                            use_mouse_hist=use_mouse_hist, human_hist=human_hist, use_human_hist=use_human_hist)
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

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed at the beginning of your main function or script
set_seed(42)

# Define parameter ranges
hidden_layers_options = [[256], [64], [32]]
learning_rates = np.logspace(-7, -0.5, num=3)
gammas = np.linspace(0.5, 0.9999, num=3)
config = RunConfig()

# Setup environment
environment_name = config.get_environment_name()
env = CustomEnv(environment_name)

# Define the list of deep agents
deep_agents = [
    "DQNAgent","TOMAC",
    "A2CAgent", "PPOAgent"
]
# Get memory length from GameConfig
game_config = GameConfig(T=4, S=0, P=1, R=3)
env_config = game_config.get_game_config(environment_name)
#memory_length = env_config['memory'] if env_config else 1
memory_length=1
# Loop through each deep agent to run experiments
for agent_name in deep_agents:
    # Create base save directory based on agent names and memory length
    base_save_directory = f'Episodes/Clamped/DD_LowVar/MEMORY_LENGTH_{memory_length}_F/{agent_name}_{agent_name}/'
    os.makedirs(base_save_directory, exist_ok=True)

    # Use predefined weights and historical data configurations
    use_predefined_weights = config.use_predefined_weights()
    use_mouse_hist = config.use_forced_actions()
    mouse_hist_agent2 = None

    if use_mouse_hist:
        with open('Mouse_choices/converted_data_1774.json', 'r') as file:
            mouse_hist_agent2 = json.load(file)

    use_human_hist = config.use_human_hist()
    human_hist_data_p1 = None
    human_hist_data_p2 = None

    if use_human_hist:

        human_hist_path_p1 = os.path.join('HUMAN_SPLIT/DD_lowest_nonzero_variability/player_1_actions.json')
        human_hist_path_p2 = os.path.join('HUMAN_SPLIT/DD_lowest_nonzero_variability/player_2_actions.json')

        with open(human_hist_path_p1, 'r') as file:
            human_hist_data_p1 = json.load(file)
        with open(human_hist_path_p2, 'r') as file:
            human_hist_data_p2 = json.load(file)
        print("human hist 1", human_hist_data_p1 )

    experiment_manager = ExperimentManager()
    data_buffer = DataBuffer()
    visualizer = MetricsVisualizer(data_buffer)

    # Add outer loop for multiple runs
    num_runs = 20

    for run in range(num_runs):
        set_seed(42 + run)
        print(f"Starting Run {run + 1}/{num_runs} for agent {agent_name}")

        # Create a new directory for this run
        run_directory = os.path.join(base_save_directory, f"RUN_{run + 1}")
        os.makedirs(run_directory, exist_ok=True)
        human_hist_idx = 0  # Initialize this at the start of each experiment
        for hidden_layers in hidden_layers_options:
            for learning_rate in learning_rates:
                for gamma in gammas:
                    human_hist_idx = 0
                    agent_params1 = config.get_agent_details('agent1')[2]
                    agent_params2 = config.get_agent_details('agent2')[2]
                    agent_params1.update({
                        'hidden_layers': hidden_layers,
                        'learning_rate': learning_rate,
                        'gamma': gamma,
                        'mouse_hist': mouse_hist_agent2,
                        'use_mouse_hist': use_mouse_hist,
                        'human_hist': human_hist_data_p1,
                        'use_human_hist': use_human_hist
                    })
                    agent1 = create_agent(env, "Deep", agent_name, **agent_params1)

                    agent_params2.update({
                        'hidden_layers': hidden_layers,
                        'learning_rate': learning_rate,
                        'gamma': gamma,
                        'mouse_hist': mouse_hist_agent2,
                        'use_mouse_hist': use_mouse_hist,
                        'human_hist': human_hist_data_p2,
                        'use_human_hist': use_human_hist
                    })
                    agent2 = create_agent(env, "Deep", agent_name, **agent_params2)

                    hidden_layers_str = '_'.join(map(str, hidden_layers))
                    experiment_id = f"HL_{hidden_layers_str}_LR_{learning_rate}_G_{gamma}"
                    experiment_path = os.path.join(run_directory, experiment_id)
                    os.makedirs(experiment_path, exist_ok=True)

                    # Setup experiment environment and data recording
                    data_buffer = DataBuffer()
                    visualizer = MetricsVisualizer(data_buffer)
                    state = (env.get_initial_state_for_agent(agent1), env.get_initial_state_for_agent(agent2))

                    # Simulation loop
                    for _ in range(env.rounds):
                        if human_hist_idx >= len(human_hist_data_p1):
                            human_hist_idx = 0  # Reset if we've gone through all data

                        # Determine the forced action for agent1
                        forced_action1 = None
                        if use_human_hist and human_hist_idx < len(human_hist_data_p1):
                            forced_action1 = human_hist_data_p1[human_hist_idx]
                        elif use_mouse_hist and human_hist_idx < len(mouse_hist_agent2):
                            forced_action1 = mouse_hist_agent2[human_hist_idx]

                        # Determine the forced action for agent2
                        forced_action2 = None
                        if use_human_hist and human_hist_idx < len(human_hist_data_p2):
                            forced_action2 = human_hist_data_p2[human_hist_idx]
                        elif use_mouse_hist and human_hist_idx < len(mouse_hist_agent2):
                            forced_action2 = mouse_hist_agent2[human_hist_idx]

                        # Agent 1 decision
                        if isinstance(agent1, TOMActorCriticAgent):
                            state_others = state[1]
                            action1 = agent1.decide_action(state[0], state_others, forced_action1)
                        elif isinstance(agent1, DQNAgent):
                            action1 = agent1.decide_action([forced_action1] if forced_action1 is not None else state[0])
                        else:
                            action1 = agent1.decide_action(state[0])

                        # Agent 2 decision
                        if isinstance(agent2, TOMActorCriticAgent):
                            state_others = state[0]
                            action2 = agent2.decide_action(state[1], state_others, forced_action2)
                        elif isinstance(agent2, DQNAgent):
                            action2 = agent2.decide_action([forced_action2] if forced_action2 is not None else state[1])
                        else:
                            action2 = agent2.decide_action(state[1])

                        print(f"Forced action 1: {forced_action1}, Chosen action 1: {action1}")
                        print(f"Forced action 2: {forced_action2}, Chosen action 2: {action2}")

                        # Agent 1 decision
                        if isinstance(agent1, TOMActorCriticAgent):
                            state_others = state[1]
                            action1 = agent1.decide_action(state[0], state_others, forced_action1)
                        else:
                            action1 = agent1.decide_action(state[0])

                        # Agent 2 decision
                        if isinstance(agent2, TOMActorCriticAgent):
                            state_others = state[0]
                            action2 = agent2.decide_action(state[1], state_others, forced_action2)
                        else:
                            action2 = agent2.decide_action(state[1])

                        print(f"Forced action 1: {forced_action1}, Chosen action 1: {action1}")
                        print(f"Forced action 2: {forced_action2}, Chosen action 2: {action2}")

                        human_hist_idx += 1

                        next_state, rewards, done, info = env.step((action1, action2), agent1, agent2)
                        state = next_state

                        if isinstance(agent1, DQNAgent):
                            agent1.store_transition(state[0], action1, action2, next_state[0], rewards[0], rewards[1])
                        if isinstance(agent2, DQNAgent):
                            agent2.store_transition(state[1], action1, action2, next_state[1], rewards[1], rewards[1])

                        visualizer.update_metrics(rewards[0], rewards[1], action1, action2, forced_action1)

                    visualizer.save_all_results_and_plots(experiment_path, experiment_id)

        print(f"Completed Run {run + 1}/{num_runs} for agent {agent_name}")

print("All runs completed.")
