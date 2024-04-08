
from envs.one_d_world.game import CustomEnv
from agents.FixedAgents.FixedAgents import UnconditionalCooperator, UnconditionalDefector, RandomAgent, Probability25Cooperator,Probability50Cooperator,Probability75Cooperator, TitForTat, SuspiciousTitForTat, GenerousTitForTat, Pavlov, GRIM


#from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla import TD0, SARSA,VanillaQLearning,TDGAMMA
from agents.LearningAgents.Algorithms.VanillaAgents.SingleAgentVanilla.VanillaValueBased import SARSAgent, TDLearningAgent, TDGammaAgent, QLearningAgent
from agents.LearningAgents.Algorithms.VanillaAgents.MultiAgentVanila import NASHQ
from Evaluation.Visualization import MetricsVisualizer
from Buffer.DataBuffer import DataBuffer
from ExperimentManager import ExperimentManager
# Assuming the rest of your imports and CustomEnv are defined elsewhere

# Prepare agent lists
fixed_agents = [UnconditionalCooperator, UnconditionalDefector, RandomAgent, Probability25Cooperator,Probability50Cooperator,Probability75Cooperator, TitForTat,
                SuspiciousTitForTat, GenerousTitForTat, Pavlov, GRIM]
vanilla_agents = [SARSAgent, TDLearningAgent, TDGammaAgent, QLearningAgent]

# Experiment Manager
experiment_manager = ExperimentManager()

# Loop through each combination of fixed agent and vanilla agent
for FixedAgent in fixed_agents:
    for VanillaAgent in vanilla_agents:
        # Initialize environment and agents
        env = CustomEnv("prisoners_dilemma")
        agent1 = VanillaAgent(env)
        agent2 = VanillaAgent(env)

        agent_names = f"{agent1.__class__.__name__}_{agent2.__class__.__name__}"
        experiment_id = f"experiment_{agent_names}"
        experiment_number = experiment_manager.get_next_experiment_number(experiment_id)
        experiment_number = f"experiment_{experiment_number}"

        data_buffer = DataBuffer()
        visualizer = MetricsVisualizer(data_buffer)

        # Simulation loop
        for _ in range(env.rounds):
            current_state = env.get_state()
            action1 = agent1.decide_action(env.player2_action)
            action2 = agent2.decide_action(env.player1_action)
            obs, actions, reward, done = env.step((action1, action2))

            reward1 = reward[0]
            reward2 = reward[1]

            visualizer.update_metrics(reward1, reward2, action1, action2)
            #env.render(pos=env.association[env.state_space])

            if done:
                break

        # Save results and plots
        visualizer.save_all_results_and_plots(experiment_id, experiment_number)
