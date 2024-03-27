from envs.one_d_world.game import CustomEnv
from agents.LearningAgents.Algorithms.SingleAgentVanilla import SARSA, TD0
from agents.LearningAgents.Algorithms.MultiAgentVanila import JointActionLearning,NASHQ
from Evaluation.Visualization import MetricsVisualizer
from Buffer.DataBuffer import DataBuffer
env = CustomEnv("prisoners_dilemma")
algorithm_type = env.algorithm_type

# Initialize agents based on the algorithm type
# Initialize the agents
if algorithm_type == "MULTI AGENT":
    # Initialize a centralized agent

    agent = NASHQ.NashQAgent(env)
elif algorithm_type == "SINGLE AGENT":
    agent1 = TD0.TDLearningAgent(env)
    agent2 = SARSA.SARSAgent(env)


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
visualizer .plot_all()




