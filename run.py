from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from envs.one_d_world.game import CustomEnv  # Ensure this is correctly imported
from agents.FixedAgents.agents import UnconditionalCooperator, UnconditionalDefector, RandomAgent


# Define the simulation function
def run_simulation(num_episodes=100, agent1_strategy=UnconditionalCooperator, agent2_strategy=UnconditionalDefector):
    env = CustomEnv()  # Initialize your environment

    # Initialize agents with strategies
    agent1 = agent1_strategy(env.history_length)
    agent2 = agent2_strategy(env.history_length)

    accumulated_rewards = np.zeros((num_episodes, 2))  # Track rewards for each agent
    actions_history = []  # List to store actions for each episode

    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = np.zeros(2)
        episode_actions = []

        done = False
        while not done:
            action1 = agent1.decide_action(None)  # For fixed strategies, opponent's last action may not be needed
            action2 = agent2.decide_action(None)
            actions = [action1, action2]

            obs, rewards, done, _ = env.step(actions=actions)
            episode_rewards += np.array(rewards)  # Ensure rewards are in a format that can be summed up
            episode_actions.append(actions)

        accumulated_rewards[episode, :] = episode_rewards
        actions_history.append(episode_actions)

    return accumulated_rewards, actions_history