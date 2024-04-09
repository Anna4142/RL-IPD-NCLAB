from sys import stdout
import numpy as np
import gym
from gym.spaces import Discrete, Box
from numpy.random import randint
import matplotlib.pyplot as plt
from envs.GameConfig import GameConfig
COOPERATE = 0
DEFECT = 1
T = 5
R = 3
P = 1
S = 0


class CustomEnv(gym.Env):
    def __init__(self, game_name):
        # Initialize game configurations
        game_config = GameConfig(T=5, S=0, P=1, R=3).get_game_config(game_name)
        if not game_config:
            raise ValueError(f"Game '{game_name}' configuration not found.")

        # Setup environment according to game configuration
        self.rounds = game_config["rounds"]
        self.payout_matrix = game_config["payout_matrix"]
        self.algorithm_type = game_config["algorithm_type"]
        self.memory = game_config.get("memory", 1)
        self.history_length = game_config.get("history_length", 1)
        self.obs_type = game_config.get("obs_type", 'both')

        self.player1_action = None
        self.player2_action = None


        self.action_space = Discrete(2)  # Assuming 2 actions: Cooperate or Defect
        self.action_size = self.action_space.n

        # Calculate observation space size and initialize observation space
        obs_size = self.calculate_observation_size()
        self.observation_space = Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)

        self.association = {"dd": (1, 1), "dc": (1, 0), "cd": (0, 1), "cc": (0, 0)}
        self.current_round = 0
        self.done = False
        self.history = [(-1, -1) for _ in range(self.history_length)]
        self.state_size = 2 ** self.history_length  # Define this method if it isn't already

    def calculate_observation_size(self):
        if self.obs_type == 'both':
            return 4 * self.history_length
        elif self.obs_type in ['self', 'other']:
            return 2 * self.history_length

    def reset(self):
        self.current_round = 0
        self.done = False
        self.history = [(-1, -1) for _ in range(self.history_length)]
        return self.get_observation()

    def step(self, actions):
        if not isinstance(actions, (tuple, list)) or len(actions) != 2:
            raise ValueError("Actions must be a tuple or list with two elements.")
        self.update_history(actions[0], actions[1])
        self.current_round += 1
        if self.current_round >= self.rounds:
            self.done = True
        rewards = self.payout_matrix[actions[0]][actions[1]]
        return self.get_observation(), rewards, self.done, {}

    def update_history(self, action1, action2):
        if len(self.history) >= self.history_length:
            self.history.pop(0)
        self.history.append((action1, action2))

    def get_observation(self):
        observation = []
        for action1, action2 in self.history:
            if self.obs_type == 'self':
                observation.extend([action1])
            elif self.obs_type == 'other':
                observation.extend([action2])
            elif self.obs_type == 'both':
                observation.extend([action1, action2])
        return np.array(observation, dtype=np.float32)

    def get_state(self):
        last_actions = self.history[-1]
        action_key = f'{last_actions[0]}{last_actions[1]}'
        state = self.association.get(action_key, -1)
        return state

    def calculate_state_size(self):
        return 4 ** self.history_length  # For binary actions and pairs of actions in history

    def reset(self):
        """
            Resets the game to the initial state and returns the initial observation.
            """
        self.player1_action = None
        self.player2_action = None
        self.observation_space=(self.player1_action, self.player2_action)
        self.current_round = 0
        self.done = False
        self.history = [(-1, -1) for _ in range(self.history_length)]  # Reset history
        return self.history_to_state()  # Convert the reset history to its state representation

    def update_history(self, action1, action2):
        # Slide history to accommodate the new action pair
        self.history.pop(0)  # Remove the6 oldest action pair
        self.history.append((action1, action2))  # Add the latest action pair

    def history_to_state(self):
        # Flatten the history into a state vector; each action pair becomes two elements in the state vector
        state = []
        for action1, action2 in self.history:
            # Encode each action pair as a binary presence in the state vector
            state.extend([action1, action2])
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        """
            Takes a tuple of actions for both players and returns the new observation, reward, done flag, and any additional information.
            """
        if not isinstance(actions, (tuple, list)) or len(actions) != 2:
                raise ValueError("actions should be a tuple or list with two elements.")

        p1_action, p2_action = actions
        self.player1_action = p1_action
        self.player2_action = p2_action
        self.update_history(self.player1_action, self.player2_action)
        if p1_action not in self.action_space or p2_action not in self.action_space:
            # Handle invalid action; for example, default to COOPERATE or DEFECT
            p1_action, p2_action = 0,0  # or some other default action



        # Calculate the payouts
        p1_payout, p2_payout = self.payout_matrix[p1_action][p2_action]

        #print(self.payout_matrix[p1_action][p2_action])
        # Increment the current round and check if the game is done
        self.current_round += 1
        if self.current_round >= self.rounds:
            self.done = True
        self.observation_space=(p2_action, p1_action)
        self.state_space=list(self.association.keys())[list(self.association.values()).index((p1_action, p2_action))]
        # Convert updated history to state representation and return step information
        state = self.history_to_state()
        # Return the new observation, state,reward, done flag, and any additional information
        #return self.observation_space,(p1_action, p2_action) , (p1_payout, p2_payout), {}
        return state,(p1_action, p2_action) , (p1_payout, p2_payout), {}

    def print_ep(obs, reward, done, info):
        print({"history": obs, "reward": reward, "simulation over": done, "info": info})
    def render(self, mode='human', pos=None, close=False):
            """
            Renders the current state of the game to the screen.
            """

            """
                    :return:
                    """

            top_right = "  "
            top_left = "  "
            bot_left = "  "
            bot_right = "  "

            if pos == (0, 0):
                top_left = "AB"
            elif pos == (0, 1):
                top_left = "A "
                bot_right = " B"

            elif pos == (1, 0):


                 bot_left = "A "
                 top_right = " B"

            elif pos == (1, 1):
                bot_right = "AB"

            stdout.write("\n\n\n")
            stdout.write("      2   \n")
            stdout.write("    C   D \n")
            stdout.write("   ╔══╦══╗\n")
            stdout.write(" C ║" + top_left + "║" + top_right + "║\n")
            stdout.write("   ║  ║  ║\n")
            stdout.write("1  ╠══╬══╣\n")
            stdout.write("   ║  ║  ║\n")
            stdout.write(" D ║" + bot_left + "║" + bot_right + "║\n")
            stdout.write("   ╚══╩══╝\n\r")
            stdout.flush()

            if close:
                return

        # Print the current round and actions taken by each player
