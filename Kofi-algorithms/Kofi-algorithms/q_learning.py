""" 
Q Learning Agent

Args: 
dim: dimensions of agents' Q table
alpha: learning rate parameter
gamma: discount parameter
epsilon: exploration parameter
num_actions: int (must be same as first dimension of dim)
dim: dimensions of Qtable (must be (action, ) + (space_x,space_y, ...))
mapping_fn: optional function to transform an observatiohn

Implements:
action_callback(state): returns action integer
experience_callback: Q Learning update
episode_callback: End episode 
reset: End episode in evalutation 

Watkins, 1989
Implemented according to Sutton 2018 formula 6.8

Implemented with epsilon decay
"""

import numpy as np 
import numpy.random as npr

class QLearningAgent(object):
    """
    Generic Q Learning implementation
    """
    def __init__(self, num_actions, dim, alpha=0.7, gamma=0.9, epsilon=1, eps_min=0.005, eps_decay=0.9999, mapping_fn=None):
        """ Initialize parameters """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min 
        self.eps_decay = eps_decay

        """ Initialize Q learning table """
        self.dim_len = len(dim)
        if num_actions != dim[0]:
            raise ValueError('Q table dimension must start with action dimension')

        self.Q = np.zeros(dim)

        self.num_actions = num_actions
        self.mapping_fn = mapping_fn

    def int_to_vector(self, action):
        """ Turns integer action into one hot vector """
        vec = np.zeros(self.num_actions)
        vec[action] = 1 
        return vec 
        
    """ Training callbacks """
    def action_callback(self, state):
        """ Returns the 1-hot encoded vector of the agent action (direction of movement)
        state: np.array of integer obesrvations or tuple
        """

        if self.mapping_fn:
            state = self.mapping_fn(state)

        """ epsilon greedy next action """
        if npr.uniform(0,1) < self.epsilon:
            new_action = npr.randint(0, self.num_actions)
        else:
            idx_actions = (slice(None),) + tuple(state)
            new_action = np.argmax(self.Q[idx_actions])

        return new_action 

    def experience_callback(self, obs, action, new_obs, reward, done):
        """ Q Learning update: Q[S,A] = Q[S,A] + alpha*[R * gamma * max_a Q[S',a] - Q[S,A]"""
        if self.mapping_fn:
            state = self.mapping_fn(obs) 
            next_state = self.mapping_fn(new_obs)

        old_idx = (action,) + tuple(state) 
        old_q_val = self.Q[old_idx] # Q[s,a]
        best_q_val = np.max(self.Q[(slice(None),) + tuple(next_state)]) # max_a Q[s',a]

        self.Q[action, state] = (1-self.alpha) * old_q_val + \
            self.alpha * (reward + self.gamma * best_q_val) 
        
        return 

    def episode_callback(self):
        """ Reset history for new episode. Epsilon decay. """
        if self.epsilon >= self.eps_min:
            self.epsilon *= self.eps_decay
        return 

    """ Evaluation callbacks """
    def policy_callback(self, state):
        if self.mapping_fn:
            state = self.mapping_fn(state) 

        idx_actions = (slice(None),) + tuple(state)
        action = np.argmax(self.Q[idx_actions])
        return action  

    def reset(self):
        """ Reset history for new episodes in evaluation """
        return 