import numpy as np 
import numpy.random as npr 

"""
Actor critic without neural networks. Not necessarily functional. Mainly for academic curiosity.
Combines policy gradient wtih Q-learning under an actor critic framework.
"""
class A2CVanillaAgent(object):
    def __init__(self, num_actions, theta, q_table_dim, alpha=0.00025, gamma=0.99,
            pg_mapping_fn=None, q_mapping_fn=None):
        """ Parameters """
        self.alpha = alpha 
        self.gamma = gamma 

        self.theta = theta

        """ Record keeping - Actor """
        self.grad_mem = []
        self.reward_mem = []
        self.value_mem = []

        """ Initialize Q table """
        self.dim_len = len(q_table_dim)

        """ Parameter Checking """
        if num_actions != q_table_dim[0]:
            raise ValueError('Q table dimension must start with action dimension')
        if num_actions != len(self.theta[-1]):
            raise ValueError('Theta dimension must end with action dimension')

        self.pg_mapping_fn = pg_mapping_fn
        self.q_mapping_fn = q_mapping_fn 

        self.num_actions = num_actions 
        self.Q = np.zeros(q_table_dim)

    def policy(self, state):
        """ Returns agent policy given state """
        probs = self.softmax(state)
        return probs
        
    def softmax(self, state):
        """ softmax(state * weights) """
        z = state.dot(self.theta)
        exp = np.exp(z)
        return exp/np.sum(exp)

    def softmax_gradient(self, softmax):
        """ Derivative of the softmax w.r.t. theta """
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    
    def compute_gradient(self, probs, state, action):
        """ Computes the gradient of log(softmax) for a state and action """
        dsoftmax = self.softmax_gradient(probs)[action, :]
        dlog = dsoftmax / probs[0, action]
        grad = state.T.dot(dlog[None,:])
        
        return grad 

    def update_weights(self):
        """ Update theta weights based on advantage w.r.t. Q Learning """
        returns = []
        R = 0 
        for r in self.reward_mem[::-1]:
            R = r + self.gamma*R 
            returns.insert(0,R)

        for i in range(len(self.reward_mem)):
            R = returns[i] 
            value = self.value_mem[i] 

            # advantage = R - value

            self.theta += self.alpha * self.grad_mem[i] * value 
        return

    """ Training Callbacks """
    def action_callback(self, observation):
        """ Act according to the actor. Store the value of the critic & the gradient of the actor"""
        if self.pg_mapping_fn:
            state = self.pg_mapping_fn(observation)
        
        state = state[None,:]
        """ Take action from the Actor """
        probs = self.policy(state)
        action = np.random.choice(self.num_actions, p=probs[0])
        grad = self.compute_gradient(probs, state, action)
            
        self.grad_mem.append(grad)

        """ Get Critic Value """
        state = observation 
        if self.q_mapping_fn:
            state = self.q_mapping_fn(state)

        critic_idx = (action,) + tuple(state)
        value = self.Q[critic_idx]
        self.value_mem.append(value)

        return action 
    
    def experience_callback(self,obs, action, new_obs, reward, done):
        if self.q_mapping_fn:
            state = self.q_mapping_fn(obs)
            next_state = self.q_mapping_fn(new_obs)

        """ update critic """
        old_idx = (action,) + tuple(state)
        old_q_val = self.Q[old_idx]
        best_val = np.max(self.Q[(slice(None),) + tuple(next_state)])

        self.Q[action, state] = (1-self.alpha) * old_q_val + \
            self.alpha * (reward + self.gamma * best_val)

        self.reward_mem.append(reward)
        return 
    
    def episode_callback(self):
        self.update_weights()

        self.grad_mem = [] 
        self.reward_mem = []
        self.value_mem = []

    """ Evaluation Callbacks """
    def policy_callback(self):
        if self.pg_mapping_fn:
            state = self.pg_mapping_fn(observation)
        
        state = state[None,:]

        """ Take action from the Actor """
        probs = self.policy(state)
        action = np.random.choice(self.num_actions, p=probs[0])
        return action 
        
    def reset(self):
        return 

