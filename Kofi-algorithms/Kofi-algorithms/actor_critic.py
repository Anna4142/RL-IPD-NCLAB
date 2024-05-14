"""

Actor Critic implementation using PyTorch
"""

import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 

eps = np.finfo(np.float32).eps.item() # ensure std deviation is not 0

class ActorCriticNetwork(nn.Module): # The network itself is separate from the agent
    def __init__(self, learning_rate, input_dims, hidden_dims, num_actions, mapping_fn=None):
        super(ActorCriticNetwork, self).__init__()

        self.input_dims = input_dims 
        self.learning_rate = learning_rate

        self.critic_linear1 = nn.Linear(*self.input_dims, hidden_dims)
        self.critic_linear2 = nn.Linear(hidden_dims, 1)

        self.actor_linear1 = nn.Linear(*self.input_dims, hidden_dims)
        self.actor_linear2 = nn.Linear(hidden_dims, num_actions)

        self.mapping_fn = mapping_fn
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, observation):
        if self.mapping_fn:
            observation = self.mapping_fn(observation) 
        
        state = T.Tensor(observation)
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value) 

        policy = F.relu(self.critic_linear1(state))
        policy = F.softmax(self.actor_linear2(policy))

        return value, policy
    
class DA2CAgent(object): # deep actor critic
    def __init__(self, learning_rate, input_dims, num_actions, gamma=0.99, hidden_size=256, mapping_fn=None):
        self.gamma = gamma 
        self.value_mem = [] # store values
        self.reward_mem = [] # store rewards
        self.action_mem = [] # store log probabilities
        self.num_actions = num_actions
        self.gamma = gamma
        # TODO add entropy term
        self.actor_critic = ActorCriticNetwork(learning_rate, input_dims, hidden_size, num_actions, mapping_fn)

    def int_to_vector(self, action):
        """ Turns integer action into one hot vector """
        vec = np.zeros(self.num_actions)
        vec[action] = 1 
        return vec 
    
    def learn(self):
        """ Calculate rewards for the episode, compute the gradient of the loss, update optimizer"""
        returns = []
        R = 0 
        for r in self.reward_mem[::-1]:
            R = r + self.gamma*R 
            returns.insert(0,R)

        returns = T.tensor(returns, dtype=T.float)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_losses = []
        value_losses = []

        for i in range(len(self.reward_mem)):
            R = returns[i] 
            log_prob = self.action_mem[i]
            value = self.value_mem[i] 

            advantage = R - value.item() 

            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, T.tensor(R)))

        self.actor_critic.optimizer.zero_grad()
        loss = T.stack(policy_losses).sum() + T.stack(value_losses).sum()

        loss.backward()
        self.actor_critic.optimizer.step()

    """ Training Callbacks """
    def action_callback(self, observation):
        value, probs = self.actor_critic.forward(observation)
        
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)

        self.action_mem.append(log_probs)
        self.value_mem.append(value)
        
        return action.item()

    def experience_callback(self, obs, action, new_obs, reward, done):
        self.reward_mem.append(reward)

    def episode_callback(self):
        """ Reset at the end of an episode"""
        self.learn() 
        self.reward_mem = [] 
        self.action_mem = [] 
        self.value_mem = []

    """ Evaluation Callbacks """
    def policy_callback(self, observation):
        _, probs = self.actor_critic.forward(observation)
        
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()

        return action.item() 

    def reset(self):
        """ Reset an the end of an episode with no further training"""
        return 
