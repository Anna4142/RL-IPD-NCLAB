import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
import numpy.random as npr
import time

""" 
PyTorch implementation of basic DQN 
- no second target network (TODO)
- two hidden layers, relu activation functions
- Adam optimizer 
- Network is the same as the policy gradient network 
- Experience replay
- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""
class DeepQNetwork(nn.Module):
    """ Generic DQN implementation """
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, num_actions):
        super(DeepQNetwork, self).__init__()


        self.input_dims = input_dims 
        self.learning_rate = learning_rate
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims
        self.num_actions = num_actions 
        self.loss = nn.MSELoss()

        self.fc1 = nn.Linear(*self.input_dims, self.fc2_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions 

class DQNAgent(object):
    def __init__(self, learning_rate, gamma, epsilon, input_dims, batch_size, num_actions, l1_size=256, l2_size=256,
        max_mem_size=100000, mapping_fn=None): 
    
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon 
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.mapping_fn = mapping_fn

        self.mem_idx = 0 

        self.Q_eval = DeepQNetwork(self.learning_rate, num_actions=num_actions, input_dims=input_dims,
            fc1_dims=l1_size, fc2_dims=l2_size)

        self.state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32) # TODO: dynamic replay buffer
        self.new_state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool)

    def int_to_vector(self, action):
        """ Turns integer action into one hot vector """
        vec = np.zeros(self.num_actions)
        vec[action] = 1 
        return vec 

    
    def learn(self):
        if self.mem_idx < self.batch_size:
            return 
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_idx, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_idx = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_mem[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_mem[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_mem[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_mem[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_mem[batch] 

        q_eval = self.Q_eval.forward(state_batch)[batch_idx, action_batch]
        q_next = self.Q_eval.forward(new_state_batch) 
        q_next[terminal_batch] = 0.0 

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0] # max of next state
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        T.cuda.empty_cache() 
        """ Epsilon decay """
        # if self.epsilon > 0.01:
        #     self.epsilon -= 0.005

    """ Training Callbacks """
    def experience_callback(self, state, action, new_state, reward, done):
        # update memory
        if self.mapping_fn:
            state = self.mapping_fn(state) 
            new_state = self.mapping_fn(new_state) 
            
        idx = self.mem_idx % self.mem_size 
        self.state_mem[idx] = state 
        self.new_state_mem[idx] = new_state 
        self.reward_mem[idx] = reward 
        self.action_mem[idx] = action 
        self.terminal_mem[idx] = done 

        self.mem_idx += 1
        
        if self.mem_idx % 100 == 0:
            self.learn() 

    def action_callback(self, observation):
        if self.mapping_fn:
            observation = self.mapping_fn(observation)

        if npr.uniform(0,1) < self.epsilon:
            action = npr.randint(0, self.num_actions)
        else:
            state = T.tensor(observation,dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state) 
            action = T.argmax(actions).item() 
        
        return action
    
    def episode_callback(self):
        return 

    """ Evaluation Callbacks """
    def policy_callback(self, observation):
        if self.mapping_fn:
            observation = self.mapping_fn(observation)
        
        if npr.uniform(0,1) < self.epsilon:
            action = npr.randint(0, self.num_actions)
        else:
            state = T.tensor(observation,dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state) 
            action = T.argmax(actions).item() 
        
        return action

    def reset(self):
        return 