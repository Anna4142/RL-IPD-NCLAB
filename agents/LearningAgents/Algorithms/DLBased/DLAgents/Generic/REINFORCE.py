import torch
import torch.optim as optim
import random
import torch.nn.functional as F
from agents.LearningAgents.Algorithms.DLBased.Networks.GenericNN import FullyConnected



class REINFORCEAgent:
    def __init__(self, env):
        print("state size in re",env.state_size)
        self.state_size = env.state_size
        self.action_size = env.action_size
        # Assuming env also provides a list of hidden layer sizes
        hidden_layers = env.hidden_layers if hasattr(env, 'hidden_layers') else [128, 128]


        self.policy_net= FullyConnected([self.state_size] + hidden_layers + [self.action_size])
        self.learning_rate=0.01
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def decide_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Adds a batch dimension
        print(self.policy_net)

        logits = self.policy_net(state)
        action_probs = F.softmax(logits, dim=-1)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        return action.item(), log_prob

    def learn(self, rewards, log_probs, gamma=0.99):   ##updating policy
        discounts = [gamma ** i for i in range(len(rewards))]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
