# visualization.py
import matplotlib.pyplot as plt

# visualization.py
import matplotlib.pyplot as plt
from Evaluation.Metrics import CumulativeRewardMetric, AverageRewardMetric, CooperationRateMetric

class MetricsVisualizer:
    def __init__(self,databuffer):
        self.data_buffer = databuffer
        self.cumulative_reward_metric = CumulativeRewardMetric(self.data_buffer)
        self.average_reward_metric = AverageRewardMetric(self.data_buffer,window_size=100)
        self.cooperation_rate_metric = CooperationRateMetric(self.data_buffer)

    def update_metrics(self, reward1,reward2, action1,action2):

        self.data_buffer.update_reward1(reward1)
        self.data_buffer.update_reward2(reward2)
        self.data_buffer.update_action1(action1)
        self.data_buffer.update_action2(action2)

        self.cumulative_reward_metric.update(reward1,reward2)
        self.average_reward_metric.update(reward1,reward2)
        self.cooperation_rate_metric.update(action1,action2)

    def plot_cumulative_rewards(self):
        cumulative_rewards = self.cumulative_reward_metric.get_metrics()
        print("cumreward",cumulative_rewards)
        """""
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_rewards, label='Cumulative Reward')
        plt.title('Cumulative Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()
        """

    def plot_average_rewards(self):
        average_rewards = self.average_reward_metric.get_metrics()
        print("avg reward",average_rewards)
        """""
        plt.figure(figsize=(12, 6))
        plt.plot(average_rewards, label='Average Reward')
        plt.title('Average Reward over Last 100 Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.show()
        """

    def print_cooperation_rate(self):
        cooperation_rate = self.cooperation_rate_metric.get_metrics()
        print(f"Overall Cooperation Rate: {cooperation_rate}")
    def plot_all(self):
        # Call the individual plotting functions
        self.plot_cumulative_rewards()
        self.plot_average_rewards()
        # And print the cooperation rate
        self.print_cooperation_rate()

