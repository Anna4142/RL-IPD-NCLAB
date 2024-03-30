# visualization.py
import matplotlib.pyplot as plt

# visualization.py
import matplotlib.pyplot as plt
from Evaluation.Metrics import CumulativeRewardMetric, AverageRewardMetric, CooperationRateMetric
import os
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


    def print_cooperation_rate(self):
        cooperation_rate = self.cooperation_rate_metric.get_metrics()
        print(f"Overall Cooperation Rate: {cooperation_rate}")

    def save_all_results_and_plots(self, experiment_id,experiment_number):

        self.cumulative_reward_metric.save_results(experiment_id, "cumulative_rewards.json", experiment_number)
        self.average_reward_metric.save_results(experiment_id, "average_rewards.json", experiment_number)
        self.cooperation_rate_metric.save_results(experiment_id, "cooperation_rate.json", experiment_number)




