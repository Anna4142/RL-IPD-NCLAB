# visualization.py
import matplotlib.pyplot as plt

# visualization.py
import matplotlib.pyplot as plt


from Evaluation.Metrics import CumulativeRewardMetric, AverageRewardMetric, CooperationRateMetric, ChoicePercentageMetric,ChoiceCountMetric,ForcedActionsMetric


import os
class MetricsVisualizer:
    def __init__(self,databuffer):

        self.data_buffer = databuffer

        self.cumulative_reward_metric = CumulativeRewardMetric(self.data_buffer)
        self.average_reward_metric = AverageRewardMetric(self.data_buffer,window_size=100)
        self.cooperation_rate_metric = CooperationRateMetric(self.data_buffer)
        self.choice_percentage_metric=ChoicePercentageMetric(self.data_buffer)
        self.choice_count_metric = ChoiceCountMetric(self.data_buffer)
        self.forced_actions_metric = ForcedActionsMetric(self.data_buffer)
    def update_metrics(self, reward1,reward2, action1,action2,mouse_hist_action):

        self.data_buffer.update_reward1(reward1)
        self.data_buffer.update_reward2(reward2)
        self.data_buffer.update_action1(action1)
        self.data_buffer.update_action2(action2)

        self.cumulative_reward_metric.update(reward1,reward2)
        self.average_reward_metric.update(reward1,reward2)
        self.cooperation_rate_metric.update(action1,action2)
        self.choice_percentage_metric.update(action1, action2)
        self.choice_percentage_metric.update(action1, action2)
        self.choice_count_metric.update(action1, action2)
        self.forced_actions_metric.update(action1, mouse_hist_action)
        print("MOUSR HIST ",mouse_hist_action)

    def format_state(self, action1, action2):
        # Converts action numbers to state string, e.g., 0,1 to "CD"
        return self.action_label(action1) + self.action_label(action2)

    def action_label(self, action):
        # Maps numeric action to character labels 'C' for Cooperate (0) and 'D' for Defect (1)
        return 'C' if action == 0 else 'D'
    def print_cooperation_rate(self):
        cooperation_rate = self.cooperation_rate_metric.get_metrics()
        print(f"Overall Cooperation Rate: {cooperation_rate}")

    def save_all_results_and_plots(self, experiment_id, experiment_number):
        # Save results first
        self.cumulative_reward_metric.save_results(experiment_id, "cumulative_rewards.json", experiment_number)
        self.average_reward_metric.save_results(experiment_id, "average_rewards.json", experiment_number)
        self.cooperation_rate_metric.save_results(experiment_id, "cooperation_rate.json", experiment_number)
        self.choice_percentage_metric.save_results(experiment_id, "ChoicePercentages.json", experiment_number)
        self.choice_count_metric.save_results(experiment_id, 'ChoiceCounts.json', experiment_number)
        self.forced_actions_metric.save_results(experiment_id, 'forced_actions.json', experiment_number)







