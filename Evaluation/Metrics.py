from Evaluation.BaseMetric import BaseMetric
import csv
import csv
from os.path import exists


class CumulativeRewardMetric(BaseMetric):
    def __init__(self, data_buffer):
        super().__init__(data_buffer)
        self.reset()  # Initialize cumulative_rewards as an empty list

    def update(self, reward1=None, reward2=None):
        # Check if rewards are not None for safety
        if reward1 is not None and reward2 is not None:
            # Calculate the new reward to be added
            latest_reward = reward1 + reward2

            # If the list is empty, add the latest_reward directly
            if not self.cumulative_rewards:
                self.cumulative_rewards.append(latest_reward)
            else:
                # Otherwise, add the latest_reward to the last cumulative reward in the list
                self.cumulative_rewards.append(self.cumulative_rewards[-1] + latest_reward)

    def get_metrics(self):
        """Retrieve the list of cumulative rewards."""
        return self.cumulative_rewards

    def reset(self):
        """Reset the cumulative rewards list."""
        self.cumulative_rewards = []
    def save_results(self, experiment_id, expnum,filename="cumulative_rewards.json"):
        super().save_results(experiment_id, expnum,filename)

class AverageRewardMetric(BaseMetric):
    def __init__(self, data_buffer, window_size=1):
        super().__init__(data_buffer)  # Pass data_buffer to BaseMetric
        self.window_size = window_size
        self.reset()

    def update(self, reward1, reward2):
        # Directly use the rewards passed to it for updating
        avg_reward = reward1 + reward2
        self.all_rewards.append(avg_reward)

        # Perform time averaging based on the window size
        if len(self.all_rewards) >= self.window_size:
            window_rewards = self.all_rewards[-self.window_size:]
            window_avg_reward = sum(window_rewards) / self.window_size
        self.average_rewards.append(avg_reward)

    def get_metrics(self):
        return self.average_rewards

    def reset(self):
        self.all_rewards = []
        self.average_rewards = []

    def save_results(self, experiment_id, expnum,filename="average_rewards.json"):
        super().save_results(experiment_id, expnum,filename)
class CooperationRateMetric(BaseMetric):
    def __init__(self, data_buffer):
        super().__init__(data_buffer)  # Pass data_buffer to BaseMetric
        self.reset()

    def update(self, action1, action2):
        self.total_actions += 1
        if action1 == 0 and action2 == 0:
            self.cooperation_count += 1

        # Calculate current cooperation percentage and record it with trial number
        current_percentage = (self.cooperation_count / self.total_actions) * 100
        self.cooperation_percentages.append(current_percentage)

    def get_metrics(self):
        if self.total_actions == 0:
            return 0
        return self.cooperation_percentages

    def reset(self):
        self.cooperation_count = 0
        self.total_actions = 0
        self.cooperation_percentages = []  # Store tuples of (trial_number, cooperation_percentage)
    def save_results(self, experiment_id, expnum,filename="cooperation_rate.json"):

        super().save_results(experiment_id, expnum,filename)
class ChoicePercentageMetric(BaseMetric):
    def __init__(self, data_buffer):
        super().__init__(data_buffer)
        self.reset()

    def update(self, action1, action2):
        # Update counts based on actions
        key = (action1, action2)
        self.action_counts[key] = self.action_counts.get(key, 0) + 1

    def get_metrics(self):
        # Calculate the percentage of each choice
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return {k: 0 for k in self.action_counts.keys()}
        return {k: (v / total_actions) * 100 for k, v in self.action_counts.items()}

    def reset(self):
        self.action_counts = {}

    def save_results(self, experiment_id,   filename):
        metrics = self.get_metrics()
        # Prepare row data
        row_data = [experiment_id] + [metrics.get((i, j), 0) for i in range(2) for j in range(2)]

        file_exists = exists(filename)
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write headers if file does not exist
                headers = ['ExperimentID',   'CC', 'CD', 'DC', 'DD']
                writer.writerow(headers)
            writer.writerow(row_data)