from Evaluation.BaseMetric import BaseMetric
import csv
import csv
import os
from os.path import exists
from os.path import exists, dirname, join
import matplotlib.pyplot as plt
import json
import numpy as np
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
class ChoiceCountMetric(BaseMetric):
    def __init__(self, data_buffer):
        super().__init__(data_buffer)
        self.choices = []  # List to store each choice pair

    def update(self, action1, action2):
        # Log the choices for each iteration
        self.choices.append((action1, action2))

    def get_metrics(self):
        # Return the list of all choice pairs
        return self.choices

    def reset(self):
        # Reset the list of choices
        self.choices = []

    def save_results(self, experiment_id, filename, experiment_number):
        # Directory for saving results
        directory = experiment_id
        os.makedirs(directory, exist_ok=True)

        # Full file path for saving the choices
        filepath = os.path.join(directory, filename)

        # Convert each tuple in the list to a string that indicates choice labels
        choice_labels = [(self._action_label(a), self._action_label(b)) for a, b in self.choices]

        # Save the choices as JSON
        with open(filepath, 'w') as file:
            json.dump(choice_labels, file)
        print(f"Choices saved to {filepath}")

    def _action_label(self, action):
        # Helper method to convert action numbers to labels
        return 'C' if action == 0 else 'D'
class ChoicePercentageMetric(BaseMetric):
    def __init__(self, data_buffer):
        super().__init__(data_buffer)
        self.reset()

    def update(self, action1, action2):
        key = (action1, action2)
        self.action_counts[key] = self.action_counts.get(key, 0) + 1

    def get_metrics(self):
        # Calculate the percentage of each choice
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return {self._format_key(k): 0 for k in self.action_counts.keys()}
        return {self._format_key(k): (v / total_actions) * 100 for k, v in self.action_counts.items()}

    def reset(self):
        self.action_counts = {}

    def save_results(self, experiment_id, filename, experiment_number):
        # Ensure directory structure and filename as per BaseMetric structure
        directory = experiment_id
        os.makedirs(directory, exist_ok=True)

        # Save metrics in JSON format
        json_filepath = os.path.join(directory, filename)
        data_to_save = self.get_metrics()
        with open(json_filepath, 'w') as file:
            json.dump(data_to_save, file)
        print(f"Saved JSON to {json_filepath}")

        # Save metrics in CSV format
        csv_filepath = os.path.join(directory, filename.replace('.json', '.csv'))
        file_exists = os.path.exists(csv_filepath)
        with open(csv_filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                headers = ['ExperimentID', 'ExperimentNum'] + list(data_to_save.keys())
                writer.writerow(headers)
            row_data = [experiment_id, experiment_number] + list(data_to_save.values())
            writer.writerow(row_data)
        print(f"Saved CSV to {csv_filepath}")

        # Generate and save histogram
        self.save_histogram(data_to_save, directory, filename)

    def save_histogram(self, metrics, directory, filename):
        labels = list(metrics.keys())
        values = list(metrics.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='blue')
        plt.xlabel('Action Combinations')
        plt.ylabel('Percentage (%)')
        plt.title('Choice Distribution Histogram')
        plt.xticks(rotation=45)  # Rotate labels for better visibility

        histogram_filepath = os.path.join(directory, filename.replace('.json', '_histogram.png'))
        #plt.savefig(histogram_filepath)
        plt.close()
        print(f"Histogram saved to {histogram_filepath}")

    def _format_key(self, key):
        # Convert numeric key tuple to string format CC, CD, DC, DD
        action_map = {0: 'C', 1: 'D'}
        return action_map[key[0]] + action_map[key[1]]


class ForcedActionsMetric(BaseMetric):
    def __init__(self, data_buffer):
        super().__init__(data_buffer)
        self.reset()

    def update(self, action2, mouse_hist_action):
        # Debug prints to trace values
        print(f"Updating with action2: {action2}, mouse_hist_action: {mouse_hist_action}")
        if action2 is None or mouse_hist_action is None:
            print("Warning: One of the inputs to update is None.")
        self.agent_actions.append(action2)
        self.forced_actions.append(mouse_hist_action)
        self.calculate_similarity_score()

    def get_metrics(self):
        print(" get metrics forced_actions", self.forced_actions)
        return {
            "agent_actions": self.agent_actions,
            "forced_actions": self.forced_actions,
            "similarity_scores": self.similarity_scores,
            "running_averages": self.running_averages
        }

    def reset(self):
        self.agent_actions = []
        self.forced_actions = []
        self.similarity_scores = []
        self.running_averages = []
        self.cumulative_sum = 0
        self.iteration_count = 0

    def save_results(self, experiment_id, filename, experiment_number):
        directory = experiment_id
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        metrics = self.get_metrics()
        with open(filepath, 'w') as file:
            json.dump(metrics, file)
        print(f"Forced actions saved to {filepath}")

        # Save the trajectory comparison plot
        trajectory_plot_filepath = os.path.join(directory, 'trajectory_comparison.png')
        #self.plot_trajectory_comparison(trajectory_plot_filepath)

        # Save the running average plot
        running_average_plot_filepath = os.path.join(directory, 'running_average.png')
        self.plot_running_average(running_average_plot_filepath)

    def plot_trajectory_comparison(self, filepath):
        if not self.forced_actions or not self.agent_actions:
            print("Error: Empty forced_actions or agent_actions lists.")
            return

        forced_actions_array = np.asarray(self.forced_actions)
        agent_actions_array = np.asarray(self.agent_actions)

        plt.figure(figsize=(10, 6))
        plt.plot(forced_actions_array, label='Forced Actions')
        plt.plot(agent_actions_array, label='Agent Actions')
        plt.xlabel('Time Step')
        plt.ylabel('Action')
        plt.title('Trajectory Comparison')
        plt.legend()
        plt.savefig(filepath)
        plt.close()
        print(f"Trajectory comparison plot saved to {filepath}")

    def calculate_similarity_score(self):
        if len(self.forced_actions) > 0 and len(self.agent_actions) > 0:
            score = int(self.forced_actions[-1] == self.agent_actions[-1])
            self.similarity_scores.append(score)
            self.cumulative_sum += score
            self.iteration_count += 1
            running_average = self.cumulative_sum / self.iteration_count
            self.running_averages.append(running_average)

    def plot_running_average(self, filepath):
        if not self.running_averages:
            print("Error: Empty running_averages list.")
            return

        running_averages_array = np.asarray(self.running_averages)

        plt.figure(figsize=(10, 6))
        plt.plot(running_averages_array)
        plt.xlabel('Iteration')
        plt.ylabel('Running Average of Similarity Score')
        plt.title('Running Average of Similarity Score vs Iterations')
        plt.savefig(filepath)
        plt.close()
        print(f"Running average plot saved to {filepath}")