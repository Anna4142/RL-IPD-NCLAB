import os
import json
import matplotlib.pyplot as plt
import numpy as np
class BaseMetric:
    def __init__(self, data_buffer):
        self.data_buffer = data_buffer

    def update(self):
        # Default implementation attempts to pull data from the data_buffer.
        # Derived classes should implement how they use this data.
        raise NotImplementedError("The update method must be implemented by the subclass and utilize the data_buffer.")

    def get_metrics(self):
        raise NotImplementedError("The get_metrics method must be implemented by the subclass.")

    def reset(self):
        raise NotImplementedError("The reset method must be implemented by the subclass.")



    def get_next_experiment_number(self,experiment_id):
        base_directory = "Episodes"
        experiment_base_dir = os.path.join(base_directory, experiment_id)

        if not os.path.exists(experiment_base_dir):
            os.makedirs(experiment_base_dir)
            return 1  # First experiment if none exist yet

        existing_dirs = [
            d for d in os.listdir(experiment_base_dir)
            if os.path.isdir(os.path.join(experiment_base_dir, d)) and d.startswith("experiment_")
        ]
        existing_numbers = [int(d.split("_")[-1]) for d in existing_dirs]

        if existing_numbers:
            return max(existing_numbers) + 1
        else:
            return 1

    def save_results(self, experiment_id, filename, experiment_number):
        # Directory for saving results
        directory = os.path.join("Episodes", experiment_id, f"{experiment_number}")
        os.makedirs(directory, exist_ok=True)

        # Path where the metric data will be saved
        data_filepath = os.path.join(directory, filename)

        # Assuming the metric data can be serialized to JSON
        data_to_save = self.get_metrics()


        with open(data_filepath, "w") as file:
            json.dump(data_to_save, file)
        print(f"Saved {filename} to {data_filepath}")

        # Generate a plot based on the metric data
        plt.figure(figsize=(12, 6))

        # Assuming `data_to_save` is a list of values; adjust this according to your data's structure
        plt.plot(data_to_save, label=filename.split('.')[0])  # Use the filename (without extension) as the label
        plt.title(f"{filename.split('.')[0].replace('_', ' ').capitalize()} per Episode")
        plt.xlabel('Episode')

        # Use the filename as part of the ylabel, removing underscores and capitalizing
        ylabel = filename.split('.')[0].replace('_', ' ').capitalize()
        plt.ylabel(ylabel)

        plt.legend()

        # Save the plot with a corresponding filename, changing the extension to .png
        plot_filename = filename.replace('.json', '.png')
        plot_filepath = os.path.join(directory, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Plot saved to {plot_filepath}")

        plt.close()  # Close the figure to free up memory