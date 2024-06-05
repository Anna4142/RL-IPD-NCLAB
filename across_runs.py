import os
import json
import numpy as np
import matplotlib.pyplot as plt


def read_json_file(file_path):
    """Reads JSON file and returns its content."""
    if os.path.exists(file_path):
        print(f"Reading file {file_path}")
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                if data:
                    return data
                else:
                    print(f"File {file_path} is empty.")
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file {file_path}: {e}")
    else:
        print(f"File {file_path} does not exist.")
    return None


def calculate_stats_across_runs(data_list):
    """Calculates mean, variance, and standard deviation across runs."""
    data_array = np.array(data_list)
    mean = np.mean(data_array, axis=0)
    variance = np.var(data_array, axis=0)
    std_dev = np.sqrt(variance)
    return mean.tolist(), variance.tolist(), std_dev.tolist()


def process_hyperparameter_runs(agent_path, filename):
    """Processes all runs and hyperparameters within the agent path."""
    all_hyperparams_data = {}

    # Loop over all run folders within the agent path
    for run_folder in os.listdir(agent_path):
        run_path = os.path.join(agent_path, run_folder)
        print(f"Checking run directory: {run_path}")
        if os.path.isdir(run_path) and run_folder.startswith("RUN"):
            # Process each experiment folder
            for subdirectory in os.listdir(run_path):
                experiment_path = os.path.join(run_path, subdirectory)
                file_path = os.path.join(experiment_path, filename)
                print(f"Checking experiment directory: {experiment_path}")
                if os.path.exists(file_path):
                    print(f"Found file: {file_path}")
                    if subdirectory not in all_hyperparams_data:
                        all_hyperparams_data[subdirectory] = []
                    data = read_json_file(file_path)
                    if data:
                        all_hyperparams_data[subdirectory].append(data)
                else:
                    print(f"File {file_path} does not exist.")
        else:
            print(f"Run path is not a directory: {run_path}")

    stats = {}
    for hyperparam, data_list in all_hyperparams_data.items():
        if data_list:
            mean, variance, std_dev = calculate_stats_across_runs(data_list)
            stats[hyperparam] = (mean, variance, std_dev)
        else:
            print(f"No data found for hyperparameter: {hyperparam}")

    return stats


def save_stats_to_json(stats, save_dir, filename_prefix, subdir):
    """Saves statistics to JSON files in specified directories."""
    os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
    stats_json = {}
    for hyperparam, (mean, variance, std_dev) in stats.items():
        stats_json[hyperparam] = {
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev
        }

    save_path = os.path.join(save_dir, subdir, f"{filename_prefix}_stats.json")
    with open(save_path, 'w') as file:
        json.dump(stats_json, file)
    print(f"Statistics saved to {save_path}")
    print(f"{subdir.capitalize()} statistics saved to: {save_path}")


def plot_results(stats, title, save_dir):
    """Plots results with standard deviation and prints folders with extreme mean values."""
    plt.figure(figsize=(12, 6))
    if not stats:
        print("No data to plot.")
        return

    sorted_stats = sorted(stats.items(), key=lambda x: np.mean(x[1][0]))
    lowest_mean = sorted_stats[0][0]
    highest_mean = sorted_stats[-1][0]
    top_3 = sorted_stats[-3:]
    bottom_3 = sorted_stats[:3]

    for hyperparam, (mean, _, std_dev) in stats.items():
        x = range(len(mean))
        label = f'{hyperparam}'
        plt.plot(x, mean, label=label)
        plt.fill_between(x, np.array(mean) - np.array(std_dev), np.array(mean) + np.array(std_dev), alpha=0.3)

    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Save the plot for all results
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

    # Plot and save top 3 results
    plt.figure(figsize=(12, 6))
    for hyperparam, (mean, _, std_dev) in top_3:
        x = range(len(mean))
        label = f'{hyperparam} (Top 3)'
        plt.plot(x, mean, label=label)
        plt.fill_between(x, np.array(mean) - np.array(std_dev), np.array(mean) + np.array(std_dev), alpha=0.3)

    plt.title(f"Top 3 {title}")
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    top_3_save_path = os.path.join(save_dir, f"Top_3_{title.replace(' ', '_')}.png")
    plt.savefig(top_3_save_path)
    print(f"Top 3 plot saved to {top_3_save_path}")
    plt.show()

    # Plot and save bottom 3 results
    plt.figure(figsize=(12, 6))
    for hyperparam, (mean, _, std_dev) in bottom_3:
        x = range(len(mean))
        label = f'{hyperparam} (Bottom 3)'
        plt.plot(x, mean, label=label)
        plt.fill_between(x, np.array(mean) - np.array(std_dev), np.array(mean) + np.array(std_dev), alpha=0.3)

    plt.title(f"Bottom 3 {title}")
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    bottom_3_save_path = os.path.join(save_dir, f"Bottom_3_{title.replace(' ', '_')}.png")
    plt.savefig(bottom_3_save_path)
    print(f"Bottom 3 plot saved to {bottom_3_save_path}")
    plt.show()

    print(f"Maximum mean is in folder: {highest_mean}")
    print(f"Minimum mean is in folder: {lowest_mean}")


def main():
    """Main function to process data and plot results."""
    base_directory = r"C:\Users\anush\OneDrive\study\BEN ENGLEHARD LAB\courses\technion courses\year 2 semester one\AI AND ROBOTICS\RL-IPD-NCLAB\Not_Clamped\MEMORY_LENGTH_1_F"
    results_directory = r"C:\Users\anush\OneDrive\study\BEN ENGLEHARD LAB\courses\technion courses\year 2 semester one\AI AND ROBOTICS\RL-IPD-NCLAB\results_across_runs"

    # Check if the base directory exists
    if not os.path.exists(base_directory):
        print(f"Base directory does not exist: {base_directory}")
        # Print the actual structure of the given path
        path_parts = base_directory.split(os.sep)
        for i in range(1, len(path_parts) + 1):
            partial_path = os.sep.join(path_parts[:i])
            if os.path.exists(partial_path):
                print(f"Directory exists: {partial_path}")
            else:
                print(f"Directory does not exist: {partial_path}")
        return

    # Extract memory length from base directory
    memory_length = os.path.basename(base_directory)
    print(f"Memory length: {memory_length}")

    # Loop over each agent folder
    for agent_folder in os.listdir(base_directory):
        agent_path = os.path.join(base_directory, agent_folder)
        if os.path.isdir(agent_path):
            print(f"Processing agent folder: {agent_path}")

            # Create subdirectories in results directory based on memory length and agent folder
            agent_results_directory = os.path.join(results_directory, memory_length, agent_folder)
            os.makedirs(agent_results_directory, exist_ok=True)
            print(f"Results directory for agent: {agent_results_directory}")

            # Create subdirectories for averages and variances
            averages_directory = os.path.join(agent_results_directory, 'averages')
            variances_directory = os.path.join(agent_results_directory, 'variances')
            os.makedirs(averages_directory, exist_ok=True)
            os.makedirs(variances_directory, exist_ok=True)

            # Create subdirectories for top and bottom performers
            top_averages_directory = os.path.join(averages_directory, 'top')
            bottom_averages_directory = os.path.join(averages_directory, 'bottom')
            top_variances_directory = os.path.join(variances_directory, 'top')
            bottom_variances_directory = os.path.join(variances_directory, 'bottom')
            os.makedirs(top_averages_directory, exist_ok=True)
            os.makedirs(bottom_averages_directory, exist_ok=True)
            os.makedirs(top_variances_directory, exist_ok=True)
            os.makedirs(bottom_variances_directory, exist_ok=True)

            files_to_process = ["cumulative_rewards.json", "average_rewards.json", "cooperation_rate.json"
                             ]

            for filename in files_to_process:
                print(f"Processing {filename} for agent {agent_folder}")
                stats = process_hyperparameter_runs(agent_path, filename)
                if stats:
                    plot_title = f"{filename.replace('.json', '').replace('_', ' ').title()}"
                    plot_results(stats, plot_title, agent_results_directory)

                    # Identify top performers (highest 3)
                    top_performers = dict(sorted(stats.items(), key=lambda x: np.mean(x[1][0]))[-3:])
                    # Identify bottom performers (lowest 3)
                    bottom_performers = dict(sorted(stats.items(), key=lambda x: np.mean(x[1][0]))[:3])

                    # Save averages and variances for top performers
                    save_stats_to_json(top_performers, top_averages_directory, filename.replace('.json', ''), '')
                    save_stats_to_json(top_performers, top_variances_directory, filename.replace('.json', ''), '')

                    # Print the paths for top averages and variances
                    print(f"Top averages saved to: {top_averages_directory}")
                    print(f"Top variances saved to: {top_variances_directory}")

                    # Save averages and variances for bottom performers
                    save_stats_to_json(bottom_performers, bottom_averages_directory, filename.replace('.json', ''), '')
                    save_stats_to_json(bottom_performers, bottom_variances_directory, filename.replace('.json', ''), '')

                    # Print the paths for bottom averages and variances
                    print(f"Bottom averages saved to: {bottom_averages_directory}")
                    print(f"Bottom variances saved to: {bottom_variances_directory}")
                else:
                    print(f"No statistics generated for {filename}")


if __name__ == "__main__":
    main()




