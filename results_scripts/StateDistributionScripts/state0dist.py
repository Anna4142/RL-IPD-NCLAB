import os
import json
import numpy as np
import matplotlib.pyplot as plt


def read_json_file(file_path):
    """Reads JSON file and returns its content."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file {file_path}: {e}")
    else:
        print(f"File {file_path} does not exist.")
    return None


def calculate_transition_probabilities(data_list):
    """Calculates transition probabilities from ChoiceCounts data."""
    transition_counts = np.zeros((4, 4))  # 4x4 matrix for CC, CD, DC, DD
    state_to_index = {'CC': 0, 'CD': 1, 'DC': 2, 'DD': 3}

    # Convert data_list into transitions
    transitions = []
    for data in data_list:
        for i in range(len(data) - 1):
            prev_state = ''.join(data[i])
            next_state = ''.join(data[i + 1])
            transitions.append((prev_state, next_state))

    # Count the transitions
    for prev_state, next_state in transitions:
        if prev_state in state_to_index and next_state in state_to_index:
            transition_counts[state_to_index[prev_state], state_to_index[next_state]] += 1

    # Debugging: Print transition counts
    print("Transition counts matrix:")
    print(transition_counts)

    # Avoid division by zero by adding a small value
    transition_probs = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-10)

    # Debugging: Print transition probabilities
    print("Transition probabilities matrix:")
    print(transition_probs)

    return transition_probs


def process_choice_counts(base_directory):
    """Processes ChoiceCounts data for all agents and hyperparameters."""
    results = {}
    for agent_folder in os.listdir(base_directory):
        agent_path = os.path.join(base_directory, agent_folder)
        if os.path.isdir(agent_path):
            results[agent_folder] = {}
            for run_folder in os.listdir(agent_path):
                run_path = os.path.join(agent_path, run_folder)
                if os.path.isdir(run_path) and run_folder.startswith("RUN"):
                    for experiment_folder in os.listdir(run_path):
                        experiment_path = os.path.join(run_path, experiment_folder)
                        choice_counts_file = os.path.join(experiment_path, "ChoiceCounts.json")
                        if os.path.exists(choice_counts_file):
                            data = read_json_file(choice_counts_file)
                            if data:
                                if experiment_folder not in results[agent_folder]:
                                    results[agent_folder][experiment_folder] = []
                                results[agent_folder][experiment_folder].append(data)
    return results


def average_transition_probabilities(transition_probabilities):
    """Averages transition probabilities across runs for each hyperparameter config."""
    averaged_results = {}
    for agent, agent_data in transition_probabilities.items():
        averaged_results[agent] = {}
        for experiment, prob_list in agent_data.items():
            avg_probs = np.mean(prob_list, axis=0)
            averaged_results[agent][experiment] = avg_probs.tolist()
    return averaged_results


def save_transition_probabilities(stats, save_dir):
    """Saves transition probabilities to JSON files."""
    for agent, agent_data in stats.items():
        agent_save_dir = os.path.join(save_dir, agent)
        os.makedirs(agent_save_dir, exist_ok=True)
        agent_save_path = os.path.join(agent_save_dir, f"{agent}_transition_probabilities.json")
        with open(agent_save_path, 'w') as file:
            json.dump(agent_data, file, indent=2)
        print(f"Transition probabilities for {agent} saved to {agent_save_path}")


def plot_transition_probabilities(stats, save_dir):
    """Plots transition probabilities for each agent."""
    state_labels = ['CC', 'CD', 'DC', 'DD']

    for agent, agent_data in stats.items():
        agent_save_dir = os.path.join(save_dir, agent)
        os.makedirs(agent_save_dir, exist_ok=True)
        for experiment, transition_probs in agent_data.items():
            plt.figure(figsize=(10, 6))
            for i, prev_state in enumerate(state_labels):
                plt.bar(np.arange(len(state_labels)) + i * 0.2, transition_probs[i], width=0.2,
                        label=f'{prev_state} ->')

            plt.title(f"{agent} {experiment} Transition Probabilities")
            plt.xlabel("Next State")
            plt.ylabel("Probability")
            plt.xticks(np.arange(len(state_labels)) + 0.3, state_labels)
            plt.legend(title="Previous State")
            plt.tight_layout()

            plot_save_path = os.path.join(agent_save_dir, f"{experiment}_transition_probabilities.png")
            plt.savefig(plot_save_path)
            plt.close()
            print(f"Plot for {agent} {experiment} saved to {plot_save_path}")


def main():
    base_directory = r"C:\Users\anush\OneDrive\study\BEN ENGLEHARD LAB\courses\technion courses\year 2 semester one\AI AND ROBOTICS\RL-IPD-NCLAB\Episodes\Clamped\MEMORY_LENGTH_3_F"
    save_directory = r"C:\Users\anush\OneDrive\study\BEN ENGLEHARD LAB\courses\technion courses\year 2 semester one\AI AND ROBOTICS\RL-IPD-NCLAB\results_across_runs\state_dist"

    all_data = process_choice_counts(base_directory)
    transition_probabilities = {}

    # Debugging: Print the structure of all_data
    print("All data structure:", json.dumps(all_data, indent=2))

    for agent, agent_data in all_data.items():
        transition_probabilities[agent] = {}
        for experiment, experiment_data in agent_data.items():
            transition_probs = calculate_transition_probabilities(experiment_data)
            if experiment not in transition_probabilities[agent]:
                transition_probabilities[agent][experiment] = []
            transition_probabilities[agent][experiment].append(transition_probs)

    # Debugging: Print the transition probabilities
    for agent, agent_data in transition_probabilities.items():
        for experiment, probs in agent_data.items():
            print(f"Transition probabilities for {agent} - {experiment}:")
            print(probs)

    averaged_transition_probabilities = average_transition_probabilities(transition_probabilities)

    # Debugging: Print the averaged transition probabilities
    print("Averaged transition probabilities:", json.dumps(averaged_transition_probabilities, indent=2))

    save_transition_probabilities(averaged_transition_probabilities, save_directory)
    plot_transition_probabilities(averaged_transition_probabilities, save_directory)


if __name__ == "__main__":
    main()
