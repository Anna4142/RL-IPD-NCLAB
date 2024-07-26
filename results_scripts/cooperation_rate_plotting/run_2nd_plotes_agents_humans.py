import os
import json
import numpy as np
import matplotlib.pyplot as plt


def read_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file {file_path}: {e}")
    else:
        print(f"File {file_path} does not exist.")
    return None


def plot_combined_results(agents_data, human_data, title, ylabel, save_path):
    plt.figure(figsize=(12, 6))

    for agent, agent_data in agents_data.items():
        x = range(len(agent_data['mean']))
        plt.plot(x, agent_data['mean'], label=f'{agent} (Mean)')
        plt.fill_between(x,
                         np.array(agent_data['mean']) - np.sqrt(np.array(agent_data['variance'])),
                         np.array(agent_data['mean']) + np.sqrt(np.array(agent_data['variance'])),
                         alpha=0.2)

    # Plot human data
    x = range(len(human_data['average']))
    plt.plot(x, human_data['average'], label='Human (Mean)', color='black', linewidth=2)
    plt.fill_between(x,
                     np.array(human_data['average']) - np.sqrt(np.array(human_data['variance'])),
                     np.array(human_data['average']) + np.sqrt(np.array(human_data['variance'])),
                     alpha=0.2, color='gray')

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    base_path = r"C:\Users\anush\OneDrive\study\BEN ENGLEHARD LAB\courses\technion courses\year 2 semester one\AI AND ROBOTICS\RL-IPD-NCLAB"
    agents_path = os.path.join(base_path, "results_across_runs","Clamped","CC_LowVar","forced_action_running_average_stats", "MEMORY_LENGTH_1_F")
    human_avg_path = os.path.join(base_path, "HUMAN_SPLIT", "Top_bottom_coop_rates", "bottom",
                                  "average_bottom_0_7_cooperation_rate.json")
    human_var_path = os.path.join(base_path, "HUMAN_SPLIT", "Top_bottom_coop_rates", "bottom",
                                  "variance_bottom_0_7_cooperation_rate.json")

    agents_data = {}
    for agent_folder in os.listdir(agents_path):
        if os.path.isdir(os.path.join(agents_path, agent_folder)):
            json_path = os.path.join(agents_path, agent_folder, "averages", "bottom", "cooperation_rate_stats.json")
            data = read_json_file(json_path)
            if data:
                agent_id = list(data.keys())[0]
                agents_data[agent_folder] = data[agent_id]

    human_avg = read_json_file(human_avg_path)
    human_var = read_json_file(human_var_path)

    if human_avg and human_var:
        human_data = {'average': human_avg['average'], 'variance': human_var['variance']}
    else:
        print("Error loading human data.")
        return

    if agents_data and human_data:
        save_path = os.path.join(base_path, 'combined_agents_and_human_plot.png')
        plot_combined_results(agents_data, human_data, "Combined Agents and Human Cooperation Rate", "Cooperation Rate",
                              save_path)
    else:
        print("No data loaded for agents or human.")


if __name__ == "__main__":
    main()