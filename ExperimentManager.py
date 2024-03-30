import os


class ExperimentManager:
    def __init__(self, base_directory="Episodes"):
        self.base_directory = base_directory

    def get_next_experiment_number(self, experiment_id):
        experiment_base_dir = os.path.join(self.base_directory, experiment_id)

        if not os.path.exists(experiment_base_dir):
            os.makedirs(experiment_base_dir, exist_ok=True)
            return 1

        existing_dirs = [
            d for d in os.listdir(experiment_base_dir)
            if os.path.isdir(os.path.join(experiment_base_dir, d))
        ]

        existing_numbers = []
        for dir_name in existing_dirs:
            parts = dir_name.split("_")
            if parts[0] == "experiment" and parts[-1].isdigit():
                existing_numbers.append(int(parts[-1]))

        return max(existing_numbers) + 1 if existing_numbers else 1
