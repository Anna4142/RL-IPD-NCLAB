class RunConfig:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        return {
            "environment": {
                "name": "prisoners_dilemma"
            },
            "agents": {
                "agent1": {
                    "type": "Fixed",
                    "name": "TitForTat",
                    "parameters": {}
                },

                "agent2": {
                    "type": "Deep",
                    "name": "TOMAC",
                    "parameters": {
                        "use_spiking_nn": False,
                        "hidden_layers": [256, 256],
                        "learning_rate": 0.001,
                        "gamma": 0.99,
                        "use_mouse_hist": False  # Default to true or based on specific setups
                    }
                }
            },
            "experiment": {
                "save_directory": "weights",
                "use_predefined_weights": False,
                "use_forced_actions": False # Added parameter
            }
        }

    def get_environment_name(self):
        return self.config['environment'].get('name', 'default_environment')

    def get_agent_details(self, agent_key):
        agent_info = self.config['agents'].get(agent_key, {})
        return (agent_info.get('type', 'Unknown'), agent_info.get('name', 'Unknown'), agent_info.get('parameters', {}))

    def get_save_directory(self):
        return self.config['experiment'].get('save_directory', 'weights')

    def use_predefined_weights(self):
        return self.config['experiment'].get('use_predefined_weights', False)

    def use_forced_actions(self):
        return self.config['experiment'].get('use_forced_actions', False)
