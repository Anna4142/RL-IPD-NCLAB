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
                    "name": "TitForTat"
                },
                "agent2": {
                    "type": "Deep",
                    "name": "ActorCriticAgent",
                    "parameters": {
                        "use_spiking_nn": False,
                        "hidden_layers": [256, 256],
                        "learning_rate": 0.001,
                        "gamma": 0.99
                    }
                }
            },
            "experiment": {
                "save_directory": "weights",
                "use_predefined_weights_id": False
            }
        }

    def get_environment_name(self):
        return self.config.get('environment', {}).get('name', 'default_environment')

    def get_agent_details(self, agent_key):
        # Returns the agent type, name, and parameters as a tuple
        agent_info = self.config.get('agents', {}).get(agent_key, {})
        return (agent_info.get('type', 'Unknown'), agent_info.get('name', 'Unknown'), agent_info.get('parameters', {}))

    def get_save_directory(self):
        return self.config.get('experiment', {}).get('save_directory', 'weights')

    def use_predefined_weights(self):
        return self.config.get('experiment', {}).get('use_predefined_weights_id', False)
