class RunConfig:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        # Configuration dictionary directly included in the class
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

    def get_environment_config(self):
        return self.config.get('environment', {})

    def get_agent_config(self, agent_key):
        return self.config.get('agents', {}).get(agent_key, {})

    def get_experiment_config(self):
        return self.config.get('experiment', {})
