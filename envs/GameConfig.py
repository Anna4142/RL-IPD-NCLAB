class GameConfig:
    def __init__(self, T, S, P, R):
        self.games = {
            "prisoners_dilemma": {
                "rounds": 1000,
                "payout_matrix": [
                    [(R, R), (S, T)],
                    [(T, S), (P, P)]
                ],
                "algorithm_type": "SINGLE AGENT",
                "memory": 5,  # Number of past rounds to remember
                "history_length": 5,  # Number of past actions to keep in history
                "obs_type": "both"  # Type of observation data to include-can be self other or both
            },
            "stag_hunt": {
                "rounds": 10,
                "payout_matrix": [
                    [(R, R), (S, P)],
                    [(P, S), (T, T)]
                ],
                "algorithm_type": "SINGLE AGENT",
                "memory": 3,
                "history_length": 3,
                "obs_type": "self"
            },
            "chicken_game": {
                "rounds": 10,
                "payout_matrix": [
                    [(T, S), (P, P)],
                    [(S, T), (R, R)]
                ],
                "algorithm_type": "SINGLE AGENT",
                "memory": 2,
                "history_length": 2,
                "obs_type": "other"
            }
            # Add more games as needed
        }

    def add_game(self, game_name, rounds, payout_matrix, algorithm_type, memory, history_length, obs_type):
        if game_name not in self.games:
            self.games[game_name] = {
                "rounds": rounds,
                "payout_matrix": payout_matrix,
                "algorithm_type": algorithm_type,
                "memory": memory,
                "history_length": history_length,
                "obs_type": obs_type
            }
        else:
            print(f"Game '{game_name}' already exists.")

    def get_game_config(self, game_name):
        if game_name in self.games:
            return self.games[game_name]
        else:
            print(f"Game '{game_name}' not found.")
            return None
