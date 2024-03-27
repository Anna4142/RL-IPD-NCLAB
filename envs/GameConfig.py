class GameConfig:
    def __init__(self, T, S, P, R):
        self.games = {
            "prisoners_dilemma": {
                "rounds": 10,
                "payout_matrix": [
                    [(R, R), (S, T)],
                    [(T, S), (P, P)]
                ],
                "algorithm_type": "MULTI AGENT"  # Example default value
            },
            "stag_hunt": {
                "rounds": 10,
                "payout_matrix": [
                    [(R, R), (S, P)],
                    [(P, S), (T, T)]
                ],
                "algorithm_type": "SINGLE AGENT"  # Example default value
            },
            "chicken_game": {
                "rounds": 10,
                "payout_matrix": [
                    [(T, S), (P, P)],
                    [(S, T), (R, R)]
                ],
                "algorithm_type": "SINGLE AGENT"  # Example default value
            }
            # Add more games as needed
        }

    def add_game(self, game_name, rounds, payout_matrix, algorithm_type):
        if game_name not in self.games:
            self.games[game_name] = {
                "rounds": rounds,
                "payout_matrix": payout_matrix,
                "algorithm_type": algorithm_type
            }
        else:
            print(f"Game '{game_name}' already exists.")

    def get_game_config(self, game_name):
        if game_name in self.games:
            return self.games[game_name]
        else:
            print(f"Game '{game_name}' not found.")
            return None
