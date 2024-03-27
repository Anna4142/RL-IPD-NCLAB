class DataBuffer:
    def __init__(self):
        self.latest_action1 = None
        self.latest_action2 = None
        self.latest_reward1 = None
        self.latest_reward2 = None
        self.current_state = None
        self.next_state = None

    def update_action1(self, action1):
        self.latest_action1 = action1

    def update_action2(self, action2):
        self.latest_action2 = action2

    def update_reward1(self, reward1):
        self.latest_reward1 = reward1

    def update_reward2(self, reward2):
        self.latest_reward2 = reward2

    def update_state(self, current_state, next_state):
        self.current_state = current_state
        self.next_state = next_state

    # Additional getters as needed

