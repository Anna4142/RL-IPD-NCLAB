class BaseMetric:
    def __init__(self, data_buffer):
        self.data_buffer = data_buffer

    def update(self):
        # Default implementation attempts to pull data from the data_buffer.
        # Derived classes should implement how they use this data.
        raise NotImplementedError("The update method must be implemented by the subclass and utilize the data_buffer.")

    def get_metrics(self):
        raise NotImplementedError("The get_metrics method must be implemented by the subclass.")

    def reset(self):
        raise NotImplementedError("The reset method must be implemented by the subclass.")
