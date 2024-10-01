import pandas as pd

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        # Load and preprocess data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        return train_data, test_data