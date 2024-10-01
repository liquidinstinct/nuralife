from data_loader import DataLoader
from model import Model
from trainer import Trainer
from evaluator import Evaluator
from config import Config

class NuraLive:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader(self.config)
        self.model = Model(self.config)
        self.trainer = Trainer(self.config, self.model)
        self.evaluator = Evaluator(self.config, self.model)

    def run(self):
        # Load and preprocess data
        train_data, test_data = self.data_loader.load_data()

        # Train the model
        self.trainer.train(train_data)

        # Evaluate the model
        self.evaluator.evaluate(test_data)