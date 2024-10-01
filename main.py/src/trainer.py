class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def train(self, train_data):
        X_train = train_data.drop(self.config.target_column, axis=1)
        y_train = train_data[self.config.target_column]
        self.model.train(X_train, y_train)