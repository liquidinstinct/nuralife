from sklearn.metrics import accuracy_score

class Evaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def evaluate(self, test_data):
        X_test = test_data.drop(self.config.target_column, axis=1)
        y_test = test_data[self.config.target_column]
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy}")