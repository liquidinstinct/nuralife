from sklearn.linear_model import LogisticRegression

class Model:
    def __init__(self, config):
        self.config = config
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)