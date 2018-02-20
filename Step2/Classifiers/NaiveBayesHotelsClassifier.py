from Step2.Classifiers.AbstractHotelsClassifier import AbstractHotelsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle


class GaussianNBHotelsClassifier(AbstractHotelsClassifier):
    def __init__(self):
        super(AbstractHotelsClassifier).__init__()
        self.bayes_model = None
        self.reset()

    def reset(self):
        self.bayes_model = GaussianNB()

    def save_model(self, path):
        with open(path, 'w') as file:
            pickle.dump(self.bayes_model, file)

    def predict(self, x):
        return self.bayes_model.predict(x)

    def predict_proba(self, x):
        return self.bayes_model.predict_proba(x)

    def load_model(self, path):
        with open(path, 'r') as file:
            self.bayes_model = pickle.load(file)

    def train(self, x, y):
        self.bayes_model.fit(x, y)
