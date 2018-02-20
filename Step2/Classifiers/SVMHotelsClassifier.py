from Step2.Classifiers.AbstractHotelsClassifier import AbstractHotelsClassifier
from sklearn import svm
import pickle


class SVMHotelsClassifier(AbstractHotelsClassifier):
    def __init__(self):
        super(AbstractHotelsClassifier).__init__()
        self.svm_model = None
        self.reset()

    def reset(self):
        self.svm_model = svm.LinearSVC()

    def save_model(self, path):
        with open(path, 'w') as file:
            pickle.dump(self.svm_model, file)

    def predict(self, x):
        return self.svm_model.predict(x)

    def predict_proba(self, x):
        return self.svm_model.predict_proba(x)

    def load_model(self, path):
        with open(path, 'r') as file:
            self.svm_model = pickle.load(file)

    def train(self, x, y):
        self.svm_model.fit(x, y)
