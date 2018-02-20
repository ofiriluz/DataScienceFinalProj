from Step2.Classifiers.AbstractHotelsClassifier import AbstractHotelsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

DEFAULT_RANDOM_STATE=42


class DecisionTreeHotelsClassifier(AbstractHotelsClassifier):
    def __init__(self, min_samples_split=40):
        super(AbstractHotelsClassifier).__init__()
        self.tree_model = None
        self.min_samples_split = min_samples_split
        self.reset()

    def reset(self):
        self.tree_model = DecisionTreeClassifier(min_samples_split=self.min_samples_split,
                                                 random_state=DEFAULT_RANDOM_STATE)

    def save_model(self, path):
        with open(path, 'w') as file:
            pickle.dump(self.tree_model, file)

    def predict(self, x):
        return self.tree_model.predict(x)

    def predict_proba(self, x):
        return self.tree_model.predict_proba(x)

    def load_model(self, path):
        with open(path, 'r') as file:
            self.tree_model = pickle.load(file)

    def train(self, x, y):
        self.tree_model.fit(x, y)
