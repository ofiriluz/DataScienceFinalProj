from abc import abstractmethod


class AbstractHotelsClassifier:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass

    @abstractmethod
    def fit_labelizer(self, x):
        pass

    @abstractmethod
    def transform_labels(self, x):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_labelizer(self):
        pass

    @abstractmethod
    def get_native_classifier(self):
        pass
