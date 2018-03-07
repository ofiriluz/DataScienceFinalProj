from Step2.Classifiers.AbstractHotelsClassifier import AbstractHotelsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle


class GaussianNBHotelsClassifier(AbstractHotelsClassifier):
    def __init__(self, **kwargs):
        super(AbstractHotelsClassifier).__init__()
        self.bayes_model = None
        self.bayes_encoders = {}
        self.reset()

    def reset(self):
        self.bayes_model = GaussianNB()

    def save_model(self, folder_path):
        with open(folder_path + '/model.pkl', 'wb') as file:
            pickle.dump(self.bayes_model, file)
        with open(folder_path + '/model_labelizer.pkl', 'wb') as file:
            pickle.dump(self.bayes_encoders, file)

    def predict(self, x):
        return self.bayes_model.predict(x)

    def predict_proba(self, x):
        return self.bayes_model.predict_proba(x)

    def load_model(self, folder_path):
        with open(folder_path + '/model.pkl', 'rb') as file:
            self.bayes_model = pickle.load(file)
        with open(folder_path + '/model_labelizer.pkl', 'rb') as file:
            self.bayes_encoders = pickle.load(file)

    def fit_labelizer(self, x):
        for col in x.columns:
            if x.dtypes[col] == 'object':
                self.bayes_encoders[col] = LabelEncoder()
                self.bayes_encoders[col].fit(x[col])

    def transform_labels(self, x):
        for col in x.columns:
            if x.dtypes[col] == 'object':
                if col in self.bayes_encoders.keys():
                    x.loc[:, col] = self.bayes_encoders[col].transform(x[col])
        return x

    def train(self, x, y):
        self.bayes_model.fit(x, y)

    def get_name(self):
        return 'NaiveBayes'

    def get_labelizer(self):
        return self.bayes_encoders

    def get_native_classifier(self):
        return self.bayes_model
