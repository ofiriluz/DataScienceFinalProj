from Step2.Classifiers.AbstractHotelsClassifier import AbstractHotelsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import pickle


class OneVsRestHotelsClassifier(AbstractHotelsClassifier):
    def __init__(self, classifer, **kwargs):
        super(AbstractHotelsClassifier).__init__()
        self.one_vs_rest_classifier = None
        self.wrapped_classifier = classifer
        self.one_vs_rest_encoders = {}
        self.reset()

    def reset(self):
        self.one_vs_rest_classifier = OneVsRestClassifier(self.wrapped_classifier.get_native_classifier())

    def save_model(self, folder_path):
        with open(folder_path + '/model.pkl', 'wb') as file:
            pickle.dump(self.one_vs_rest_classifier, file)
        with open(folder_path + '/model_labelizer.pkl', 'wb') as file:
            pickle.dump(self.one_vs_rest_encoders, file)

    def predict(self, x):
        return self.one_vs_rest_classifier.predict(x)

    def predict_proba(self, x):
        return self.one_vs_rest_classifier.predict_proba(x)

    def load_model(self, folder_path):
        with open(folder_path + '/model.pkl', 'rb') as file:
            self.one_vs_rest_classifier = pickle.load(file)
        with open(folder_path + '/model_labelizer.pkl', 'rb') as file:
            self.one_vs_rest_encoders = pickle.load(file)

    def fit_labelizer(self, x):
        for col in x.columns:
            if x.dtypes[col] == 'object':
                self.one_vs_rest_encoders[col] = LabelEncoder()
                self.one_vs_rest_encoders[col].fit(x[col])

    def transform_labels(self, x):
        for col in x.columns:
            if x.dtypes[col] == 'object':
                if col in self.one_vs_rest_encoders.keys():
                    x.loc[:, col] = self.one_vs_rest_encoders[col].transform(x[col])
        return x

    def train(self, x, y):
        self.one_vs_rest_classifier.fit(x, y)

    def get_name(self):
        return 'OneVsRest'

    def get_labelizer(self):
        return self.one_vs_rest_encoders

    def get_native_classifier(self):
        return self.one_vs_rest_classifier

