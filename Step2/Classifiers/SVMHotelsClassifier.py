from Step2.Classifiers.AbstractHotelsClassifier import AbstractHotelsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import pickle


class SVMHotelsClassifier(AbstractHotelsClassifier):
    def __init__(self, **kwargs):
        super(AbstractHotelsClassifier).__init__()
        self.svm_model = None
        self.svm_encoders = {}
        self.reset()

    def reset(self):
        self.svm_model = svm.LinearSVC()

    def save_model(self, folder_path):
        with open(folder_path + '/model.pkl', 'wb') as file:
            pickle.dump(self.svm_model, file)
        with open(folder_path + '/model_labelizer.pkl', 'wb') as file:
            pickle.dump(self.svm_encoders, file)

    def predict(self, x):
        return self.svm_model.predict(x)

    def predict_proba(self, x):
        return self.svm_model.predict_proba(x)

    def load_model(self, folder_path):
        with open(folder_path + '/model.pkl', 'rb') as file:
            self.svm_model = pickle.load(file)
        with open(folder_path + '/model_labelizer.pkl', 'rb') as file:
            self.svm_encoders = pickle.load(file)

    def fit_labelizer(self, x):
        for col in x.columns:
            if x.dtypes[col] == 'object':
                self.svm_encoders[col] = LabelEncoder()
                self.svm_encoders[col].fit(x[col])

    def transform_labels(self, x):
        for col in x.columns:
            if x.dtypes[col] == 'object':
                if col in self.svm_encoders.keys():
                    x.loc[:, col] = self.svm_encoders[col].transform(x[col])
        return x

    def train(self, x, y):
        self.svm_model.fit(x, y)

    def get_name(self):
        return 'SVM'

    def get_labelizer(self):
        return self.svm_encoders

    def get_native_classifier(self):
        return self.svm_model


