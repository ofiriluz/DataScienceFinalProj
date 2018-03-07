from Step2.Classifiers.AbstractHotelsClassifier import AbstractHotelsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

DEFAULT_RANDOM_STATE = 42


class DecisionTreeHotelsClassifier(AbstractHotelsClassifier):
    def __init__(self, min_samples_split=40, **kwargs):
        print(min_samples_split)
        super(AbstractHotelsClassifier).__init__()
        self.tree_model = None
        self.tree_encoders = {}
        self.min_samples_split = min_samples_split
        self.reset()

    def reset(self):
        self.tree_model = DecisionTreeClassifier(min_samples_split=self.min_samples_split,
                                                 random_state=DEFAULT_RANDOM_STATE)

    def save_model(self, folder_path):
        with open(folder_path + '/model.pkl', 'wb') as file:
            pickle.dump(self.tree_model, file)
        with open(folder_path + '/model_labelizer.pkl', 'wb') as file:
            pickle.dump(self.tree_encoders, file)

    def predict(self, x):
        return self.tree_model.predict(x)

    def fit_labelizer(self, x):
        for col in x.columns:
            if x.dtypes[col] == 'object':
                self.tree_encoders[col] = LabelEncoder()
                self.tree_encoders[col].fit(x[col])

    def transform_labels(self, x):
        for col in x.columns:
            if x.dtypes[col] == 'object':
                if col in self.tree_encoders.keys():
                    x.loc[:, col] = self.tree_encoders[col].transform(x[col])
        return x

    def predict_proba(self, x):
        return self.tree_model.predict_proba(x)

    def load_model(self, folder_path):
        with open(folder_path + '/model.pkl', 'rb') as file:
            self.tree_model = pickle.load(file)
        with open(folder_path + '/model_labelizer.pkl', 'rb') as file:
            self.tree_encoders = pickle.load(file)

    def train(self, x, y):
        self.tree_model.fit(x, y)

    def get_name(self):
        return "DecisionTree"

    def get_labelizer(self):
        return self.tree_encoders

    def get_native_classifier(self):
        return self.tree_model
