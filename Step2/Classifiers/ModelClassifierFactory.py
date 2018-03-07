from Step2.Classifiers.DecisionTreeHotelsClassifier import DecisionTreeHotelsClassifier
from Step2.Classifiers.SVMHotelsClassifier import SVMHotelsClassifier
from Step2.Classifiers.NaiveBayesHotelsClassifier import GaussianNBHotelsClassifier
from Step2.Classifiers.OneVsRestHotelsClassifier import OneVsRestHotelsClassifier
import json
import os


class ModelClassifierFactory:
    class __impl:
        def __init__(self):
            self.classifiers_map = {'NaiveBayes': GaussianNBHotelsClassifier,
                                    'SVM': SVMHotelsClassifier,
                                    'DecisionTree': DecisionTreeHotelsClassifier,
                                    'OneVsRest': OneVsRestHotelsClassifier}

        def create_classifier(self, name):
            if name in self.classifiers_map.keys():
                return self.classifiers_map[name]
            return None

        def load_classifier(self, folder_path):
            # Load metadata json file
            if os.path.exists(folder_path + '/metadata.json'):
                with open(folder_path + '/metadata.json', 'r') as json_file:
                    metadata_json = json.load(json_file)
                    classifier_name = metadata_json['ClassifierName']
                    if classifier_name in self.classifiers_map.keys():
                        classifier = self.classifiers_map[classifier_name]()
                        classifier.load_model(metadata_json['ModelFolderPath'])
                        return classifier
                return None

        def save_classifier(self, classifier, folder_path):
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            metadata_json = {'ClassifierName': classifier.get_name(),
                             'ModelFolderPath': folder_path}
            classifier.save_model(folder_path)
            with open(folder_path + '/metadata.json', 'w') as json_file:
                json.dump(metadata_json, json_file)

    # storage for the instance reference
    __instance = None

    def __init__(self):
        if ModelClassifierFactory.__instance is None:
            ModelClassifierFactory.__instance = ModelClassifierFactory.__impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_ModelClassifierFactory__instance'] = ModelClassifierFactory.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)