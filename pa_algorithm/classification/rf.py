from sklearn import ensemble
from .classifier import Classifier


class RandomForestClassifier(Classifier):
    def __build__(self, params):
        return ensemble.RandomForestClassifier(**params)
