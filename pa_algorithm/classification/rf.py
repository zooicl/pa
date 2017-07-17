from sklearn import ensemble
from .classifier import Classifier


class RandomForestClassifier(Classifier):
    def __fit__(self, params):
        return ensemble.RandomForestClassifier(**params)
