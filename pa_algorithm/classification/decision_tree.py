from sklearn.tree import DecisionTreeClassifier

from .classifier import Classifier


class DecisionTree(Classifier):
    def __fit__(self, params):
        return DecisionTreeClassifier(**params)
