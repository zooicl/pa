from sklearn.tree import DecisionTreeClassifier

from classification.classifier import Classifier


class DecisionTree(Classifier):
    def __build__(self, params):
        return DecisionTreeClassifier(**params)
