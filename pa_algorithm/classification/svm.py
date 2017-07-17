from sklearn import svm

from .classifier import Classifier


class SVC(Classifier):
    def __build__(self, params):
        return svm.SVC(**params)
