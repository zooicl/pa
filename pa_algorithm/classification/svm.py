from sklearn import svm

from .classifier import Classifier


class SVC(Classifier):
    def __fit__(self, params):
        return svm.SVC(**params)
