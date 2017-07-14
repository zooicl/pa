import abc
import time

from classification.decision_tree import DecisionTree
from classification.xgb import XGB


class Classifier(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.clf = None
        return

    def build(self, X_train, y_train, params):
        t = time.time()

        self.clf = self.__build__(params)

        if self.clf is None:
            raise ValueError('clf is None')

        self.clf = self.clf.fit(X_train, y_train)
        print(self.algo, 'build', time.time() - t)

        return

    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        y_prob = self.clf.predict_prob(X_test)[:, 1]

        return y_pred, y_prob

    @abc.abstractmethod
    def __build__(self, params):
        pass


class ClassifierFactory(object):
    @classmethod
    def create_model(cls, algorithm=None):
        if algorithm is None:
            raise ValueError('algo should be defined, not allowed None.')

        algorithm = str(algorithm).strip().upper()

        if algorithm == 'DT':
            model = DecisionTree()
        elif algorithm == 'XGB':
            model = XGB()
        elif algorithm == 'SVM':
            # model = SVC()
            pass
        elif algorithm == 'RF':
            # model = RandomForestClassifier()
            pass
        else:
            raise ValueError('algorithm name ({}) is not defined.'.format(algorithm))

        return model
