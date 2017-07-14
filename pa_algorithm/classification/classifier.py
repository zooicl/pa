import abc
import time


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
        print('Build', time.time() - t)

        return

    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        y_prob = self.clf.predict_proba(X_test)[:, 1]

        return y_pred, y_prob

    @abc.abstractmethod
    def __build__(self, params):
        pass
