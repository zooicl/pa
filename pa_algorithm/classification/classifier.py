import abc
import time
import operator
from sklearn import model_selection


class Classifier(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.__clf = None
        return

    def fit(self, X_train, y_train, params):
        t = time.time()

        self.__clf = self.__fit__(params)

        if self.__clf is None:
            raise ValueError('clf is None')

        self.__clf = self.__clf.fit(X_train, y_train)
        print('fit', time.time() - t)

        return

    def cross_val_score(self, X_train, y_train, params):
        return model_selection.cross_val_score(self.__clf, X_train, y_train, **params)

    def predict(self, X_test):
        y_pred = self.__clf.predict(X_test)

        y_prob = None
        if hasattr(self.__clf, 'predict_proba'):
            y_prob = self.__clf.predict_proba(X_test)[:, 1]

        return y_pred, y_prob

    def get_model(self):
        return self.__clf

    def get_feature_importances(self, feature_names):
        if hasattr(self.__clf, 'feature_importances_'):
            feature_map = {}
            for i, val in enumerate(self.__clf.feature_importances_):
                feature_map[feature_names[i]] = val
            return sorted(feature_map.items(), key=operator.itemgetter(1), reverse=True)
        else:
            return 'No provided!'

    @abc.abstractmethod
    def __fit__(self, params):
        pass
