from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib


class ClassifierFactory(object):
    @classmethod
    def create_model(cls, algorithm=None):
        if algorithm is None:
            raise ValueError('algo should be defined, not allowed None.')

        algorithm = str(algorithm).strip().upper()

        if algorithm == 'DT':
            model = DecisionTreeClassifier()
        elif algorithm == 'XGB':
            model = XGBClassifier()
        elif algorithm == 'SVM':
            model = SVC()
            pass
        elif algorithm == 'RF':
            model = RandomForestClassifier()
            pass
        else:
            raise ValueError('algorithm name ({}) is not defined.'.format(algorithm))

        return model

    @classmethod
    def dump_model(cls, clf, filename):
        joblib.dump(clf, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
