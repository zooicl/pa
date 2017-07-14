from .decision_tree import DecisionTree
from .xgb import XGB


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
