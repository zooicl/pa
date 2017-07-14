from xgboost import XGBClassifier

from .classifier import Classifier


class XGB(Classifier):
    def __build__(self, params):
        return XGBClassifier(**params)
