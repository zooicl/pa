from xgboost import XGBClassifier

from classification.classifier import Classifier


class XGB(Classifier):
    def __build__(self, params):
        return XGBClassifier(**params)
