from ..helper.reporter import Reporter

from .classifier_factory import ClassifierFactory


class Modeling:
    def __init__(self, algorithm, X_train, y_train, params, desc):
        self.reporter = Reporter(algorithm, params, desc)

        self.model = ClassifierFactory.create_model(algorithm)
        self.model.build(X_train, y_train, params)

        return

    def evaluate(self, X_test, y_test, desc, y_id=None):
        y_pred, y_prob = self.model.predict(X_test)
        self.reporter.add(y_true=y_test, y_pred=y_pred, y_prob=y_prob, desc=desc, y_id=y_id)

        return
