import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from pa_algorithm.classification.classifier_factory import ClassifierFactory
from pa_algorithm.helper.reporter import Reporter

example = load_breast_cancer()

X = example.data
y = example.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

dt = {
    'name': 'DT',
    'params': {},
    'desc': 'breast_cancer'
}

xgb = {
    'name': 'XGB',
    'params': {},
    'desc': 'breast_cancer'
}

svm = {
    'name': 'SVM',
    'params': {},
    'desc': 'breast_cancer'
}

rf = {
    'name': 'RF',
    'params': {},
    'desc': 'breast_cancer'
}

for algo in [dt, xgb, svm, rf]:
    # for algo in [rf]:
    name = algo['name']
    params = algo['params']
    desc = algo['desc']

    model = ClassifierFactory.create_model(name)
    model.fit(X_train, y_train, params)

    y_pred, y_prob = model.predict(X_test)

    reporter = Reporter(name, params, model.get_feature_importances(example.feature_names), desc)
    reporter.add(y_true=y_test, y_pred=y_pred, y_prob=y_prob, desc=desc, y_id=None)
    reporter.write()
