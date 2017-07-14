from sklearn.datasets import load_breast_cancer
from pa_algorithm.classification.binomial import Modeling

example = load_breast_cancer()

model = Modeling('XGB', example.data, example.target, {}, 'example_breast_cancer')
model.evaluate(example.data, example.target, 'predict')




