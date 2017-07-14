from sklearn import datasets
from classification.binomial import Modeling

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target[:, 1]


params = {

}
model = Modeling('DT', X, Y, params, 'iris')


