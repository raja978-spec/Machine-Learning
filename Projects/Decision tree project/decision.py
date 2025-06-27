from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()

features = iris.data
labels = iris.target

model = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    min_samples_leaf=4
)
model.fit(features, labels)


fig, ax = plt.subplots(figsize=(10,10))

dt = tree.plot_tree(model, ax=ax, feature_names=['sepal_length',
                                          'sepal_width',
                                          'petal_length',
                                          'petal_width'
                                          ])
plt.show()