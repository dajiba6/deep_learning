# from sklearn.datasets import load_iris
# from sklearn import tree
# import sys
# import os

# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)

# with open("iris.dot", "w") as f:
#     f = tree.export_graphviz(clf, out_file=f)


# import pydotplus

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris.pdf")

from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
x = iris.data[:, [0, 2]]
y = iris.target

clf = DecisionTreeClassifier(max_depth=4)

clf.fit(x, y)

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1  # 特征1最大值最小值
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1  # 特征2最大值最小值
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z.shape)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.8)
plt.show()

import pydotplus
from sklearn import tree

dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
