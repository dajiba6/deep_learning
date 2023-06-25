import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


x = np.array([[3, 3], [4, 3], [1, 1]])
y = [1, -1, 1]

w = [0, 0]
b = 0
yita = 1


def misclassificationCheck(x, y, w, b):
    misFlag = False
    count = 0
    misFlag_index = 0
    for i in range(0, len(y)):
        if y[i] * (np.dot(w, x[i]) + b) <= 0:
            count += 1
            misFlag_index = i
    if count > 0:
        misFlag = True
    return misFlag, misFlag_index


def update(x, y, w, b, i):
    w = w + yita * y[i] * x[i]
    b = b + yita * y[i]
    return w, b


def optimization(x, y, w, b):
    misFlag, misFlag_index = misclassificationCheck(x, y, w, b)
    while misFlag:
        print("误分类的点：", misFlag_index)
        w, b = update(x, y, w, b, misFlag_index)
        print("采用误分类点 {} 更新后的权重为:w是 {} , b是 {} ".format(misFlag_index, w, b))
        misFlag, misFlag_index = misclassificationCheck(x, y, w, b)
    return w, b


w, b = optimization(x, y, w, b)
print(w, b)

# x_axis = np.linspace(-5, 5, 100)
# fig, ax = plt.subplots()
# ax.scatter(x[2:, 0], x[2:, 1], color="green", label="1")
# ax.scatter(x[:2, 0], x[:2, 1], color="red", label="-1")

# p_y = w[0] / w[1] * x_axis + b / w[1]
# ax.plot(x_axis, p_y, color="blue", label="rap")

# ax.legend()
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# plt.show()
