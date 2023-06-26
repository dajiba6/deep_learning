import numpy as np

X = np.array([[1, 1], [2, 1], [3, 2], [11, 1], [12, 1], [3, 12]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB, MultinomialNB

clf = GaussianNB()
# 拟合数据
clf.fit(X, Y)
print("==Predict result by predict==")
print(clf.predict([[100, 11]]))
print("==Predict result by predict_proba==")
print(clf.predict_proba([[100, 11]]))
print("==Predict result by predict_log_proba==")
print(clf.predict_log_proba([[100, 11]]))
print("==classes==")
print(clf.classes_)
print("=====================================")
mnb = MultinomialNB()
mnb.fit(X, Y)
print("==Predict result by predict==")
print(mnb.predict([[100, 11]]))
print("==Predict result by predict_proba==")
print(mnb.predict_proba([[100, 11]]))
print("==Predict result by predict_log_proba==")
print(mnb.predict_log_proba([[100, 11]]))
print("==classes==")
print(mnb.classes_)
