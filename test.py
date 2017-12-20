__author__ = 'davidwer'
__author__ = 'davidwer'
# David Wertenteil

"""

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import statistics
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict



# The digits dataset
digits = datasets.load_digits()
# *****************************************************************************************************************
# --------------------------------------- 20 ----------------------------------------------------------------------


# *****************************************************************************************************************
# --------------------------------------- 21 ----------------------------------------------------------------------


def center_values(img):
    """

    :param img:
    :return:
    """
    return img[19] + img[27] + img[35] + img[43] + img[20] + img[28] + img[36] + img[44]
# ------------------------------------------------------------------------------------------------------


def num_of_zeros(img):
    return img.count(0.0)
# ------------------------------------------------------------------------------------------------------


def modulus(img):
    a = []
    for i in img:
        a.append(-(int(i) % 16))
    return sum(a)
# ------------------------------------------------------------------------------------------------------


def circle_finder(img):
    c = 0
    flag = False
    for i in img[32:]:
        if i > 7 and not flag:
            flag = True
            c += 1
        elif i <= 7 and flag:
            flag = False
            c += 1
    return int(c <= 8)
# ------------------------------------------------------------------------------------------------------


def var(img):
    return np.var(img)
# ------------------------------------------------------------------------------------------------------


class Classifier:
    def __init__(self, classi):
        self.classi = classi
        self.classification = 0
        self.array = []

    def fit(self, data, target):
        zeros = []
        ones = []
        for i, j in zip(data, target):
            a = np.ndarray.tolist(i)
            if j == 0:
                zeros.append(self.classi(a))
            else:
                ones.append(self.classi(a))
        self.classification = (statistics.mean(ones) + statistics.mean(zeros)) / 2

    def predict(self, data):
        ima = []
        for i in data:
            a = np.ndarray.tolist(i)
            ima.append(self.classi(a))
        return ima
# ------------------------------------------------------------------------------------------------------


indices_0_1 = np.where(np.logical_and(digits.target >= 0, digits.target <= 1))
n_samples = len(digits.images[indices_0_1])
data = digits.images[indices_0_1].reshape((360, -1))

classifier_a = Classifier(circle_finder)
classifier_b = Classifier(modulus)
classifier_c = Classifier(center_values)
classifier_d = Classifier(num_of_zeros)
classifier_e = Classifier(var)

predicted_a = classifier_a.predict(data)
predicted_b = classifier_b.predict(data)
predicted_c = classifier_c.predict(data)
predicted_d = classifier_d.predict(data)
predicted_e = classifier_e.predict(data)


fig = plt.figure()
ax = Axes3D(fig)
fig.suptitle("num_of_zeros, circle_finder, center_values")
ax.set_xlabel('num_of_zeros')
ax.set_ylabel('circle_finder')
ax.set_zlabel('center_values')
ax.scatter(predicted_a, predicted_b, predicted_d, c=digits.target[indices_0_1],cmap=plt.cm.Set1,  edgecolor='k', s=30)
plt.show()


# creating the X (feature)
X = np.column_stack((predicted_a, predicted_b, predicted_c, predicted_d, predicted_e))
# scaling the values for better classification performance
X_scaled = preprocessing.scale(X)
# the predicted outputs
Y = digits.target[indices_0_1]  # Training Logistic regression
logistic_classifier = linear_model.LogisticRegression()
logistic_classifier.fit(X_scaled, Y)
# show how good is the classifier on the training data
expected = Y
predicted = logistic_classifier.predict(X_scaled)

print("Logistic regression using [featureA, featureB] features:\n%s\n" % ( metrics.classification_report( expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
# estimate the generalization performance using cross validation
predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
print("Logistic regression using [featureA, featureB] features cross validation:\n%s\n" % (metrics.classification_report(expected, predicted2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))
