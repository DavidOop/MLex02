__author__ = 'davidwer'
__author2__ = 'omersc'
# David Wertenteil
# Omer Schwartz

"""

"""

# Standard scientific Python imports
import numpy as np
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict

# The digits data set
digits = datasets.load_digits()


def var(img):
    return np.var(img) * sum(img)
def tf(d):
    return np.var([sum(i) for i in d[:4]]) * sum(d[:4])
def bf(d):
    return np.var([sum(i) for i in d[4:]]) * sum(d[4:])

def lf(d):
    return np.var([sum(i) for i in d[:, :4]]) * sum(d[:, :4])

def rf(d):
    return np.var([sum(i) for i in d[:, 4:]]) * sum(d[:, 4:])


def trf(d):
    return np.var([sum(i) for i in d[:4, 4:]]) * sum(d[:4, 4:]) + \
        np.var([sum(i) for i in d[:4, :4]]) * sum(d[:4, :4]) + \
        np.var([sum(i) for i in d[4:, :4]]) * sum(d[4:, :4]) + \
        np.var([sum(i) for i in d[4:, 4:]]) * sum(d[4:, 4:])


def cf(d):
    return np.var([sum(i) for i in d[2:-2, 2:-2]]) * sum(d[2:-2, 2:-2])


def srvf(d):
    return np.var([sum(i) for i in d]) * sum(d)


n_samples = len(digits.images)
data = digits.images

variance = []
top_mat = []
bottom_mat = []
left_mat = []
right_mat = []
center_mat = []
quarters_mat = []
var_of_sums = []
for i in data:
    variance.append(var(i))
    top_mat.append(tf(i))
    bottom_mat.append(bf(i))
    left_mat.append(lf(i))
    right_mat.append(rf(i))
    center_mat.append(cf(i))
    quarters_mat.append(trf(i))
    var_of_sums.append(srvf(i))

x = np.column_stack((variance, top_mat, bottom_mat, right_mat, left_mat,
                     center_mat, quarters_mat, var_of_sums))
# scaling the values for better classification performance
x_scaled = preprocessing.scale(x)

# the predicted outputs
y = digits.target  # Training Logistic regression
logistic_classifier = linear_model.LogisticRegression()
logistic_classifier.fit(x_scaled, y)
# show how good is the classifier on the training data
expected = y
predicted = logistic_classifier.predict(x_scaled)

print("Logistic regression using "
      "features:\n%s\n" %
      (metrics.classification_report(expected, predicted)))
print(
    "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
# estimate the generalization performance using cross validation
predicted2 = cross_val_predict(logistic_classifier, x_scaled, y, cv=10)
