__author__ = 'davidwer'
__author__ = 'omersc'
# David Wertenteil
# Omer Schwartz

"""

"""

# Standard scientific Python imports
import matplotlib.pyplot as plt
import statistics
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict

# The digits dataset
digits = datasets.load_digits()


def var(img):
    return np.var(img)
def tf(d):
    return np.var(d[:4])
def bf(d):
    return np.var(d[4:])
def lf(d):
    return np.var(d[:, :4])
def rf(d):
    return np.var(d[:, 4:])
def trf(d):
    return np.var(d[:4, 4:])
def tlf(d):
    return np.var(d[:4, :4])
def brf(d):
    return np.var(d[4:, 4:])
def blf(d):
    return np.var(d[4:, :4])
def srvf(d):
    return np.var([np.mean(i) for i in d])
def maxf(d):
    return max(np.var(d[:4, 4:]), np.var(d[:4, :4]), np.var(d[4:, 4:]), np.var(d[4:, :4]))
def vbf(d):
    s = []
    for i in d:
        for j in i:
            if j >= 5:
                s.append(80)
            else:
                s.append(0)
    return np.var(s)

n_samples = len(digits.images)
data = digits.images

v = []
t = []
b = []
l = []
r = []
tr = []
tl = []
br = []
bl = []
srv = []
vb = []
max_a = []
for i in data:
    v.append(var(i))
    t.append(tf(i))
    b.append(bf(i))
    l.append(lf(i))
    r.append(rf(i))
    tr.append(trf(i))
    tl.append(tlf(i))
    br.append(brf(i))
    bl.append(blf(i))
    srv.append(srvf(i))
    vb.append(vbf(i))
    max_a.append(maxf(i))


x = np.column_stack((v, t, b, l, r, tr, tl, br, bl, srv, vb, max_a))
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