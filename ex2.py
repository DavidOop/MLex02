__author__ = 'davidwer'
# David Wertenteil

"""

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
import statistics
# The digits dataset
digits = datasets.load_digits()

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# *****************************************************************************************************************
# --------------------------------------- 20 ----------------------------------------------------------------------

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
wrong_predict = []
for i in range(len(expected)):
    if expected[i] != predicted[i]:
        wrong_predict.append([expected[i], predicted[i], digits.images[n_samples // 2 + i]])
plt.suptitle("Test. mis-classification: expected - predicted")
# for index, (i, j, image) in enumerate(wrong_predict):
#     plt.subplot(3, 10, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('%i %i' % (i, j))
# plt.show()
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
            if self.classi(a) > self.classification:
                ima.append(1)
            else:
                ima.append(0)
        return ima
# ------------------------------------------------------------------------------------------------------


indices_0_1 = np.where(np.logical_and(digits.target >= 0, digits.target <= 1))
n_samples = len(digits.images[indices_0_1])
data = digits.images[indices_0_1].reshape((360, -1))

classifier = Classifier(var)

classifier.fit(data[:n_samples // 2], digits.target[indices_0_1][:n_samples // 2])
expected = digits.target[indices_0_1][n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.
         classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

plt.scatter(predicted, expected)
plt.xlabel("predicted")
plt.ylabel("expected")
plt.show()

# wrong_predict = []
# for i in range(len(expected)):
#     if expected[i] != predicted[i]:
#         wrong_predict.append([expected[i], predicted[i], digits.images[indices_0_1][n_samples // 2 + i]])
#         # print(digits.images[indices_0_1][n_samples // 2 + i], )
# plt.suptitle("Test. mis-classification: expected - predicted")
#
# for index, (i, j, image) in enumerate(wrong_predict):
#     plt.subplot(7, 10, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('%i %i' % (i, j))
# plt.show()
# -------------------------------------------------------------------------------------
# # creating the X (feature)
# X = np.column_stack((featureA[indices_0_1], featureB[indices_0_1]))
# # scaling the values for better classification performance
# X_scaled = preprocessing.scale(X)
# # the predicted outputs
# Y = digits.target[indices_0_1] # Training Logistic regression
# logistic_classifier = linear_model.LogisticRegression()
# logistic_classifier.fit(X_scaled, Y)
# # show how good is the classifier on the training data
# expected = Y
# predicted = logistic_classifier.predict(X_scaled)
# print("Logistic regression using [featureA, featureB] features:\n%s\n" % ( metrics.classification_report( expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
# # estimate the generalization performance using cross validation
# predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
# print("Logistic regression using [featureA, featureB] features cross validation:\n%s\n" % (metrics.classification_report(expected, predicted2)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))


#
# with open('numbers.txt', 'w') as f:
#
#     for i, j in zip(data, digits.target[indices_0_1]):
#         rr = []
#         for k, t in zip(i, range(64)):
#             rr.append(k)
#             if t != 0.0 and t % 8 == 0:
#                 f.write(str(rr) + '\n')
#                 rr = []
#         f.write('\n' + str(j))
#         f.write('\n' + '================================================================' + '\n\n')
