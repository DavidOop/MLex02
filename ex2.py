__author__ = 'davidwer'
__author__ = 'omersc'
# David Wertenteil
# Omer Schwartz

# Standard scientific Python imports
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# *****************************************************************************************************************
# --------------------------------------- 20 ----------------------------------------------------------------------

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

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
for index, (i, j, image) in enumerate(wrong_predict):
    plt.subplot(3, 10, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i %i' % (i, j))
# plt.show()
# *****************************************************************************************************************
# ------------------------------- 21 Our Classifiers --------------------------------------------------------------


def center_values(img):
    """
    Confusion matrix:
    [[88  0]
    [10 82]]

    :param img:
    :return:
    """
    return img[19] + img[27] + img[35] + img[43] + img[20] + img[28] + img[36] \
            + img[44]


# ------------------------------------------------------------------------------


def num_of_zeros(img):
    """
    Confusion matrix:
    [[65 23]
    [ 10 82]]
    :param img:
    :return:
    """
    return img.count(0.0)


# ------------------------------------------------------------------------------


def modulus(img):
    """
    Confusion matrix:
    [[85  3]
    [ 4 88]]

    :param img:
    :return:
    """
    a = []
    for i in img:
        a.append(-(int(i) % 16))
    return sum(a)


# ------------------------------------------------------------------------------


def circle_finder(img):
    """
    Confusion matrix:
    [[88  0]
    [ 1 91]]
    :param img:
    :return:
    """
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


# ------------------------------------------------------------------------------


def var(img):
    """
    Confusion matrix:
    [[78 10]
    [ 24 68]]
    :param img:
    :return:
    """
    return np.var(img)


# ------------------------------------------------------------------------------


def classify(classifier, test_set):
    ima = []
    for i in test_set:
        pixel_list = np.ndarray.tolist(i)
        ima.append(classifier(pixel_list))
    return ima


# -------------------------------- 21e 3D figure ----------------------------------------
indices_0_1 = np.where(np.logical_and(digits.target >= 0, digits.target <= 1))
n_samples = len(digits.images[indices_0_1])
data = digits.images[indices_0_1].reshape((360, -1))

circle_finder_arr = classify(circle_finder, data)
modulus_arr = classify(modulus, data)
center_values_arr = classify(center_values, data)
num_of_zeros_arr = classify(num_of_zeros, data)
var_arr = classify(var, data)

fig = plt.figure()
ax = Axes3D(fig)
fig.suptitle("num_of_zeros, circle_finder, center_values")
ax.set_xlabel('Circle Finder')
ax.set_ylabel('circle_finder')
ax.set_zlabel('center_values')
ax.scatter(circle_finder_arr, modulus_arr, center_values_arr, c=digits.target[indices_0_1],
           cmap=plt.cm.Set1, edgecolor='k', s=30)
plt.show()
# ------------------------------ 21f  Logistic Classifier -----------------------------------------------------------


# creating the X (feature)
X = np.column_stack((circle_finder_arr, modulus_arr, center_values_arr, num_of_zeros_arr))
# scaling the values for better classification performance
X_scaled = preprocessing.scale(X)
# the predicted outputs
Y = digits.target[indices_0_1]  # Training Logistic regression
logistic_classifier = linear_model.LogisticRegression()
logistic_classifier.fit(X_scaled, Y)
# show how good is the classifier on the training data
expected = Y
predicted = logistic_classifier.predict(X_scaled)

print("Logistic regression using Circle Finder, Modulus, "
      "Center Values, Number of Zeros features:\n%s\n" %
      (metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
# estimate the generalization performance using cross validation
predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
print(
    "Logistic regression using Circle Finder, Modulus, "
    "Center Values, Number of Zeros features with cross validation:"
    "\n%s\n" % (metrics.classification_report(expected, predicted2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))


# ---------------------------------------------- 21g -----------------------------------------------------------
ima = []
classi = (np.mean(circle_finder_arr[:180]) + np.mean(modulus_arr[:180]) + np.mean(center_values_arr[:180]))/3
print(classi)
for i, j ,k in zip(circle_finder_arr[180:], modulus_arr[180:], center_values_arr[180:]):
    if classi > (i + j + k)/3:
        ima.append(0)
    else:
        ima.append(1)

print(
    "num_of_zeros_arr features:"
    "\n%s\n" % (metrics.classification_report( digits.target[indices_0_1][n_samples // 2:], ima)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix( digits.target[indices_0_1][n_samples // 2:], ima))

# ---------------------------------------------- 21h -----------------------------------------------------------
ima = []
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
mat = [[]] * 10


for i, j in zip(data[:n_samples // 2], digits.target[:n_samples // 2]):
    pixel_list = np.ndarray.tolist(i)
    mat[j].append(modulus(pixel_list))

sum_arr = []
for i in mat:
    # print(i)
    sum_arr.append(np.mean(i))

predict_mat = []

for i in data[n_samples // 2:]:
    pixel_list = np.ndarray.tolist(i)
    n = [abs(j - pixel_list) for j in sum_arr]
    print(n[1])
    # n[1] is a list of lists
    predict_mat.append()

print(
    "num_of_zeros_arr features:"
    "\n%s\n" % (metrics.classification_report( digits.target[n_samples // 2:], predict_mat)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix( digits.target[n_samples // 2:], predict_mat))

