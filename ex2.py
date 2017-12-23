__author__ = 'davidwer'
__author2__ = 'omersc'
# David Wertenteil
# Omer Schwartz

import matplotlib.pyplot as plt
import numpy as np
# Standard scientific Python imports
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, svm, metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict

# ******************************************************************************
# -------------------- Import the database ----------------------

# The digits dataset
digits = datasets.load_digits()


# ------------------------------ Question 20 -----------------------------------
def q20():
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # We train on the digits on the first half of the data set
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])
    # print("Classification report for classifier %s:\n%s\n"
    #       % (classifier, metrics.classification_report(expected, predicted)))
    # print("Confusion matrix:\n%s" %
    #       metrics.confusion_matrix(expected, predicted))
    wrong_predict = []
    for index, expectation, prediction in zip(range(len(expected)),
                                              expected,
                                              predicted):
        if expectation != prediction:
            wrong_predict.append((expectation,
                                  prediction,
                                  digits.images[(n_samples // 2) + index]))

    plt.suptitle("Test. mis-classification: expected - predicted")
    for index, (expectation, prediction, image) in enumerate(wrong_predict):
        plt.subplot(3, 10, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%i %i' % (expectation, prediction))
    plt.show()


# ******************************************************************************
# ------------------------------- 21 Our Classifiers ---------------------------

def center_values(img):
    """
    Sum the middle columns' pixel values

    Confusion matrix:
    [[88  0]
    [10 82]]

    :param img: the 8x8 pixel matrix as a list:
    :return sum of 2 middle columns:
    """
    return img[19] + img[27] + img[35] + img[43] + img[20] + img[28] + img[36] \
           + img[44]


# ------------------------------------------------------------------------------


def num_of_zeros(img):
    """
    Count the amount of zero pixels in the image

    Confusion matrix:
    [[65 23]
    [ 10 82]]
    :param img: the 8x8 pixel matrix as a list:
    :return: the amount of zero pixels in the image
    """
    return img.count(0.0)


# ------------------------------------------------------------------------------


def modulus(img):
    """
    Remove all black

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


# -------------------------------- 21e 3D figure -------------------------------
def q21():
    indices_0_1 = np.where(
        np.logical_and(digits.target >= 0, digits.target <= 1))
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
    ax.scatter(circle_finder_arr, modulus_arr, center_values_arr,
               c=digits.target[indices_0_1],
               cmap=plt.cm.Set1, edgecolor='k', s=30)
    plt.show()

    # ------------------- 21f  Logistic Classifier ---------------

    # creating the X (feature)
    X = np.column_stack((circle_finder_arr, modulus_arr, center_values_arr,
                         num_of_zeros_arr))
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
    print(
        "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    # estimate the generalization performance using cross validation
    predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
    print(
        "Logistic regression using Circle Finder, Modulus, "
        "Center Values, Number of Zeros features with cross validation:"
        "\n%s\n" % (metrics.classification_report(expected, predicted2)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected,
                                                             predicted2))

    # ---------------------------- 21g ---------------------------

    class Classifier:
        def __init__(self):
            self.fit_zero = 0
            self.fit_one = 0
            self.zero = lambda x: abs(self.fit_one - x) < abs(self.fit_zero - x)

        def fit(self, properties, targets):
            zero = []
            one = []
            for i, j in zip(properties, targets):
                if j:
                    zero.append(i)
                else:
                    one.append(i)
            self.fit_zero = np.mean(zero)
            self.fit_one = np.mean(one)

        def predict(self, properties):
            predicted = []
            for i in properties:
                if self.zero(i):
                    predicted.append(0)
                else:
                    predicted.append(1)
            return predicted

    # ------------------------- 21g ---------------------------
    ima = []
    classy = (np.mean(circle_finder_arr[:180]) + np.mean(modulus_arr[:180])
              + np.mean(center_values_arr[:180])) / 3
    print(classy)
    for i, j, k in zip(circle_finder_arr[180:], modulus_arr[180:],
                       center_values_arr[180:]):
        if classy > (i + j + k) / 3:
            ima.append(0)
        else:
            ima.append(1)

    classifier = Classifier()
    classifier.fit(circle_finder_arr[:n_samples // 2],
                   digits.target[indices_0_1][:n_samples // 2])
    predicted = classifier.predict(circle_finder_arr[n_samples // 2:])
    print(
        "num_of_zeros_arr features:"
        "\n%s\n" %
        (metrics.classification_report(
            digits.target[indices_0_1][n_samples // 2:], predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(
        digits.target[indices_0_1][n_samples // 2:], predicted))
    print(
        "num_of_zeros_arr features:"
        "\n%s\n" %
        (metrics.classification_report(
            digits.target[indices_0_1][n_samples // 2:], ima)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(
        digits.target[indices_0_1][n_samples // 2:], ima))

    # ---------------------------- 21h ---------------------------
    # ima = []
    # n_samples = len(digits.images)
    # data = digits.images.reshape((n_samples, -1))
    # mat = [[]] * 10
    #
    #
    # for i, j in zip(data[:n_samples // 2], digits.target[:n_samples // 2]):
    #     pixel_list = np.ndarray.tolist(i)
    #     mat[j].append(modulus(pixel_list))
    #
    # sum_arr = []
    # for i in mat:
    #     # print(i)
    #     sum_arr.append(np.mean(i))
    #
    # predict_mat = []
    #
    # for i in data[n_samples // 2:]:
    #     pixel_list = np.ndarray.tolist(i)
    #     n = [abs(j - pixel_list) for j in sum_arr]
    #     print(n[1])
    #     # n[1] is a list of lists
    #     predict_mat.append()

    # print(
    #     "num_of_zeros_arr features:"
    #     "\n%s\n" % (metrics.classification_report(
    #         digits.target[n_samples // 2:], predict_mat)))
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(
    #     digits.target[n_samples // 2:], predict_mat))
    # ------------------------- 21h ---------------------------


q20()
q21()
