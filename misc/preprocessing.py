import numpy as np
import math


def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """

    sample_means = np.mean(x, axis=1)
    res = [x[i, ] - sample_means[i] for i in range(len(sample_means))]
    return np.array(res, dtype='float32', copy=True)


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """

    sample_variances = np.var(x, axis=1)
    res = [scale * x[i, ] / math.sqrt(bias + sample_variances[i]) for i in range(len(sample_variances))]
    return np.array(res, dtype='float32', copy=True)


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """

    feature_means = np.mean(x, axis=0)
    res_train = np.array([x[:, i] - feature_means[i] for i in range(len(feature_means))], dtype='float32', copy=True)
    res_test = np.array([xtest[:, i] - feature_means[i] for i in range(len(feature_means))], dtype='float32', copy=True)
    return np.transpose(res_train), np.transpose(res_test)


def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """

    n = x.shape[0]
    m = x.shape[1]
    identity_matrix = np.identity(m, dtype='float32')
    sigma_matrix = np.dot(np.transpose(x), x) / n + bias * identity_matrix
    u, s, v = np.linalg.svd(sigma_matrix)
    pca_matrix = np.dot(np.dot(u, np.diag(1. / np.sqrt(s))), np.transpose(u))
    zca_train = np.dot(x, pca_matrix)
    zca_test = np.dot(xtest, pca_matrix)
    return np.array(zca_train, dtype='float32', copy=True), np.array(zca_test, dtype='float32', copy=True)


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """

    samples_train = x.shape[0]
    samples_test = xtest.shape[0]
    xtrain_zm = sample_zero_mean(x)
    xtest_zm = sample_zero_mean(xtest)
    xtrain_zm_gcn = gcn(xtrain_zm)
    xtest_zm_gcn = gcn(xtest_zm)
    xtrain_fzm, xtest_fzm = feature_zero_mean(xtrain_zm_gcn, xtest_zm_gcn)
    xtrain_zca, xtest_zca = zca(xtrain_fzm, xtest_fzm)
    res_train = np.reshape(xtrain_zca, (samples_train, 3, image_size, image_size))
    res_test = np.reshape(xtest_zca, (samples_test, 3, image_size, image_size))
    return res_train, res_test
