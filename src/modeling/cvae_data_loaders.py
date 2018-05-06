from __future__ import print_function
import torch.utils.data as data
import errno
import os
from functools import reduce

import numpy as np
import torch
from torch.utils.data import DataLoader


# This file is inspired by Pyro SS-VAE tutorial
# It contains utilities for caching, transforming and splitting NHANES data
# efficiently. By default, a Pytorch DataLoader will apply the transform every epoch
# we avoid this by caching the data early on in the NHANES class


# transformations for data
def fn_x_nhanes(x, use_cuda):

    xt = np.full(x.shape,1)
    i_below_llod = (x==0)
    i_missing = (x==9)
    xt[i_below_llod] = 0
    xt[i_missing] = 0

    # convert to torch
    xp = torch.from_numpy(xt)
    xp = xp.type(torch.FloatTensor)

    # send the data to GPU(s)
    if use_cuda:
        xp = xp.cuda()

    return xp


def fn_y_nhanes(y, use_cuda):

    # convert to torch
    yt = torch.from_numpy(y)

    yp = torch.zeros(yt.size(0), 5)

    # send the data to GPU(s)
    if use_cuda:
        yp = yp.cuda()
        yt = yt.cuda()

    # transform the label y (integer between 0 and 5) to a one-hot
    yp = yp.scatter_(1, yt.view(-1, 1), 1.0)

    return yp


def get_ss_indices_per_class(y, valid_num, test_num):

    # number of indices to consider
    n_idxs = y.shape[0]

    # obtain indices per class
    idxs_per_class = {j: [] for j in range(2)}

    for i in range(n_idxs):
        if y[i] in list(range(2)):
            idxs_per_class[y[i]].append(i)


    #print(idxs_per_class)

    train_idx = []
    valid_idx = []
    test_idx = []

    for j in range(2):

        n_idxs_class = len(idxs_per_class[j])
        n_valid_class = int(float(n_idxs_class) * float(valid_num) / float(n_idxs))
        n_test_class = int(float(n_idxs_class) * float(test_num) / float(n_idxs))

        np.random.shuffle(idxs_per_class[j])

        test_idx.extend(idxs_per_class[j][:n_test_class])
        valid_idx.extend(idxs_per_class[j][n_test_class:n_valid_class+n_test_class])
        train_idx.extend(idxs_per_class[j][n_valid_class+n_test_class:n_idxs_class])

    return train_idx, valid_idx, test_idx


def split_train_valid_test(dat, yidx,  valid_num, test_num=1000):
    """
    helper function for splitting the data into train, validation parts and
    test parts
    :param X: predictors (socio-deographics, examination, diet, supplements)
    :param y: labels (measurements)
    :param validation_num: what number of examples to use for validation
    :param test_num: what number of examples to use for test
    :return: splits of data
    """
    X = np.delete(dat,yidx,axis=1)
    y = np.take(dat,yidx,axis=1)

    train_idx, valid_idx, test_idx = get_ss_indices_per_class(y, valid_num, test_num)

    # test set
    X_test = X[test_idx]
    y_test = y[test_idx]

    # validation set
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    # train set
    X_train = X[train_idx]
    y_train = y[train_idx]

    return y_train, X_train,  y_valid, X_valid, y_test,  X_test


def print_distribution_labels(y):
    """
    helper function for printing the distribution of class labels in a dataset
    :param y: tensor of class labels given as one-hots
    :return: a dictionary of counts for each label from y
    """
    counts = {j: 0 for j in range(10)}
    for i in range(y.size()[0]):
        for j in range(10):
            if y[i][j] == 1:
                counts[j] += 1
                break
    print(counts)


class NHANES(data.Dataset):
    """
    Class to transform, load and cache NHANES data
    once at the beginning of the inference
    """

    # static class variables for caching training data
    train_data_size = 5215
    validation_size = 1000
    test_size = 1000

    train_data, train_labels = None, None
    valid_data, valid_labels = None, None
    test_data, test_labels = None, None

    raw_data = 'data/npy/quantized_dense_labdata_2013-2014.npy'
    #processed_folder = 'data/processed'
    #training_file = 'training.pt'
    #test_file = 'test.pt'

    def __init__(self, root, mode, ychem_idx=66, use_cuda=True, *args, **kwargs):

        self.root = os.path.expanduser(root)
        self.ychem_idx = ychem_idx # default will take blood lead as target

        #try:
        #    os.makedirs(os.path.join(self.root, NHANES.processed_folder))
        #except OSError as e:
        #    if e.errno == errno.EEXIST:
        #        pass
        #    else:
        #        raise

        try:
            self.rawnpy = np.load(os.path.join(self.root, NHANES.raw_data))
        except OSError as e:
            raise


        self.mode = mode

        assert mode in ["train", "test", "valid"], "invalid train/test option values"

        # Split data

        NHANES.train_data, NHANES.train_labels, \
            NHANES.valid_data, NHANES.valid_labels,  \
            NHANES.test_data, NHANES.test_labels = \
            split_train_valid_test(self.rawnpy, self.ychem_idx, NHANES.validation_size, NHANES.test_size)

        # transformations on NHANES data
        def transform(x):
            return fn_x_nhanes(x, use_cuda)

        def target_transform(y):
            return fn_x_nhanes(y, use_cuda)

        if mode == "train":
            self.train_data = (transform(self.train_data)).float()
            self.train_labels = (target_transform(self.train_labels))

        elif mode == "valid":
            self.train_data = (transform(self.valid_data)).float()
            self.train_labels = (target_transform(self.valid_labels))

        else:
            self.train_data = (transform(self.test_data)).float()
            self.train_labels = (target_transform(self.test_labels))

    def __getitem__(self, index):
        """
        :param index: Index or slice object
        :returns tuple: (ino, target) where target is index of the target class.
        """
        y, x = self.train_data[index], self.train_labels[index]
        return y, x

    def __len__(self):
        return len(self.train_data)


def setup_data_loaders(dataset, use_cuda, batch_size, root='.', **kwargs):
    """
        helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param dataset: the data to use
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param root: where on the filesystem should the dataset be
    :param kwargs: other params for the pytorch data loader
    :return: three data loaders: (supervised data for training, supervised data for validation,
                                  supervised data for testing)
    """
    # instantiate the dataset as training/testing sets
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 0, 'pin_memory': False}

    cached_data = {}
    loaders = {}
    for mode in ["train", "test", "valid"]:

        cached_data[mode] = dataset(root=root, mode=mode, use_cuda=use_cuda)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs)

    return loaders


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


EXAMPLE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
DATA_DIR = os.path.join(EXAMPLE_DIR, 'data')
RESULTS_DIR = os.path.join(EXAMPLE_DIR, 'results')
