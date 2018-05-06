from __future__ import print_function
import torch.utils.data as data
import errno
import os
from functools import reduce

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler, RobustScaler


# This file is inspired by Pyro SS-VAE tutorial
# It contains utilities for caching, transforming and splitting NHANES data
# efficiently. By default, a Pytorch DataLoader will apply the transform every epoch
# we avoid this by caching the data early on in the NHANES class


# transformations for data
def fn_x_nhanes(x, reference_x = None):

    scaler = RobustScaler(quantile_range=(5, 95))

    if reference_x is None:
        scaler.fit(x)
    else:
        scaler.fit(reference_x)

    xt = scaler.transform(x)
    xp = torch.from_numpy(xt)

    return xp


def fn_y_nhanes(y, reference_y=None):

    scaler = RobustScaler(with_centering=False, quantile_range=(0, 95))

    if reference_y is None:
        scaler.fit(y)
    else:
        scaler.fit(reference_y)

    # get mask before transforming
    m = np.zeros((y.shape))
    yc = np.ones((y.shape))
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            if y[r,c]>0:
                m[r, c] = 1
                yc[r,c] = y[r,c]

    yt = scaler.transform(yc)

    # convert to torch
    yp = torch.from_numpy(yt)
    mp = torch.from_numpy(m)

    return yp, mp


def get_ss_indices_per_class(y, valid_num, test_num):

    # number of indices to consider
    n_idxs = y.shape[0]
    idxs = list(range(n_idxs))

    np.random.shuffle(idxs)

    train_idx = idxs[:test_num]
    valid_idx = idxs[test_num:test_num+valid_num]
    test_idx = idxs[test_num+valid_num:n_idxs]

    return train_idx, valid_idx, test_idx


def max_vals(matr):
    nr, nc = matr.shape

    pct = np.zeros(nc)
    mx = pct
    for i in range(nc):
        col = matr[:,i].reshape(-1)
        col = col[col > 0]
        if col.shape[0] > 0:
            pct[i] = np.percentile(col, 95)
            mx[i] = np.max(col)

    return pct, mx

def cut_max(matr, upperBound):
    nr, nc = matr.shape
    for r in range(nr):
        for c in range(nc):
            matr[r,c] = min(matr[r,c],  upperBound[c])


def split_train_valid_test(datorig, yidx,  valid_num, test_num=1000):
    """
    helper function for splitting the data into train, validation parts and
    test parts.
    :param X: predictors (socio-deographics, biometrics, diet, supplements)
    :param Y: measurements
    :param validation_num: what number of examples to use for validation
    :param test_num: what number of examples to use for test
    :return: scaled splits of data
    """

    # TODO this data processing step is only relevant for lab data (measurements)

    dat = np.zeros((datorig.shape[0],datorig.shape[1]))

    for r in range(datorig.shape[0]):
        for c in range(datorig.shape[1]):
            dat[r,c] = datorig[r,c,1]

    X = np.delete(dat,yidx,axis=1) # TODO we need to attach the relevant data, not the remainder of the chemicals
    Y = np.take(dat,yidx,axis=1)

    train_idx, valid_idx, test_idx = get_ss_indices_per_class(Y, valid_num, test_num)

    # train set
    X_train = X[train_idx]
    Y_train = Y[train_idx]

    # test set
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    # validation set
    X_valid = X[valid_idx]
    Y_valid = Y[valid_idx]

    return Y_train,  X_train,  Y_valid,  X_valid, Y_test, X_test



class NHANES(data.Dataset):
    """
    Class to transform, load and cache NHANES data
    once at the beginning of the inference
    """

    # List of chemicals
    chemlist = [43, 45, 50, 60, 62, 64, 66, 73]

    # static class variables for caching training data
    train_data_size = 5215
    validation_size = 1000
    test_size = 1000

    train_data, train_mask, train_labels = None, None, None
    valid_data, valid_mask, valid_labels = None, None, None
    test_data, test_mask, test_labels = None, None, None

    raw_data = 'data/npy/dense_labdata_2013-2014.npy'
    processed_folder = 'data/processed'
    training_file = 'training.pt'
    validation_file = 'validation.pt'
    test_file = 'test.pt'

    def __init__(self, root, mode, ychem_idx=chemlist, use_cuda=True, *args, **kwargs):

        self.root = os.path.expanduser(root)
        self.ychem_idx = ychem_idx # default will take blood lead as target

        try:
            os.makedirs(os.path.join(self.root, NHANES.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

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

        if mode == "train":
            self.train_data, self.train_mask = fn_y_nhanes(self.train_data)
            self.train_labels = fn_x_nhanes(self.train_labels)
            torch.save((self.train_data, self.train_mask, self.train_labels),
                       NHANES.processed_folder+"/"+NHANES.training_file)

        elif mode == "valid":
            self.train_data, self.train_mask = fn_y_nhanes(self.valid_data,self.train_data)
            self.train_labels = fn_x_nhanes(self.valid_labels, self.train_labels)
            torch.save((self.train_data, self.train_mask, self.train_labels),
                       NHANES.processed_folder+"/"+NHANES.validation_file)

        else:
            self.train_data, self.train_mask = fn_y_nhanes(self.test_data,self.train_data)
            self.train_labels = fn_x_nhanes(self.test_labels, self.train_labels)
            torch.save((self.train_data, self.train_mask, self.train_labels),
                       NHANES.processed_folder+"/"+NHANES.test_file)

        self.train_data = self.train_data.type(torch.FloatTensor)
        self.train_mask = self.train_mask.type(torch.FloatTensor)
        self.train_labels = self.train_labels.type(torch.FloatTensor)

        if use_cuda:
            self.train_data.cuda()
            self.train_mask.cuda()
            self.train_labels.cuda()

    def __getitem__(self, index):
        """
        :param index: Index or slice object
        :returns tuple: (y, m ,x) where target is index of the target class.
        """
        y, m, x = self.train_data[index], self.train_mask[index], self.train_labels[index]
        return y, m, x

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
