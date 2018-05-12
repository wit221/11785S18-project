from __future__ import print_function
import torch.utils.data as data
import errno
import os
import math
from functools import reduce

import numpy as np
import torch
from torch.utils.data import DataLoader

from robust_dim_scaler import RobustDimScaler


# This file is inspired by Pyro SS-VAE tutorial
# It contains utilities for caching, transforming and splitting NHANES data
# efficiently. By default, a Pytorch DataLoader will apply the transform every epoch
# we avoid this by caching the data early on in the NHANES class


# transformations for data
def fn_x_nhanes(x, reference_x = None):

    xt = np.zeros(x.shape)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            if x[r,c] == 1:
                xt[r,c] = 1

    #print(np.max(x), np.max(xt))
    xp = torch.from_numpy(xt)

    return xp


def fn_y_nhanes(y, reference_y=None):

    if reference_y is None:
        scaler = RobustDimScaler(y[:,:,0], -1., 0, 95)
    else:
        scaler = RobustDimScaler(reference_y[:,:,0], -1., 0, 95)

    # get mask before transforming
    m = np.zeros((y.shape[0],y.shape[1]))
    yc = np.ones((y.shape[0],y.shape[1]))
    ld = np.zeros((y.shape[0],y.shape[1]))
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            if y[r,c,0]>0:
                m[r, c] = 1
                if y[r,c,1] < 1:
                    ld[r,c] = 1
                    yc[r, c] = y[r, c, 0]
                else:
                    ld[r, c] = 0
                    yc[r, c] = y[r, c, 0] * math.sqrt(2)

    #yt = scaler.transform(yc)
    yt = yc

    # convert to torch
    yp = torch.from_numpy(yt)
    mp = torch.from_numpy(m)
    ldp = torch.from_numpy(ld)

    return yp, mp, ldp


def get_ss_indices_per_class(train_num, valid_num, test_num):

    # number of indices to consider
    #n_idxs = y.shape[0]
    n_idxs = train_num + valid_num + test_num
    idxs = list(range(n_idxs))

    np.random.shuffle(idxs)

    test_idx = idxs[:test_num]
    valid_idx = idxs[test_num:test_num+valid_num]
    train_idx = idxs[test_num+valid_num:n_idxs]

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


def split_train_valid_test(daty,datx, yidx,  train_num, valid_num, test_num=1000):
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

    # dat = np.zeros((datorig.shape[0],datorig.shape[1]))
    #
    # for r in range(datorig.shape[0]):
    #     for c in range(datorig.shape[1]):
    #         dat[r,c] = datorig[r,c,1]

    X = np.delete(datx,yidx,axis=1) # TODO we need to attach the relevant data, not the remainder of the chemicals
    Y = daty

    train_idx, valid_idx, test_idx = get_ss_indices_per_class(train_num, valid_num, test_num)

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
    chemlist = [i for i in range(32612,32620)]#[43, 45, 50, 60, 62, 64, 66, 73]
    # chemlist = [43, 45, 50, 60, 62, 64, 66, 73]

    # static class variables for caching training data
    #train_data_size = 3000
    #validation_size = 1000
    #test_size = 1000

    #train_size = 19480 # THIS IS FOR PREDICTION
    train_size = 9580 # THIS IS FOR SIMULATION
    validation_size = 2000
    test_size = 2000


    train_data, train_mask, train_ld, train_labels = None, None, None, None
    valid_data, valid_mask, valid_ld, valid_labels = None, None, None, None
    test_data, test_mask, test_ld, test_labels = None, None, None, None

    #raw_datax = '/Users/annabelova/Courses/2018SpringDeepLearning/Project/data/npy/predictorSample.npy'
    #raw_datay = '/Users/annabelova/Courses/2018SpringDeepLearning/Project/data/npy/sampleLab.npy'

    #raw_datax = '../Data/data_adult_2007-2014.npy'
    raw_datax='../Data/data_adult_2007-2014_counterfactual.npy'
    raw_datay = '../Data/labdata_2007-2014.npy'
    raw_suppl = '../Data/supplConsumers_adult_2007-2014.npy'
    raw_suppl_sim = 1 # 0 for overall, 1 for supplement consumers

    #raw_datax = '../Data/predictorSample.npy'
    #raw_datay = '../Data/sampleLab.npy'


    processed_folder = 'data/processed'
    training_file = 'training.pt'
    validation_file = 'validation.pt'
    test_file = 'test.pt'
    predictions_file = 'predictions.pt'

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
            if self.raw_suppl_sim==0:
                self.rawnpyx = np.load(os.path.join(self.root, NHANES.raw_datax))
                self.rawnpyy = np.load(os.path.join(self.root, NHANES.raw_datay))
            else:
                self.rawnpyx = np.load(os.path.join(self.root, NHANES.raw_datax))
                self.rawnpyy = np.load(os.path.join(self.root, NHANES.raw_datay))
                self.supplidx = np.load(os.path.join(self.root, NHANES.raw_suppl))
                self.rawnpyx = np.take(self.rawnpyx,self.supplidx,axis=0)
                self.rawnpyy = np.take(self.rawnpyy,self.supplidx,axis=0)

        except OSError as e:
            raise


        self.mode = mode

        assert mode in ["train", "test", "valid","prediction"], "invalid train/test option values"

        # Split data

        NHANES.train_data, NHANES.train_labels, \
            NHANES.valid_data, NHANES.valid_labels,  \
            NHANES.test_data, NHANES.test_labels = \
            split_train_valid_test(self.rawnpyy,self.rawnpyx, self.ychem_idx, NHANES.train_size, NHANES.validation_size, NHANES.test_size)

        if mode == "train":
            self.train_data, self.train_mask, self.train_ld = fn_y_nhanes(self.train_data)
            self.train_labels = fn_x_nhanes(self.train_labels)
            #torch.save((self.train_data, self.train_mask, self.train_ld, self.train_labels),
            #           NHANES.processed_folder+"/"+NHANES.training_file)

        elif mode == "valid":
            self.train_data, self.train_mask, self.train_ld = fn_y_nhanes(self.valid_data,self.train_data)
            self.train_labels = fn_x_nhanes(self.valid_labels, self.train_labels)
            #torch.save((self.train_data, self.train_mask, self.train_ld, self.train_labels),
            #           NHANES.processed_folder+"/"+NHANES.validation_file)

        elif mode == "test":
            self.train_data, self.train_mask, self.train_ld = fn_y_nhanes(self.test_data,self.train_data)
            self.train_labels = fn_x_nhanes(self.test_labels, self.train_labels)
            #torch.save((self.train_data, self.train_mask, self.train_ld, self.train_labels),
            #           NHANES.processed_folder+"/"+NHANES.test_file)

        else:
            self.train_data1, self.train_mask1, self.train_ld1 = fn_y_nhanes(self.train_data)
            self.train_labels1 = fn_x_nhanes(self.train_labels)
            self.train_data2, self.train_mask2, self.train_ld2 = fn_y_nhanes(self.valid_data,self.train_data)
            self.train_labels2 = fn_x_nhanes(self.valid_labels, self.train_labels)
            self.train_data3, self.train_mask3, self.train_ld3 = fn_y_nhanes(self.test_data,self.train_data)
            self.train_labels3 = fn_x_nhanes(self.test_labels, self.train_labels)

            self.train_data = torch.cat((self.train_data1,self.train_data2,self.train_data3), 0)
            self.train_mask = torch.cat((self.train_mask1,self.train_mask2,self.train_mask3), 0)
            self.train_ld = torch.cat((self.train_ld1,self.train_ld2,self.train_ld3), 0)
            self.train_labels = torch.cat((self.train_labels1,self.train_labels2,self.train_labels3), 0)

            #torch.save((self.train_data, self.train_mask, self.train_ld, self.train_labels),
            #           NHANES.processed_folder+"/"+NHANES.predictions_file)


        self.train_data = self.train_data.type(torch.FloatTensor)
        self.train_mask = self.train_mask.type(torch.FloatTensor)
        self.train_ld = self.train_ld.type(torch.FloatTensor)
        self.train_labels = self.train_labels.type(torch.FloatTensor)

        if use_cuda:
            self.train_data.cuda()
            self.train_mask.cuda()
            self.train_ld.cuda()
            self.train_labels.cuda()

    def __getitem__(self, index):
        """
        :param index: Index or slice object
        :returns tuple: (y, m ,x) where target is index of the target class.
        """
        y, m, l, x = self.train_data[index], self.train_mask[index], self.train_ld[index], self.train_labels[index]
        return y, m, l, x

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

    #for mode in ["train", "test", "valid"]:
    for mode in ["prediction"]:

        cached_data[mode] = dataset(root=root, mode=mode, use_cuda=use_cuda)

        if mode == "prediction":
            loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=False, **kwargs)
        else:
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
