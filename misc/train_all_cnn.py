#!/usr/bin/env python3
"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from preprocessing import cifar_10_preprocess
#from all_cnn_actual import all_cnn_module
from all_cnn_adverse import all_cnn_module
import torch.nn.init as init
import torch.cuda as cuda

meanGradInLen = 0
meanGradOutLen = 0
gradQty = 0


def hook_func(module, grad_input, grad_output):
    global meanGradInLen
    global meanGradOutLen
    global gradQty
    # print('====================')
    norm_in = float(np.linalg.norm(grad_input))
    norm_out = float(np.linalg.norm(grad_output))

    meanGradInLen += (norm_in - meanGradInLen) / (gradQty + 1)
    meanGradOutLen += (norm_out - meanGradOutLen) / (gradQty + 1)

    print('##', norm_in, norm_out, meanGradInLen, meanGradOutLen, gradQty)

    gradQty += 1


class MakeData(Dataset):
    __xs = []
    __ys = []

    def __init__(self, fdata, ldata):
        self.__xs = fdata
        self.__ys = ldata

    # Override
    def __getitem__(self, index):
        fsub = self.__xs[index]
        lsub = self.__ys[index]
        return fsub, lsub

    # Override
    def __len__(self):
        return len(self.__xs)

class MakeDataFake(Dataset):
    __xs = []
    __ys = []

    def __init__(self, fdata, target_cat):
        self.__xs = fdata
        self.__ys = np.repeat(target_cat, fdata.shape[0], axis=0)

    # Override
    def __getitem__(self, index):
        fsub = self.__xs[index]
        lsub = self.__ys[index]
        return fsub, lsub

    # Override
    def __len__(self):
        return len(self.__xs)


def init_weights(m):
    if type(m) == nn.Conv2d:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.0)


def write_results(predictions, output_file='predictions.txt'):
    """
    Write predictions to file for submission.
    File should be:
        named 'predictions.txt'
        in the root of your tar file
    :param predictions: iterable of integers
    :param output_file:  path to output file.
    :return: None
    """
    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y))


def main(argv):
    if len(argv) != 6:
        sys.stderr.write(
            'Usage: <data dir><seed><batch size><epochs><cuda/cpu>\n')
        sys.exit(1)

    data_dir = argv[1] + '/'
    seed = int(argv[2])
    batch_size = int(argv[3])
    epoch_qty = int(argv[4])
    cudaArg = argv[5]
    np.random.seed(seed)

    if cudaArg=='cuda':
        USE_CUDA = True
    else:
        USE_CUDA = False

    m_o_d_e_l__n_a_m_e = 'model.pkl'

    ftrain = np.load(data_dir + 'train_feats.npy')
    ltrain = np.load(data_dir + 'train_labels.npy')
    ftest = np.load(data_dir + 'test_feats.npy')

    xtrain, xtest = cifar_10_preprocess(ftrain, ftest)
    xtest = torch.from_numpy(xtest)
    train_dataset = MakeData(xtrain, ltrain)
    #train_dataset = MakeDataFake(xtrain, 0)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    if os.path.exists(m_o_d_e_l__n_a_m_e):
        print('Loading model')
        model = all_cnn_module()
        model.load_state_dict(torch.load(m_o_d_e_l__n_a_m_e))
        if USE_CUDA:
            model = model.cuda()
        else:
            model = model.cpu()
    else:
        print('Creating model from scratch')
        if USE_CUDA:
            model = all_cnn_module().cuda()
        else:
            model = all_cnn_module().cpu()
        model.apply(init_weights)

    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001, nesterov=True)

    if True:
        model.train()
        #model.register_backward_hook(hook_func)
        for epoch in range(epoch_qty):
            for i, (features, labels) in enumerate(train_loader):

                if USE_CUDA:
                    features = Variable(features.cuda())
                    labels = Variable(labels.cuda()).long()
                else:
                    features = Variable(features.cpu())
                    labels = Variable(labels.cpu()).long()

                grad_qty = 0.0
                mean_grad_in_len = 0
                mean_grad_out_len = 0
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                for param in model.parameters():
                    print(param.data)

                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Averge gradient norm: %g/%g'
                          % (epoch + 1, epoch_qty, i + 1, len(train_dataset) // batch_size, loss.data[0],
                             mean_grad_in_len, mean_grad_out_len))

    # Save the Trained Model
    torch.save(model.state_dict(), m_o_d_e_l__n_a_m_e)

    # Test the Model
    model.eval()
    correct = 0
    total = 0

    for features, labels in train_loader:

        if USE_CUDA:
            features = Variable(features.cuda())
            # labels = Variable(labels.cuda()).long()
        else:
            features = Variable(features.cpu())
            # labels = Variable(labels.cpu()).long()

        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Train Accuracy of the model: %d %%' % (100 * correct / total))

    xtest_dataset = MakeData(xtest, np.zeros(xtest.shape[0], dtype=np.int32))
    xtest_loader = DataLoader(dataset=xtest_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    all_predictions = []

    for features, _ in xtest_loader:
        if USE_CUDA:
            features_test = Variable(features.cuda())
        else:
            features_test = Variable(features.cpu())
        output_test = model(features_test)
        _, predictions = torch.max(output_test.data, 1)
        all_predictions.extend(predictions.cpu().numpy())

    write_results(all_predictions)


if __name__ == '__main__':
    main(sys.argv)
