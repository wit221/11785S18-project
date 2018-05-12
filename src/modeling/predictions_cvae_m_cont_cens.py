import argparse, sys, pdb, math
import os

import torch
import torch.nn as nn

import numpy as np

import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log, set_seed
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from predictions_cvae_m_cont_cens_data_loaders import NHANES, mkdir_p, setup_data_loaders
from cens_distr import LogNormCensPyro

def squishing_nonlin(x):
    return torch.sqrt(torch.sqrt(x))
    #return torch.sqrt(torch.log(x+1.01))
    #return torch.log(1.01 + torch.sigmoid(x))

def check_valid_range(var):
    return np.sum(np.isinf(var.data.cpu().numpy())) == 0 and \
            np.sum(np.isnan(var.data.cpu().numpy())) == 0

def my_initializer(m):
    """
    Simple initializer
    """
    if hasattr(m, 'weight'):
        torch.nn.init.xavier_uniform_(m.weight.data, gain =0.1)
    if hasattr(m, 'bias'):
        m.bias.data.zero_()

class Encoder(nn.Module):
    '''
    PyTorch module that parameterizes the
    diagonal gaussian distribution q(z|y,x)
    '''
    def __init__(self, z_dim, hidden_dim, y_dim, x_dim, use_cuda):
        super(Encoder, self).__init__()

        self.use_cuda = use_cuda

        if self.use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()


        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim

        # setup the three linear transformations
        self.fcy = nn.Linear(self.y_dim, hidden_dim)
        self.fcx = nn.Linear(self.x_dim, hidden_dim)
        self.fc0 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.fc1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.fc21 = nn.Linear(2*hidden_dim, self.z_dim)
        self.fc22 = nn.Linear(2*hidden_dim, self.z_dim)

        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.apply(my_initializer)

    def forward(self, y, x):
        '''
        Forward computation on measurements (y) and conditioning variables (x)
        :param y: measurements
        :param x: conditioning variables
        :return: location vector, scale vector for latent variables
        '''
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        #print(x.size())
        #print(y.size())

        # TODO: determine if this is needed
        # first shape the mini-batch to have data in the rightmost dimension
        y = y.reshape(-1, self.y_dim)
        x = x.reshape(-1, self.x_dim)

        # compute the hidden units
        hiddenx = squishing_nonlin(self.softplus(self.fcx(x)))
        hiddeny = squishing_nonlin(self.softplus(self.fcy(y)))
        hidden0 = squishing_nonlin(self.softplus(torch.cat((hiddeny, hiddenx), 1)))
        hidden1 = squishing_nonlin(self.softplus(hidden0))
        hidden2 = squishing_nonlin(self.softplus(hidden1))

        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.softplus(self.fc21(hidden2))
        z_scale = squishing_nonlin(self.softplus(self.fc22(hidden2)))

        #print("zloc",z_loc)
        #print("zscale", z_scale)

        return z_loc, z_scale


class Decoder(nn.Module):
    '''
    PyTorch module that parameterized the observation
    likelihood p(y|z,x)
    '''
    def __init__(self, z_dim, hidden_dim, y_dim, x_dim, use_cuda):
        super(Decoder, self).__init__()

        self.use_cuda = use_cuda

        if self.use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim

        # setup the two linear transformations used
        self.fcx = nn.Linear(self.x_dim, hidden_dim)
        self.fcz = nn.Linear(self.z_dim, hidden_dim)
        self.fc0 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.fc1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.fc21 = nn.Linear(2*hidden_dim, self.y_dim)
        self.fc22 = nn.Linear(2*hidden_dim, self.y_dim)


        # setup the non-linearities
        self.softplus = nn.Softplus()

        self.apply(my_initializer)

    def forward(self, z, x):
        '''
        Forward computation on latent variables z and conditioning variables x
        :param z: latent variables
        :param x: conditioning variables
        :return: parameters of the measurement distribution
        '''

        if self.use_cuda:
            z = z.cuda()
            x = x.cuda()

        hiddenz = squishing_nonlin(self.softplus(self.fcz(z)))
        hiddenx = squishing_nonlin(self.softplus(self.fcx(x)))
        hidden0 = squishing_nonlin(self.softplus(torch.cat((hiddenz, hiddenx), 1)))
        hidden1 = squishing_nonlin(self.softplus(hidden0))
        hidden2 = squishing_nonlin(self.softplus(hidden1))

        # return the parameters

        mu = self.fc21(hidden2)
        sigma = squishing_nonlin(self.softplus(self.fc22(hidden2)))

        return mu, sigma


class CVAE(nn.Module):
    '''
    PyTorch module for conditional VAE
    '''

    def __init__(self, y_dim, x_dim, z_dim=50, hidden_dim=400,  use_cuda=False):
        super(CVAE, self).__init__()

        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim, y_dim, x_dim, use_cuda)
        self.decoder = Decoder(z_dim, hidden_dim, y_dim, x_dim, use_cuda)


        self.use_cuda = use_cuda

        if self.use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_dim = x_dim

        self.debug_qty = 0


    def model(self, y, m, l, x):
        '''
        Define the model p(y|z,x)p(z)
        :param y: measurements
        :param x: conditioning variables
        :return: generated parameters of measurement distribution
        '''

        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        with pyro.iarange("data", x.size(0)):

            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.size(0), self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.size(0), self.z_dim)))

            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent_prior", dist.Normal(z_loc, z_scale).independent(1))

            #print('MODEL z=', z)


            # decode the latent code z
            mu, sigma = self.decoder.forward(z, x)

            #print('MODEL mu=', mu, 'mu=', sigma)


            # score against actual measurements
            pyro.sample("obs",
                        LogNormCensPyro(mu, sigma, l).mask(m.reshape(-1,self.y_dim)).independent(1),
                        obs=y.reshape(-1, self.y_dim))

            #pyro.sample("obs",
            #            dist.LogNormal(mu, sigma).mask(m.reshape(-1,self.y_dim)).independent(1),
            #            obs=y.reshape(-1, self.y_dim))


            # return the loc so we can visualize it later
            return mu, sigma

    def guide(self, y, m, l, x):
        '''
        Define the guide (i.e. variational distribution), q(z| y, x)
        :param y: measurements
        :param x: conditioning variables
        :return: None
        '''

        if self.use_cuda:
            y = y.cuda()
            x = x.cuda()
            m = m.cuda()
            l = l.cuda()

        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # print(x.is_cuda, y.is_cuda, m)
        yFused = self.sim_measurements(x, 1) * (1 - m ) + m * y
        #yFused = y

        with pyro.iarange("data", x.size(0)):

            # use the encoder to get the parameters used to define q(z|y,x)
            z_loc, z_scale = self.encoder.forward(yFused, x)

            # sample the latent code z
            pyro.sample("latent_posterior", dist.Normal(z_loc, z_scale).independent(1))

        self.debug_qty += 1


    def sim_measurements(self, x, stepQty, params=False):
        '''
        TODO: There is likely a more efficient way of implementing sampling
        Function for generating predictions
        :param x: conditioning variables
        :return: simulated measurements
        '''

        if self.use_cuda:
            x = x.cuda()

        yout = torch.ones(x.size()[0],self.y_dim)
        #print(x)

        for i in range(stepQty):

            # encode image y, conditional on x
            z_loc, z_scale = self.encoder(yout, x)

            # sample in latent space
            z = dist.Normal(z_loc, z_scale).sample()

            # decode the measurements (note we don't sample in measurement space)
            mu, sigma = self.decoder(z,x)

            # generate prediction
            mu = torch.clamp(mu, -1, 1)
            sigma = torch.clamp(sigma, 0, 0.5)

            yout = torch.exp(mu + 0.5 * torch.pow(sigma, 2.0))

            if not check_valid_range(yout):
                print('mu=', mu, 'mu=', sigma, 'yout=', yout)
                raise Exception('Invalid value is generated, aborting!')

        if params is False:
            return yout
        else:
            return mu, sigma


def run_inference_for_epoch(batch_size, data_loaders, loss, use_cuda=False):
    """
    runs the inference algorithm for an epoch
    """

    # compute number of batches for an epoch
    batches_per_epoch = len(data_loaders["train"])

    # initialize variables to store loss values
    epoch_losses = 0.

    # setup the iterators for training data loader
    iterator = iter(data_loaders["train"])

    for i in range(batches_per_epoch):

        (ys, ms, ls, xs) = next(iterator)

        if use_cuda:
            ys = ys.cuda()
            ms = ms.cuda()
            ls = ls.cuda()
            xs = xs.cuda()

        # run the inference
        new_loss = loss.step(ys, ms, ls, xs)

        #print('Loss:', new_loss/batch_size)
        epoch_losses += new_loss

        print(i+1,"/",batches_per_epoch,": ",new_loss)

    # return the values of all losses
    return epoch_losses

def get_accuracy(data_loader, classifier_fn):
    """
    compute the accuracy based on complete observations
    """
    predictions, actuals, lods, masks = [], [], [], []

    # use the appropriate data loader
    for (ys, ms, ls, xs) in data_loader:

        # use classification function to compute all predictions for each batch
        predictions.append(classifier_fn(xs, 10))
        actuals.append(ys)
        lods.append(ls)
        masks.append(ms)

    # compute the number of accurate predictions
    error = 0.0
    tot_preds = 0.0

    for pred, act, lod, mask in zip(predictions, actuals, lods, masks):

        pred = pred.cpu()
        for i in range(pred.size(0)):

            for j in range(pred.size(1)):

                if mask[i,j] == 1:
                    tot_preds += 1
                    error += abs(pred[i,j] - act[i,j])

    # calculate the accuracy between 0 and 1
    errors = (error * 1.0) / tot_preds
    return errors

def get_predictions(data_loader, classifier_fn):
    """
    compute the predictions for the entire dataset
    """
    mu, sigma, actuals, lods, masks = [], [], [], [], []

    # use the appropriate data loader
    for (ys, ms, ls, xs) in data_loader:

        # use classification function to compute all predictions for each batch
        m , s = classifier_fn(xs, 10, params=True)
        mu.append(m)
        sigma.append(s)
        actuals.append(ys)
        masks.append(ms)
        lods.append(ls)

    return mu, sigma, actuals, lods, masks


def main(args):
    """
    run inference for CVAE
    :param args: arguments for CVAE
    :return: None
    """
    if args.seed is not None:
        set_seed(args.seed, args.cuda)

    if os.path.exists('cvae.model.pt'):
        print('Loading model %s' % 'cvae.model.pt')
        cvae = torch.load('cvae.model.pt')

    else:

        cvae = CVAE(z_dim=args.z_dim, y_dim=8, x_dim=32612,
                       hidden_dim=args.hidden_dimension,
                       use_cuda=args.cuda)

    print(cvae)

    # setup the optimizer
    adam_params = {"lr": args.learning_rate,
                   "betas": (args.beta_1, 0.999),
                   "clip_norm":0.5}
    optimizer = ClippedAdam(adam_params)
    guide = config_enumerate(cvae.guide, args.enum_discrete)

    # set up the loss for inference.
    loss = SVI(cvae.model, guide, optimizer, loss=TraceEnum_ELBO(max_iarange_nesting=1))

    try:
        # setup the logger if a filename is provided
        logger = open(args.logfile, "w") if args.logfile else None

        data_loaders = setup_data_loaders(NHANES, args.cuda, args.batch_size)
        print(len(data_loaders['prediction']))

        #torch.save(cvae, 'cvae.model.pt')

        mu, sigma, actuals, lods, masks = get_predictions(data_loaders["prediction"], cvae.sim_measurements)

        torch.save((mu, sigma, actuals, lods, masks), 'cvae.predictions.pt')


    finally:
        # close the logger file object if we opened it earlier
        if args.logfile:
            logger.close()


EXAMPLE_RUN = "python cvae.py --seed 0 --cuda -n 2 -enum parallel -zd 100 -hd 256 -lr 0.000001 -b1 0.95 -bs 5 -log ./tmp.log"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CVAE\n{}".format(EXAMPLE_RUN))

    parser.add_argument('--cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('-n', '--num-epochs', default=50, type=int,
                        help="number of epochs to run")
    parser.add_argument('-enum', '--enum-discrete', default="parallel",
                        help="parallel, sequential or none. uses parallel enumeration by default")
    parser.add_argument('-zd', '--z-dim', default=50, type=int,
                        help="size of the tensor representing the latent variable z " )
    #parser.add_argument('-hl', '--hidden-layers', nargs='+', default=[500], type=int,
    #                    help="a tuple (or list) of MLP layers to be used in the neural networks "
    #                         "representing the parameters of the distributions in our model")
    parser.add_argument('-hd', '--hidden-dimension', default=256, type=int,
                        help="number of units in the MLP layers to be used in the neural networks ")
    parser.add_argument('-lr', '--learning-rate', default=0.00042, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-b1', '--beta-1', default=0.9, type=float,
                        help="beta-1 parameter for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=10, type=int,
                        help="number of examples to be considered in a batch")
    parser.add_argument('-log', '--logfile', default="./tmp.log", type=str,
                        help="filename for logging the outputs")
    parser.add_argument('--seed', default=None, type=int,
                        help="seed for controlling randomness in this example")
    args = parser.parse_args()

    # some assertions to make sure that batching math assumptions are met
    assert NHANES.validation_size % args.batch_size == 0, \
        "batch size should divide the number of validation examples"
    assert NHANES.train_size % args.batch_size == 0, \
        "batch size doesn't divide total number of training data examples"
    assert NHANES.test_size % args.batch_size == 0, "batch size should divide the number of test examples"

    try:
        main(args)

    except:
        # tb is traceback
        exType, value, tb = sys.exc_info()
        print(value)
        print(tb)
        pdb.post_mortem(tb)
