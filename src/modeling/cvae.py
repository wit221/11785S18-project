import argparse

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log, set_seed
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from cvae_data_loaders import NHANES, mkdir_p, setup_data_loaders


def my_initializer(m):
    """
    Simple initializer
    """
    if hasattr(m, 'weight'):
        torch.nn.init.xavier_uniform_(m.weight.data)
    if hasattr(m, 'bias'):
        m.bias.data.zero_()

class Encoder(nn.Module):
    '''
    PyTorch module that parameterizes the
    diagonal gaussian distribution q(z|y,x)
    '''
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()

        # setup the three linear transformations
        self.fcy = nn.Linear(1, hidden_dim)
        self.fcx = nn.Linear(187, hidden_dim)
        self.fc21 = nn.Linear(2*hidden_dim, z_dim)
        self.fc22 = nn.Linear(2*hidden_dim, z_dim)

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

        #print(x.size())
        #print(y.size())

        # TODO: determine if this is needed
        # first shape the mini-batch to have data in the rightmost dimension
        y = y.reshape(-1, 1)
        x = x.reshape(-1, 187)

        # compute the hidden units
        hiddenx = self.softplus(self.fcx(x))
        hiddeny = self.softplus(self.fcy(y))
        hidden = torch.cat((hiddeny, hiddenx), 1)

        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))

        return z_loc, z_scale


class Decoder(nn.Module):
    '''
    PyTorch module that parameterized the observation
    likelihood p(y|z,x)
    '''
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()

        # setup the two linear transformations used
        self.fcx = nn.Linear(187, hidden_dim)
        self.fcz = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(2*hidden_dim, 1)

        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.apply(my_initializer)

    def forward(self, z, x):
        '''
        Forward computation on latent variables z and conditioning variables x
        :param z: latent variables
        :param x: conditioning variables
        :return: parameters of the measurement distribution
        '''

        hiddenz = self.softplus(self.fcz(z))
        hiddenx = self.softplus(self.fcx(x))
        hidden = torch.cat((hiddenz, hiddenx), 1)

        # return the parameter for the output Bernoulli
        # each is of size batch_size x 1
        loc_measurements = self.sigmoid(self.fc21(hidden))

        return loc_measurements


class CVAE(nn.Module):
    '''
    PyTorch module for conditional VAE
    '''

    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(CVAE, self).__init__()

        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, y, x):
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
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))

            # decode the latent code z
            loc_measurements = self.decoder.forward(z, x)

            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_measurements).independent(1), obs=y.reshape(-1, 1))

            # return the loc so we can visualize it later
            return loc_measurements

    def guide(self, y, x):
        '''
        Define the guide (i.e. variational distribution), q(z| y, x)
        :param y: measurements
        :param x: conditioning variables
        :return: None
        '''

        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)

        with pyro.iarange("data", x.size(0)):

            # use the encoder to get the parameters used to define q(z|y,x)
            z_loc, z_scale = self.encoder.forward(y, x)

            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))

    def sim_measurements(self, x):
        '''
        TODO: There is likely a more efficient way of implementing sampling
        Function for generating predictions
        :param x: conditioning variables
        :return: simulated measurements
        '''

        yout = torch.zeros(x.size()[0],1)
        #print(x)

        for i in range(10):

            # encode image y, conditional on x
            z_loc, z_scale = self.encoder(yout,x)

            # sample in latent space
            z = dist.Normal(z_loc, z_scale).sample()

            # decode the measurements (note we don't sample in measurement space)
            loc_measurement = self.decoder(z,x)

            # generate prediction
            yout = torch.round(loc_measurement)

        return yout


def run_inference_for_epoch(data_loaders, loss):
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

        (ys, xs) = next(iterator)

        # run the inference
        new_loss = loss.step(ys, xs)
        epoch_losses += new_loss

    # return the values of all losses
    return epoch_losses

def get_accuracy(data_loader, classifier_fn, batch_size):
    """
    compute the accuracy based on complete observations
    """
    predictions, actuals = [], []

    # use the appropriate data loader
    for (ys, xs) in data_loader:

        # use classification function to compute all predictions for each batch
        predictions.append(classifier_fn(xs))
        actuals.append(ys)

    # compute the number of accurate predictions
    accurate_preds = 0
    for pred, act in zip(predictions, actuals):
        for i in range(pred.size(0)):
            v = torch.sum(pred[i] == act[i])
            #print(pred[i])
            #print(act[i])
            #print(v)
            accurate_preds += (v.item() == 1)

    # calculate the accuracy between 0 and 1
    accuracy = (accurate_preds * 1.0) / (len(predictions) * batch_size)
    return accuracy

def main(args):
    """
    run inference for CVAE
    :param args: arguments for CVAE
    :return: None
    """
    if args.seed is not None:
        set_seed(args.seed, args.cuda)

    # batch_size: number of images (and labels) to be considered in a batch
    cvae = CVAE(z_dim=args.z_dim,
                hidden_dim=args.hidden_dimension,
                   use_cuda=args.cuda)

    # setup the optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999)}
    optimizer = ClippedAdam(adam_params)

    # set up the loss for inference.
    guide = config_enumerate(cvae.guide, args.enum_discrete)
    loss = SVI(cvae.model, guide, optimizer, loss=TraceEnum_ELBO(max_iarange_nesting=1))


    try:
        # setup the logger if a filename is provided
        logger = open(args.logfile, "w") if args.logfile else None

        data_loaders = setup_data_loaders(NHANES, args.cuda, args.batch_size)

        # initializing local variables to maintain the best validation acc
        # seen across epochs over the supervised training set
        # and the corresponding testing set and the state of the networks
        best_valid_acc, corresponding_test_acc = 0.0, 0.0

        # run inference for a certain number of epochs
        for i in range(0, args.num_epochs):

            # get the losses for an epoch
            epoch_losses = \
                run_inference_for_epoch(data_loaders, loss)

            # compute average epoch losses i.e. losses per example
            avg_epoch_losses = epoch_losses / NHANES.train_data_size

            # store the losses in the logfile
            str_loss = str(avg_epoch_losses)

            str_print = "{} epoch: avg loss {}".format(i, "{}".format(str_loss))

            validation_accuracy = get_accuracy(data_loaders["valid"], cvae.sim_measurements, args.batch_size)
            str_print += " validation accuracy {}".format(validation_accuracy)

            # this test accuracy is only for logging, this is not used
            # to make any decisions during training
            test_accuracy = get_accuracy(data_loaders["test"], cvae.sim_measurements, args.batch_size)
            str_print += " test accuracy {}".format(test_accuracy)

            # update the best validation accuracy and the corresponding
            # testing accuracy and the state of the parent module (including the networks)
            if best_valid_acc < validation_accuracy:
                best_valid_acc = validation_accuracy
                corresponding_test_acc = test_accuracy

            print_and_log(logger, str_print)

        final_test_accuracy = get_accuracy(data_loaders["test"], cvae.sim_measurements, args.batch_size)

        print_and_log(logger, "best validation accuracy {} corresponding testing accuracy {} "
                              "last testing accuracy {}".format(best_valid_acc, corresponding_test_acc,
                                                                final_test_accuracy))

    finally:
        # close the logger file object if we opened it earlier
        if args.logfile:
            logger.close()


EXAMPLE_RUN = "python cvae.py --seed 0 --cuda -n 2 -enum parallel -zd 50 -hd 500 -lr 0.00042 -b1 0.95 -bs 200 -log ./tmp.log"

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
    assert NHANES.train_data_size % args.batch_size == 0, \
        "batch size doesn't divide total number of training data examples"
    assert NHANES.test_size % args.batch_size == 0, "batch size should divide the number of test examples"

    main(args)
