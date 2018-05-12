import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.distributions import Normal, LogNormal
from torch.distributions.transforms import ExpTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from pyro.distributions.torch_distribution import TorchDistributionMixin

class NormalCens(Normal):
    def __init__(self, loc, scale, censMsk):
        super().__init__(loc, scale)
        self.censMsk = censMsk

    def log_prob(self, value):

        if self._validate_args:
            self._validate_sample(value)

        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()

        density = -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

        #cumul = torch.log(0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2))))

        #Approximating normal cdf
        cumul = -torch.log(1 + torch.exp( -1.65451 * (value - self.loc) * self.scale.reciprocal()))

        #print(cumul)

        res = self.censMsk * density + (1 - self.censMsk) * cumul

        return res

    def cdf(self, value):
        assert(False)

    def icdf(self, value):
        assert(False)

    def entropy(self):
        assert (False)

    @property
    def _natural_params(self):
        assert (False)

    def _log_normalizer(self, x, y):
        assert (False)


class LogNormalCens(TransformedDistribution):
    r"""
    Creates a log-normal distribution parameterized by
    `loc` and `scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log ofthe distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, censMsk, validate_args=None):
        super(LogNormalCens, self).__init__(NormalCens(loc, scale, censMsk), ExpTransform(), validate_args=validate_args)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def variance(self):
        return (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()

    def entropy(self):
        return self.base_dist.entropy() + self.loc


class LogNormCensPyro(LogNormalCens, TorchDistributionMixin):
    pass
