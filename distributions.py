import numbers

import numpy as np
from scipy import stats as st
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._continuous_distns import _norm_pdf
from scipy.stats._distn_infrastructure import rv_sample

from utils import equal_ifarray

"""
Notes:
about the probability samplers - we can sample from customely provided map by using
https://emcee.readthedocs.io/en/v2.2.1/
and for the similarity: https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms
http://louistiao.me/notes/calculating-kl-divergence-in-closed-form-versus-monte-carlo-estimation/
https://stackoverflow.com/questions/21100716/fast-arbitrary-distribution-random-sampling
https://stackoverflow.com/questions/49211126/efficiently-sample-from-arbitrary-multivariate-function/53350242#53350242
https://tmramalho.github.io/
https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d
https://dit.readthedocs.io/en/latest/generalinfo.html
https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html
so scipy has nice classes for stats distributions and can draw from distribution defined by cdf.
dit has only discrete distributions
sampling from any map can be done nicely using markov chains (emcee) or simple methods (or by subclassing the scipy
 stats).
 
If we want to calculate the distance of two distributions, jensen shannon distance is the distance we want to use
the computation can be done by scipy function or by own custom monte carlo code
(which might be better, we can provide variable number of samples for the montecarlo).
"""


class gaussian_smoothed_discrete(rv_continuous):
    """
    Just pick a gaussian distribution based on discrete model and then apply gaussian.
    """
    
    def __init__(self, discrete_d, smooth_scale=0.01, **kwargs):
        rv_continuous.__init__(self, **kwargs)
        assert isinstance(discrete_d, rv_sample)
        self.discrete_d = discrete_d
        self.smooth_scale = smooth_scale
    
    def _pdf(self, x, *args):
        accum = np.zeros((len(x)))
        for pos, prob in zip(self.discrete_d.xk, self.discrete_d.pk):
            x_transl = (x - pos) / self.smooth_scale
            accum = np.asarray([acc + prob * _norm_pdf(x_) for acc, x_ in zip(accum, x_transl)])
        # accum /= len(self.discrete_d.xk)
        return accum
    
    def _rvs(self, *args):
        sampled = self.discrete_d.rvs(*args, size=self._size, random_state=self._random_state)
        # here we use the fact, that only locations are changed but sizes are the same and just add them
        return sampled + self.smooth_scale * self._random_state.standard_normal(self._size)


class conditioned_continuous(rv_continuous):
    """
    Just pick a gstandart distribution based on discrete model and then apply it.
    """
    
    def __init__(self, discrete_d, continuous_cases, **kwargs):
        rv_continuous.__init__(self, **kwargs)
        assert isinstance(discrete_d, rv_sample)
        self.discrete_d = discrete_d
        self.continuous_cases = continuous_cases
    
    def _pdf(self, x, *args):
        accum = np.zeros((len(x)))
        for pos, prob in zip(self.discrete_d.xk, self.discrete_d.pk):
            accum += prob * self.continuous_cases[pos].pdf(x)
        return accum
    
    def _rvs(self, *args):
        sampled_d = self.discrete_d.rvs(*args, size=self._size, random_state=self._random_state)
        sampled = np.empty_like(sampled_d, dtype=np.float)
        if len(sampled.shape) > 0:
            for i in np.ndindex(sampled.shape):
                sampled[i] = self.continuous_cases[sampled_d[i]].rvs(1, random_state=self._random_state)
        else:
            sampled = self.continuous_cases[sampled_d].rvs(1, random_state=self._random_state)
        return sampled


def jensen_snannon_divergence_monte_carlo(distribution_p, distribution_q, n_samples=10 ** 5):
    # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
    # all the logarithms are defined as log2 (because of information entrophy)
    X = distribution_p.rvs(size=n_samples)
    p_X = distribution_p.pdf(X)
    q_X = distribution_q.pdf(X)
    log_mix_X = np.log2(p_X + q_X)
    
    Y = distribution_q.rvs(size=n_samples)
    p_Y = distribution_p.pdf(Y)
    q_Y = distribution_q.pdf(Y)
    log_mix_Y = np.log2(p_Y + q_Y)
    
    jsp_m = np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
    jsq_m = np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))
    
    return (jsp_m + jsq_m) / 2


def jensen_snannon_distance_monte_carlo(distribution_p, distribution_q, n_samples=10 ** 5):
    return np.sqrt(jensen_snannon_divergence_monte_carlo(distribution_p, distribution_q, n_samples))


def smoothed_js_distances(distribution_p, distribution_q, n_samples=10 ** 5, smooth_scale=0.01):
    """
    If provided with discrete distributions, smooth them with a factor and then apply jensen shannon distance.
    """
    if isinstance(distribution_q, rv_discrete):
        distribution_q = gaussian_smoothed_discrete(distribution_q, smooth_scale)
    if isinstance(distribution_p, rv_discrete):
        distribution_p = gaussian_smoothed_discrete(distribution_p, smooth_scale)
    return np.sqrt(jensen_snannon_divergence_monte_carlo(distribution_p, distribution_q, n_samples))


class StochasticScorable(object):
    def __init__(self):
        pass
    
    def dim(self):
        return 1
    
    def draw_samples(self, size, random_state=None):
        return [None] * size
    
    def score_samples(self, samples):
        return [None] * len(samples)


class Determined(StochasticScorable):
    """
    Param from constant
    """
    
    def __init__(self, val):
        self.val = val
    
    def draw_samples(self, size, random_state=None):
        return np.full(size, self.val)
    
    def score_samples(self, samples):
        return np.asarray([1.0 if equal_ifarray(sample, self.val) else 0.0 for sample in samples])
    
    def __str__(self):
        return str(self.val)


def clipnorm(val_min, val_max, loc=0.0, scale=1.0):
    # https://stackoverflow.com/questions/41316068/truncated-normal-distribution-with-scipy-in-python
    a, b = (val_min - loc) / scale, (val_max - loc) / scale
    return st.truncnorm(a=a, b=b, loc=loc, scale=scale)


class StochasticScorableWrapper(StochasticScorable):
    """
    Param from rv_continuous / rv_discrete

    Example:
        norm2d = StochasticScorableWrapper(st.norm(loc=0, scale=1))


    """
    
    def __init__(self, rv_object):
        StochasticScorable.__init__(self)
        self.dist = rv_object
    
    def draw_samples(self, size, random_state=None):
        return self.dist.rvs(size=size, random_state=random_state)
    
    def score_samples(self, x):
        if hasattr(self.dist, 'pdf'):
            return self.dist.pdf(x)
        else:
            return self.dist.ppf(x)


class FixdimDistribution(StochasticScorableWrapper):
    """
    Param from rv_continuous / rv_discrete with dimension info

    Example:
        norm2d = FixdimDistribution(st.norm(loc=[0, 1], scale=[1, 1]), 2)
    """
    
    def __init__(self, rv_object, item_dimension):
        StochasticScorableWrapper.__init__(self, rv_object)
        self._dim = item_dimension
    
    def dim(self):
        return self._dim
    
    def draw_samples(self, size, random_state=None):
        if isinstance(size, tuple):
            size = list(size)
        if not isinstance(size, list):
            size = [size]
        if self.dim() > 0:
            size = size + [self.dim()]
        return self.dist.rvs(size=size, random_state=random_state)


def _squeeze_last(array):
    if array.shape[-1] == 1 and len(array.shape) > 1:
        return np.squeeze(array, axis=-1)
    return array


class MultiDistribution(StochasticScorable):
    """
    Param created from more FixdimDistributions that are independent.

    Example:
        norm2d = MultiDistribution([StochasticScorableWrapper(st.norm(loc=0, scale=1)),
         StochasticScorableWrapper(st.norm(loc=0, scale=1))])

    """
    
    def __init__(self, fixdim_distributions):
        StochasticScorable.__init__(self)
        for dist in fixdim_distributions:
            assert isinstance(dist, StochasticScorable)
            assert dist.dim() == 1
            # so far only works for stacking onedimensional things. To change to variable number of dims,
            # change squeeze&stack
            # into some concatenation and checking if the last dimension returned is ALWAYS the dist.dim()
            # the edgecase is dim() == 1 & returns shape (5,1) vs (5,) (impicit shape 1)
        self.dists = fixdim_distributions
    
    def dim(self):
        return sum([dist.dim() for dist in self.dists])
    
    """
    @classmethod
    def _expand_lastcheck(cls, array, dim):
        if array.shape[-1] == dim:
            return array
        else:
            assert dim==1, "if the last dimension of the array is not equal to the provided dimension and not one,
             that means that there is a mistake in the generator that gave us this result"
            return np.expand_dims(array)

    def draw_samples(self, size, random_state=None):
        # for stacking drawn onedimensional things ontop of each other we need to squeeze all dimensions ending on one
        samples = [self._expand_lastcheck(dist.draw_samples(size, random_state=random_state), dist.dim())
                   for dist in self.dists]
        return np.concatenate(samples, axis=-1)
    """
    
    def draw_samples(self, size, random_state=None):
        # for stacking drawn onedimensional things ontop of each other we need to squeeze all dimensions ending on one
        samples = [_squeeze_last(dist.draw_samples(size, random_state=random_state))
                   for dist in self.dists]
        return np.stack(samples, axis=-1)
    
    def score_samples(self, x):
        pos = 0
        scores = np.ones(x.shape[0])
        for dist in self.dists:
            scores *= dist.score_samples(x[:, pos:pos + dist.dim()])
            pos += dist.dim()
        return scores


class DistributionAsRadial(StochasticScorable):
    """
    r, phi
    """
    
    def __init__(self, inp):
        assert inp.dim() == 2
        self.inp = inp
    
    def dim(self):
        return 2
    
    def draw_samples(self, size, random_state=None):
        drawn = self.inp.draw_samples(size, random_state)
        assert drawn.shape[-1] == 2, "the last dimension must be two for plaanar coordinates"
        r = drawn[..., 0:1]
        phi = drawn[..., 1:2]
        x = r * np.sin(phi)
        y = r * np.cos(phi)
        return np.concatenate([x, y], axis=-1)
    
    def score_samples(self, samples):
        x = samples[..., 0:1]
        y = samples[..., 1:2]
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        orig = np.concatenate([r, phi], axis=-1)
        return self.inp.score_samples(orig)


def is_np_array(val):
    """
    Checks whether a variable is a numpy array.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a numpy array. Otherwise False.

    """
    # using np.generic here seems to also fire for scalar numpy values even
    # though those are not arrays
    # return isinstance(val, (np.ndarray, np.generic))
    return isinstance(val, np.ndarray)


def is_single_integer(val):
    """
    Checks whether a variable is an integer.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is an integer. Otherwise False.

    """
    return isinstance(val, numbers.Integral)


def is_single_float(val):
    """
    Checks whether a variable is a float.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a float. Otherwise False.

    """
    return isinstance(val, numbers.Real) and not is_single_integer(val)


def is_single_number(val):
    """
    Checks whether a variable is a number, i.e. an integer or float.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a number. Otherwise False.

    """
    return is_single_integer(val) or is_single_float(val)


def is_iterable(val):
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is an iterable. Otherwise False.

    """
    # TODO make this more abstract, not just restricted
    return isinstance(val, (tuple, list, np.ndarray))


def single_input_as_bool_param_s(p):
    if isinstance(p, bool):
        return Determined(1.0 if p else 0.0)
    # todo:
    # elif is_single_float(p) or is_single_integer(p):
    #    assert 0 <= p <= 1
    #    return Binomial(p)
    elif isinstance(p, StochasticScorable):
        return p
    else:
        raise Exception("Expected bool or float/int in range [0, 1] or StochasticScorable as p, got %s." % (type(p),))


def single_number_param_s(inp):
    if isinstance(inp, StochasticScorable):
        return inp
    elif is_single_number(inp):
        return Determined(inp)
    else:
        raise Exception("Expected float, int or parameter class, got %s" % (type(inp),))


def tuple_param_s(inp, assert_dims=None):
    if isinstance(inp, StochasticScorable):
        if assert_dims is not None and inp.dim() != assert_dims:
            raise Exception("Dimensions of input param mismatch")
        return inp  # todo hope it is 2dimensional
    elif is_iterable(inp):
        ret = [single_number_param_s(item) for item in inp]
        if assert_dims is not None:
            expected_dims = sum([item.dim() for item in ret])
            if expected_dims != assert_dims:
                raise Exception("Expected the input to be %s dimensions, got %s" % (assert_dims, expected_dims))
        return MultiDistribution(ret)
    elif is_single_number(inp) and assert_dims == 1 or assert_dims == None:
        return single_number_param_s(inp)
    else:
        raise Exception("Expected tuple of (float, int or Parametrized class), got %s, %s." % (type(inp),))
