import numpy as np

from ids_teach import utils

from scipy.special import multigammaln
from scipy.special import gammaln
from numpy.linalg import slogdet
from math import log
from math import pi

from scipy.stats import multivariate_normal as mvnorm

LOG2PI = log(2*pi)
LOG2 = log(2)


# _________________________________________________________________________________________________
# Conjugate models w/ closed form marginal and predictive probabilities
# `````````````````````````````````````````````````````````````````````````````````````````````````
class CollapsibleDistribution(object):
    '''
    Abstract base class for a familiy of conjugate distributions.
    '''
    @staticmethod
    def log_likelihood(X, *likelihood_params):
        '''
        Log of the likelihood P(X|likelihood_params)
        '''
        pass

    def log_prior(self, *likelihood_params):
        '''
        Log of the prior probability
        '''
        pass

    def log_marginal_likelihood(self, X):
        '''
        Log of the marginal likelihood, P(X|prior).
        '''
        pass

    def log_posterior_predictive(self, Y, X):
        '''
        Log of the posterior predictive probability, P(Y|X,prior)
        '''
        pass

    def map_parameters(self, X):
        '''
        Return MAP parametrs given the data, X, and the current prior.
        '''
        pass

    def set_parameters(self, **parameters):
        '''
        Set the prior parameters.
        '''
        pass

    def get_parameters(self):
        '''
        Returns a dict of the parameters.
        '''
        pass

    def draw_data_from_prior(self, n):
        '''
        Drawn n data point from the prior.
        '''
        pass


# _________________________________________________________________________________________________
# Normal w/ Normal, Inverse-Wishart prior TODO: fill out docstring
# `````````````````````````````````````````````````````````````````````````````````````````````````
class NormalInverseWishart(CollapsibleDistribution):
    """
    Mulitvariate Normal likelihood with multivariate Normal prior on mean and Inverse-Wishart prior
    on the covaraince matrix.

    All math taken from Kevin Murphy's 2007 technical report, 'Conjugate Bayesian analysis of the
    Gaussian distribution'.
    """

    def __init__(self, **prior_hyperparameters):
        self.nu_0 = prior_hyperparameters['nu_0']
        self.mu_0 = prior_hyperparameters['mu_0']
        self.kappa_0 = prior_hyperparameters['kappa_0']
        self.lambda_0 = prior_hyperparameters['lambda_0']

        self.d = float(len(self.mu_0))

        self.log_z = self.calc_log_z(self.mu_0, self.lambda_0, self.kappa_0, self.nu_0)

    @classmethod
    def with_vague_prior(cls, target_model):
        """
        Initialize using a set of vague hyperparamers derived from the target model.
        NOTE: This it the exact prior used in the IDS paper.
        """
        mu_0 = np.zeros(target_model['parameters'][0][0].shape)
        lambda_0 = np.zeros(target_model['parameters'][0][1].shape)
        for mu, sigma in target_model['parameters']:
            mu_0 += mu
            lambda_0 += sigma

        mu_0 /= float(len(target_model['parameters']))
        lambda_0 /= float(len(target_model['parameters']))
        kappa_0 = 1.0
        nu_0 = 3.0

        prior_hyperparameters = {
            'mu_0': mu_0,
            'nu_0': nu_0,
            'kappa_0': kappa_0,
            'lambda_0': lambda_0,
        }
        return cls(**prior_hyperparameters)

    @staticmethod
    def log_likelihood(X, *likelihood_params):
        return np.sum(mvnorm.logpdf(X, *likelihood_params))

    def log_prior(self, *likelihood_params):
        mu, sigma = likelihood_params
        logp_mu = self.log_likelihood(mu, self.mu_0, sigma/self.kappa_0)
        logp_sigma = utils.invwish_logpdf(sigma, self.lambda_0, self.nu_0)
        return logp_mu+logp_sigma

    @staticmethod
    def update_parameters(X, _mu, _lambda, _kappa, _nu, _d):
        xbar = np.mean(X, 0)
        n = X.shape[0]
        kappa_n = _kappa + n
        nu_n = _nu + n
        mu_n = (_kappa*_mu + n*xbar)/kappa_n

        S = np.zeros(_lambda.shape) if n == 1 else (n-1)*np.cov(X.T)
        dt = (xbar-_mu)[np.newaxis]

        back = np.dot(dt.T, dt)
        lambda_n = _lambda + S + (_kappa*n/kappa_n)*back

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(lambda_n.shape[0] == _lambda.shape[0])
        assert(lambda_n.shape[1] == _lambda.shape[1])

        return mu_n, lambda_n, kappa_n, nu_n

    @staticmethod
    def calc_log_z(_mu, _lambda, _kappa, _nu):
        d = len(_mu)
        log_z = LOG2*(_nu*d/2.0)  + (d/2.0)*log(2*pi/_kappa) + multigammaln(_nu/2, d) -\
            (_nu/2.0)*slogdet(_lambda)[1]

        return log_z

    def log_marginal_likelihood(self, X):
        n = X.shape[0]
        params_n = self.update_parameters(X, self.mu_0, self.lambda_0, self.kappa_0,
                                          self.nu_0, self.d)
        log_z_n = self.calc_log_z(*params_n)

        return log_z_n - self.log_z - LOG2PI*(n*self.d/2)

    def draw_data_from_prior(self, n):
        return np.random.multivariate_normal(self.mu_0, self.lambda_0/self.kappa_0, n)


# _________________________________________________________________________________________________
# multiprocessing helper functions
# `````````````````````````````````````````````````````````````````````````````````````````````````
def do_log_crp(H, n, crp_alpha):
    """
    Log probability of a partition of n objects given the concentraion parameter, alpha, under CRP.

    Inputs:
        H (numpy.ndarry<float>): H[i] is the number of items assigned to category i
        n (int): The number of datapoints in H
        crp_alpha (float): CRP alpha

    Returns:
        float: logp probability

    Examples:
        >>> do_log_crp(np.array([4,1]), 5, 1.0) # assume Z = [0,0,0,0,1]
        -2.9957322735539909
        >>> do_log_crp(np.array([4,1]), 5, 0.15)
        -3.5811010872527058
        >>> do_log_crp(np.array([4,1]), 5, 12.12)
        -6.4311926735416876
    """
    K = len(H)
    return np.sum(gammaln(H)) + K*log(crp_alpha) + gammaln(crp_alpha) - gammaln(n + crp_alpha)


def _do_log_marginal(model, X, Z):
    """
    Calcualtes the marinal likelihood of the data X, partitioned by Z given model
    """
    indices = np.nonzero(Z == 0)[0]
    logp = model.log_marginal_likelihood(X[indices])
    i = 0

    while True:
        i += 1
        indices = np.nonzero(Z == i)[0]
        if len(indices) == 0:
            break
        logp += model.log_marginal_likelihood(X[indices, :])
        assert(i <= len(Z))

    return logp


def mp_log_marginals(mp_args):
    return do_log_marginals(*mp_args)


def do_log_marginals(model, X, Z, k, h, crp_alpha, num_to_do):
    ret = np.zeros(num_to_do)

    lmp = _do_log_marginal(model, X, Z)
    ret[0] = lmp + do_log_crp(h, X.shape[0], crp_alpha)

    for i in range(1, num_to_do):
        zkh = utils.next_partition(Z, k, h)
        assert zkh is not None
        Z, k, h =  zkh
        lmp = _do_log_marginal(model, X, Z)
        ret[i] = lmp + do_log_crp(h, X.shape[0], crp_alpha)

    return ret

# `````````````````````````````````````````````````````````````````````````````````````````````````
if __name__ == "__main__":
    import doctest
    doctest.testmod()
