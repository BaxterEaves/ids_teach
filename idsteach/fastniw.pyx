#!python
# cython: boundscheck=False

# IDSTeach: Generate data to teach continuous categorical data.
# Copyright (C) 2015  Baxter S. Eaves Jr.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from libcpp.vector cimport vector
from libcpp cimport bool

from scipy.special import gammaln
from scipy.misc import logsumexp
from math import log

import random
import numpy as np

from idsteach.utils import bell_number as bn
from idsteach.utils import crp_gen
from idsteach.utils import lcrp


cdef extern from "dpgmm.hpp":
    cdef cppclass DPGMM:
        DPGMM(vector[vector[double]] X,
              vector[double] mu_0,
              vector[vector[double]] lambda_0,
              double kappa_0,
              double nu_0,
              double crp_alpha,
              bool single_cluster_init,
              int seed) except +

        double fit(size_t n_iter, 
                 double sm_prop,
                 size_t num_sm_sweeps,
                 size_t sm_burn)

        double seqinit(size_t n_sweeps)

        vector[size_t] predict(vector[vector[double]])

        vector[size_t] get_Z()
        vector[double] get_Nk()
        size_t get_K()

        vector[size_t] get_ks()
        vector[double] get_logps()


cdef extern from "niw_mmml.hpp":
    cdef double lgniwmmml(vector[vector[double]] X,
                     vector[vector[double]] lambda_0,
                     vector[double] mu_0,
                     double kappa_0,
                     double nu_0,
                     double crp_alpha,
                     vector[size_t] Z_start,
                     vector[size_t] k_start,
                     vector[double] hist_start,
                     size_t do_n_calculations)

    cdef double lgniwml(vector[vector[double]] X,
                   vector[double] mu_0,
                   vector[vector[double]] lambda_0,
                   double kappa_0,
                   double nu_0,
                   double Z_0)

    cdef double lgniwpp(vector[vector[double]] Y,
                   vector[vector[double]] X,
                   vector[double] mu_0,
                   vector[vector[double]] lambda_0,
                   double kappa_0,
                   double nu_0)


cdef class PyDPGMM:
    """
    Dirichlet Process Gaussian Mixture Model.
    """
    cdef DPGMM *thisptr;
    def __cinit__(self, data_model, data, crp_alpha=1.0,
                  init_mode='from_prior', seed=20630405):
        self.thisptr = new DPGMM(data, data_model.mu_0, data_model.lambda_0,
                                 data_model.kappa_0, data_model.nu_0,
                                 crp_alpha, init_mode=='single_cluster',
                                 seed=seed)

    def __dealloc__(self):
        del self.thisptr

    def seqinit(self, n_sweeps=0):
        logp = self.thisptr.seqinit(n_sweeps)
        return logp

    def fit(self, n_iter=1, sm_prop=.1, num_sm_sweeps=5, sm_burn=10):
        """
        Runs n_iter Gibbs sweeps on the data and returns the assignment vector.
        FIXME: fill in additional arguments
        """
        _ = self.thisptr.fit(n_iter, sm_prop, num_sm_sweeps, sm_burn)
        return self.thisptr.get_Z()

    def predict(self, data):
        return self.thisptr.predict(data)

    def get_Nk(self):
        Nk = np.array(self.thisptr.get_Nk())
        return Nk

    def get_Z(self):
        Z = np.array(self.thisptr.get_Z(), dtype=int)
        return Z

    def get_K(self):
        return self.thisptr.get_K()

    def get_ks(self):
        ks = np.array(self.thisptr.get_ks(), dtype=float)
        return ks

    def get_logps(self):
        logps = np.array(self.thisptr.get_logps(), dtype=float)
        return logps


def __check_2d_and_reshape(X):
    """
    Enforces that all numpy arrays are 2-D
    """
    if len(X.shape) == 1:
        X = np.reshape(X, (-1, X.shape[0]))
    return X

def brute_estimator(X, data_model, crp_alpha, n_samples, return_logps=False):
    n_data = X.shape[0]
    logps = np.zeros(n_samples)

    lambda_0 = data_model.lambda_0
    mu_0 = data_model.mu_0
    kappa_0 = data_model.kappa_0
    nu_0 = data_model.nu_0
    Z_0 = data_model.log_z

    for i in range(n_samples):
        lp = 0.0
        Z, _, K = crp_gen(n_data, crp_alpha)
        for k in range(K):
            X_k = X[Z==k, :]
            lp += niw_ml(X_k, lambda_0, mu_0, kappa_0, nu_0, Z_0)
        logps[i] = lp

    est = logsumexp(logps) - log(float(n_samples))

    if return_logps:
        return est, logps
    else:
        return est


def pgibbs_estimator(X, data_model, crp_alpha, n_samples=100, 
                    n_sweeps=0, return_logps=False):
    """ 'Partial Gibbs' importance sampling estimator for NIW DPMM marginal
    likelihood.

    Parameters
    ----------
    X : nump.ndarray(n, d)
        Data
    data_model : FIXME
        FIXME
    crp_alpha : float, (0, Inf)
        CRP concentration parameter
    n_samples : int
        number of samples to use for the estimate

    Returns
    -------
    estimator : float
        the Chibb estimator
    """
    n_data = X.shape[0]
    logps = np.zeros(n_samples)
    dpgmm = PyDPGMM(data_model, X, crp_alpha, seed=random.randrange(2**31))

    lambda_0 = data_model.lambda_0
    mu_0 = data_model.mu_0
    kappa_0 = data_model.kappa_0
    nu_0 = data_model.nu_0
    Z_0 = data_model.log_z

    for i in range(n_samples):
        qi = dpgmm.seqinit(n_sweeps)

        Nk = dpgmm.get_Nk()
        K = dpgmm.get_K()
        Z = dpgmm.get_Z()

        logp = lcrp(n_data, Nk, crp_alpha)

        for k in range(K):
            X_k = X[Z==k, :]
            logp += niw_ml(X_k, lambda_0, mu_0, kappa_0, nu_0, Z_0)

        logps[i] = logp - qi

    est= logsumexp(logps) - log(n_samples)

    if return_logps:
        return est, logps
    else:
        return est


def niw_mmml_mp(args):
    """
    log Normal, Normal-inverse-Wishart mixture model marginal likelihood
    (multiprocessing wrapper). See niw_mmml
    """
    return niw_mmml(*args)


def niw_mmml(X, lambda_0, mu_0, kappa_0, nu_0, crp_alpha, Z_start, k_start,
             hist_start, ncalc):
    """
    log Normal, Normal-inverse-Wishart mixture model marginal likelihood.
    Computes, through enumeration, the log marginal likelihood of the data X
    under NIW.

    Parameters
    ----------
    X : numpy.ndarray<float>
        The data, each row is a data point
    lambda : numpy.ndarray<float>
        a d by d prior scale matrix
    mu_0 : numpy.ndarray<float>
        a dims-length prior mean
    kappa_0 : float
        prior observations
    nu_0 : float
        prior degrees of freedom
    crp_alpha : float
        CRP concentration parameter
    Z_start : numpy.ndarray<int>
        Starting partiton (assignment of data to components) for the
        calculation. Allows the process to be done in parallel
    k_start : array-like<int>
        Starting pseudo counts for the partition generator
    hist_start : numpy.ndarray<float>
        Starting histogram for P(Z|alpha) calculations

    Returns
    -------
    ml : float
        exact log marginal likelihood
    """
    ml = lgniwmmml(X, lambda_0, mu_0, kappa_0, nu_0, crp_alpha, Z_start,
                   k_start, hist_start, ncalc)
    return ml


def niw_ml(X, lambda_0, mu_0, kappa_0, nu_0, Z_0):
    """
    log Normal, Normal-inverse-Wishart marginal likelihood

    Parameters
    ----------
    X : numpy.ndarray<float>
        The data, each row is a data point
    lambda : numpy.ndarray<float>
        a d by d prior scale matrix
    mu_0 : numpy.ndarray<float>
        a d-length prior mean
    kappa_0 : float
        prior observations
    nu_0 : float
        prior degrees of freedom
    Z_0 : float
        the pre-computed log normalizing constant

    Returns
    -------
    ml : float
        log marginal likelihood
    """
    X = __check_2d_and_reshape(X)
    ml = lgniwml(X, mu_0, lambda_0, kappa_0, nu_0, Z_0)
    return ml


def niw_pp(Y, X, lambda_0, mu_0, kappa_0, nu_0):
    """
    log Normal, Normal-inverse-Wishart posterior predictive probabiliy, P(Y|X)

    Parameters
    ----------
    Y : numpy.ndarray<float>
        The data to query.
    X : numpy.ndarray<float>
        The data to condition on. Each row is a data point.
    lambda : numpy.ndarray<float>
        a d by d prior scale matrix
    mu_0 : numpy.ndarray<float>
        a d-length prior mean
    kappa_0 : float
        prior observations
    nu_0 : float
        prior degrees of freedom

    Returns
    -------
    pp : float
        log predictive probability
    """
    X = __check_2d_and_reshape(X)
    Y = __check_2d_and_reshape(Y)
    pp = lgniwpp(Y, X, mu_0, lambda_0, kappa_0, nu_0)
    return pp
