from libcpp.vector cimport vector
import numpy as np

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


def __check_2d_and_reshape(X):
    if len(X.shape) == 1:
        X = np.reshape(X, (-1, X.shape[0]))
    return X


def niw_mmml_mp(args):
    """
    log Normal, Normal-inverse-Wishart mixture model marginal likelihood (multiprocessing wrapper)
    See niw_mmml
    """
    return niw_mmml(*args)


def niw_mmml(X, lambda_0, mu_0, kappa_0, nu_0, crp_alpha, Z_start, k_start, hist_start, ncalc):
    """
    log Normal, Normal-inverse-Wishart mixture model marginal likelihood
    Computes, through enumeration, the log marginal likelihood of the data X under NIW.

    Inputs:
        X (numpy.ndarray<float>): The data, each row is a data point
        lambda (numpy.ndarray<float>): a d by d prior scale matrix
        mu_0 (numpy.ndarray<float>): a d-length prior mean
        kappa_0 (float): prior observations
        nu_0 (float): prior degrees of freedom
        crp_alpha (float): CRP concentration parameter
        Z_start (numpy.ndarray<int>): Starting partiton (assignment of data to components) for the
            calcation. Allows the process to be done in parallel
        k_start (array-like<int>): Starting pseudo counts for the partition generator
        hist_start (numpy.ndarray<float>): Starting histogram for P(Z|alpha) calculations

    Returns:
        float: exact log marginal likelihood
    """
    val = lgniwmmml(X, lambda_0, mu_0, kappa_0, nu_0, crp_alpha, Z_start, k_start, hist_start, ncalc)
    return val


def niw_ml(X, lambda_0, mu_0, kappa_0, nu_0, Z_0):
    """
    log Normal, Normal-inverse-Wishart marginal likelihood

    Inputs:
        X (numpy.ndarray<float>): The data, each row is a data point
        lambda (numpy.ndarray<float>): a d by d prior scale matrix
        mu_0 (numpy.ndarray<float>): a d-length prior mean
        kappa_0 (float): prior observations
        nu_0 (float): prior degrees of freedom
        Z_0 (float): the pre-computed log normalizing constant

    Returns:
        float: log marginal likelihood
    """
    X = __check_2d_and_reshape(X)
    val = lgniwml(X, mu_0, lambda_0, kappa_0, nu_0, Z_0)
    return val


def niw_pp(Y, X, lambda_0, mu_0, kappa_0, nu_0):
    """
    log Normal, Normal-inverse-Wishart posterior predictive probabiliy, P(Y|X)

    Inputs:
        Y (numpy.ndarray<float>): The data to query.
        X (numpy.ndarray<float>): The data to condition on. Each row is a data point.
        lambda (numpy.ndarray<float>): a d by d prior scale matrix
        mu_0 (numpy.ndarray<float>): a d-length prior mean
        kappa_0 (float): prior observations
        nu_0 (float): prior degrees of freedom

    Returns:
        float: log predictive probability
    """
    X = __check_2d_and_reshape(X)
    Y = __check_2d_and_reshape(Y)
    val = lgniwpp(Y, X, mu_0, lambda_0, kappa_0, nu_0)
    return val
