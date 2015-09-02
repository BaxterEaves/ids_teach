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

import itertools as it
import numpy as np

from numpy import logaddexp
from math import log
from math import exp
from math import expm1
from math import lgamma

""" Bayesian hierarchical clustering.

Used to approximate the marginal likelihood of a DPMM.
"""


class Node(object):
    """ A node in the hierarchical clustering.

    Attributes
    ----------
    nk : int
        Number of data points assigned to the node
    data : numpy.ndarrary (n, d)
        The data assigned to the Node. Each row is a datum.
    crp_alpha : float
        CRP concentration parameter
    log_dk : float
        Some kind of number for computing probabilities
    log_pi : float
        For to compute merge probability
    """

    def __init__(self, data, data_model, crp_alpha=1.0, log_dk=None,
                 log_pi=0.0):
        """
        Parameters
        ----------
        data : numpy.ndarray
            Array of data_model-appropriate data
        data_model : idsteach.CollapsibleDistribution
            For to calculate marginal likelihoods
        crp_alpha : float (0, Inf)
            CRP concentration parameter
        log_dk : float
            Cached probability variable. Do not define if the node is a leaf.
        log_pi : float
            Cached probability variable. Do not define if the node is a leaf.
        """
        self.data_model = data_model
        self.data = data
        self.nk = data.shape[0]
        self.crp_alpha = crp_alpha
        self.log_pi = log_pi

        if log_dk is None:
            self.log_dk = log(crp_alpha)
        else:
            self.log_dk = log_dk

        self.logp = self.data_model.log_marginal_likelihood(self.data)

    @classmethod
    def as_merge(cls, node_left, node_right):
        """ Create a node from two other nodes

        Parameters
        ----------
        node_left : Node
            the Node on the left
        node_right : Node
            The Node on the right
        """
        crp_alpha = node_left.crp_alpha
        data_model = node_left.data_model
        data = np.vstack((node_left.data, node_right.data))

        nk = data.shape[0]
        log_dk = logaddexp(log(crp_alpha) + lgamma(nk),
                           node_left.log_dk + node_right.log_dk)
        log_pi = log(crp_alpha) + lgamma(nk) - log_dk

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return cls(data, data_model, crp_alpha, log_dk, log_pi)


def bhc(data, data_model, crp_alpha=1.0):
    """ Bayesian hierarchical clustering to approximate marginal likelihood of
    some data under CRP mixture model. """
    # initialize the tree
    nodes = dict((i, Node(np.array([x]), data_model, crp_alpha))
                 for i, x in enumerate(data))
    n_nodes = len(nodes)
    assignment = [i for i in range(n_nodes)]
    assignments = [list(assignment)]
    rks = [0]

    while n_nodes > 1:
        max_rk = float('-Inf')
        merged_node = None
        for left_idx, right_idx in it.combinations(nodes.keys(), 2):
            tmp_node = Node.as_merge(nodes[left_idx], nodes[right_idx])

            logp_left = nodes[left_idx].logp
            logp_right = nodes[right_idx].logp
            logp_comb = tmp_node.logp

            log_pi = tmp_node.log_pi

            numer = log_pi + logp_comb

            neg_pi = log(-expm1(log_pi))
            denom = logaddexp(numer, neg_pi+logp_left+logp_right)

            log_rk = numer-denom

            if log_rk > max_rk:
                max_rk = log_rk
                merged_node = tmp_node
                merged_right = right_idx
                merged_left = left_idx

        rks.append(exp(max_rk))

        del nodes[merged_right]
        nodes[merged_left] = merged_node

        for i, k in enumerate(assignment):
            if k == merged_right:
                assignment[i] = merged_left
        assignments.append(list(assignment))

        n_nodes -= 1

    # The denominator of log_rk is at the final merge is an estimate of the
    # marginal likelihood of the data under DPMM
    return assignments, denom


if __name__ == '__main__':
    from idsteach.models import NormalInverseWishart
    from idsteach import fastniw as fniw
    from idsteach.utils import bell_number
    # from scipy.stats import linregress
    import matplotlib.pyplot as plt

    import random

    n_trials = 500

    nlst = [(i, 1) for i in range(2, 11)]
    nlst += [(i, i) for i in range(2, 11)]
    nlst += [(10, 5), (10, 2)]
    nlst += [(9, 3)]
    nlst += [(8, 4), (8, 2)]
    nlst += [(6, 6), (6, 3), (6, 2)]
    nlst += [(4, 4), (4, 2)]

    def gen_data(n_per_cat, n_cats=3):
        cov = np.eye(2)*0.2
        X = np.random.multivariate_normal([0., 0.], cov, n_per_cat)
        for _ in range(1, n_cats):
            mu = np.random.multivariate_normal([0., 0.], np.eye(2)*5.)
            Xi = np.random.multivariate_normal(mu, cov, n_per_cat)
            X = np.vstack((X, Xi))

        return X

    hypers = {
        'mu_0': np.zeros(2),
        'nu_0': 3.0,
        'kappa_0': 1.0,
        'lambda_0': np.eye(2)
    }
    data_model = NormalInverseWishart(**hypers)

    p_bhc = np.zeros(n_trials)
    p_chib = np.zeros(n_trials)
    p_enum = np.zeros(n_trials)

    n_per_cats_list = np.zeros(n_trials, dtype=int)
    n_cats_list = np.zeros(n_trials, dtype=int)
    n_list = np.zeros(n_trials, dtype=int)
    for t in range(n_trials):
        N, n_per_cat = random.choice(nlst)
        n_cats = int(int(N+.5)/int(n_per_cat+.5))

        n_per_cats_list[t] = n_per_cat
        n_cats_list[t] = n_cats
        n_list[t] = N

        data = gen_data(n_per_cat, n_cats)
        p_Dk_enum = fniw.niw_mmml(
            data, hypers['lambda_0'], hypers['mu_0'], hypers['kappa_0'],
            hypers['nu_0'], 1.0, [0]*N, [0]*N, [float(N)], bell_number(N))

        p_Dk_chib = fniw.pgibbs_estimator(data, data_model, 1., n_samples=1000,
                                          n_sweeps=0)
        asgn, p_Dk_bhc = bhc(data, data_model)
        print("N: {}, K: {}".format(N, n_cats))
        print("exp(pgibbs - enum): %f" % (exp(p_Dk_chib - p_Dk_enum),))
        print("1/exp(pgibbs - enum): %f" % (1./exp(p_Dk_chib - p_Dk_enum),))
        print("P_enum(Dk) = {}, P_bhc(Dk) = {}, P_gibbs(Dk) = {}".format(
            p_Dk_enum, p_Dk_bhc, p_Dk_chib))

        p_bhc[t] = p_Dk_bhc
        p_chib[t] = p_Dk_chib
        p_enum[t] = p_Dk_enum

    diag = np.linspace(min(p_enum), max(p_enum), 3)
    plt.figure(tight_layout=True, facecolor='white')

    plt.subplot(1, 3, 1)
    plt.plot(diag, diag, c='red')
    plt.scatter(p_enum, p_bhc, c=n_cats_list, alpha=.7, cmap='Set1')
    plt.xlabel('Enumeration')
    plt.ylabel('BHC')
    plt.title('DPMM marginal log(P(D)), by # categories')

    plt.subplot(1, 3, 2)
    plt.plot(diag, diag, c='red')
    plt.scatter(p_enum, p_bhc, c=n_list, alpha=.7, cmap='Set1')
    plt.xlabel('Enumeration')
    plt.ylabel('BHC')
    plt.title('DPMM marginal log(P(D)), by N')

    plt.subplot(1, 3, 3)
    plt.hist([a-b for a, b, in zip(p_bhc, p_enum)], 31)
    plt.title('Bias histogram (BHC)')
    plt.xlabel('log(P(Dk)) bias')

    plt.show()

    plt.subplot(1, 3, 1)
    plt.plot(diag, diag, c='red')
    plt.scatter(p_enum, p_chib, c=n_cats_list, alpha=.7, cmap='Set1')
    plt.xlabel('Enumeration')
    plt.ylabel('BHC')
    plt.title('DPMM marginal log(P(D)), by # categories')

    plt.subplot(1, 3, 2)
    plt.plot(diag, diag, c='red')
    plt.scatter(p_enum, p_chib, c=n_list, alpha=.7, cmap='Set1')
    plt.xlabel('Enumeration')
    plt.ylabel('PGibbs')
    plt.title('DPMM marginal log(P(D)), by N')

    plt.subplot(1, 3, 3)
    plt.hist([a-b for a, b, in zip(p_chib, p_enum)], 31)
    plt.title('Bias histogram (PGibbs)')
    plt.xlabel('log(P(Dk)) bias')

    plt.show()

    # -------------------------------------------------------------------------
    # plt.figure(tight_layout=True, facecolor='white')
    # for i, n in enumerate(range(2, 11)):
    #     p_enum_n = p_enum[n_list == n]
    #     p_chib_n = p_chib[n_list == n]
    #     n_cats_n = n_cats_list[n_list == n]
    #     diag = np.linspace(min(p_enum_n), max(p_enum_n), 3)

    #     res = linregress(p_enum_n, p_chib_n)
    #     slope = res[0]

    #     plt.subplot(2, 9, i+1)
    #     plt.plot(diag, diag, c='red')
    #     plt.scatter(p_enum_n, p_chib_n, c=n_cats_n, alpha=.7, cmap='Set1')
    #     plt.xlabel('Enumeration')
    #     plt.ylabel('PGIBBS')
    #     plt.title(' N: {}, m: {}'.format(n, slope))

    #     plt.subplot(2, 9, i+10)
    #     plt.hist([a-b for a, b, in zip(p_chib_n, p_enum_n)], 31)
    #     plt.title('Bias (PGIBBS)')
    #     plt.xlabel('log(P(Dk)) bias')

    # plt.show()

    # # Sanity check: grab the assignment that has three components.
    # data = gen_data(15, 3)
    # asgn, _ = bhc(data, data_model)
    # z = np.array(asgn[-3], dtype=float)
    # plt.figure(tight_layout=True, facecolor='white')
    # plt.scatter(data[:, 0], data[:, 1], c=z, cmap='Set1', s=225)
    # plt.show()
