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

""" Bayesian heriarchical clustering.

Used to approximate the marginal likelihood of a DPMM.
"""


class Node(object):
    """ A node in the herarchical clustering.

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
            For to calculate marginal likeilhoods
        crp_alpha : float (0, Inf)
            CRP concentration parameter
        log_dk : float
            Cached probabilitiy variable. Do not define if the node is a leaf.
        log_pi : float
            Cached probabilitiy variable. Do not define if the node is a leaf.
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

        return cls(data, data_model, crp_alpha, log_dk, log_pi)


def bhc(data, data_model, crp_alpha=1.0):
    """ Bayesian heirarchical clustering to approximate marginal likelihood of
    some data under CRP micture model. """
    # intialize the tree
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

            # FIXME: this hack
            if n_nodes == 2:
                denom = numer
            else:
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

    # The denomenator of log_rk is at the final merge is an estimate of the
    # marginal likelihood of the data under DPMM
    # FIXME: +2 is an ad-hoc bias correction. Figure out where the bias is
    # coming from.
    return assignments, denom+2


if __name__ == '__main__':
    from idsteach.models import NormalInverseWishart
    from idsteach import fastniw as fniw
    from idsteach.utils import bell_number
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_trials = 500
    n_per_cat = 3

    def gen_data(n_per_cat):
        cov = np.eye(2)*0.2
        X0 = np.random.multivariate_normal([-2.0, 0.0], cov, n_per_cat)
        X1 = np.random.multivariate_normal([2.0, 0.0], cov, n_per_cat)
        X2 = np.random.multivariate_normal([0.0, 1.8], cov, n_per_cat)

        data = np.vstack((X0, X1, X2))
        return data

    hypers = {
        'mu_0': np.zeros(2),
        'nu_0': 3.0,
        'kappa_0': 1.0,
        'lambda_0': np.eye(2)
    }
    data_model = NormalInverseWishart(**hypers)

    p_bhc = []
    p_enum = []
    for _ in range(n_trials):
        data = gen_data(n_per_cat)
        N = n_per_cat * 3
        p_Dk_enum = fniw.niw_mmml(
            data, hypers['lambda_0'], hypers['mu_0'], hypers['kappa_0'],
            hypers['nu_0'], 1.0, [0]*N, [0]*N, [float(N)], bell_number(N))

        asgn, p_Dk_bhc = bhc(data, data_model)
        print("P_enum(Dk) = {}, P_bhc(Dk) = {}".format(p_Dk_enum, p_Dk_bhc))

        p_bhc.append(p_Dk_bhc)
        p_enum.append(p_Dk_enum)

    diag = np.linspace(min(p_enum), max(p_enum), 3)
    plt.figure(tight_layout=True, facecolor='white')

    plt.subplot(1, 2, 1)
    plt.plot(diag, diag, c='red')
    plt.scatter(p_enum, p_bhc, c='dodgerblue', alpha=.7)
    plt.xlabel('Enumeration')
    plt.ylabel('BHC')
    plt.title('DPMM marginal log(P(D))')

    plt.subplot(1, 2, 2)
    plt.hist([a-b for a, b, in zip(p_bhc, p_enum)], 31)
    plt.title('Bias histogram')
    plt.xlabel('log(P(Dk)) bias')
    plt.show()

    # Sanity check: grab the assignment that has three components.
    z = np.array(asgn[-3], dtype=float)
    plt.figure(tight_layout=True, facecolor='white')
    plt.scatter(data[:, 0], data[:, 1], c=z, cmap='Set1', s=225)
    plt.show()
