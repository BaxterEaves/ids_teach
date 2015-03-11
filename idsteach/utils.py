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


import os
import math
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns  # never called explictly, but changes pyplot settings

from scipy.misc import logsumexp
from scipy.special import multigammaln
from scipy.special import gammaln
from numpy.linalg import slogdet
from numpy.linalg import solve
from random import shuffle
from scipy import linalg
from numpy import trace
from math import log

ITERATIONS = 1000


def dist(a, b):
    """
    Euclidian distance between two numpy arrays

    Example:
        >>> import numpy as np
        >>> a = np.array([1.2, -3.4])
        >>> b = np.array([0.0, 6.8])
        >>> dist(a, b)
        10.270345661174213
    """
    return np.sum((a-b)**2.0)**.5


def invwish_logpdf(X, S, df):
    """
    Calculate the log of the Inverse-Wishart pdf

    Inputs:
        X (numpy.ndarray): Square array
        S (numpy.ndarray): Square array, scale matrix parameter
        df (float): Degrees of freedom

    Example:
        >>> import numpy as np
        >>> X = np.array([[0.275540735784, -0.01728072206], [-0.01728072206, 0.17214874805]])
        >>> invwish_logpdf(X, np.eye(2, 2), 2)
        0.35682871083019663
        >>> invwish_logpdf(X, X, 3.436)
        0.81037760291240524
    """
    d = X.shape[0]
    if df < d:
        raise ValueError('df must be greater than or equal to the number of dimensions of S')
    if d != X.shape[1]:
        raise ValueError('X must be square.')
    if S.shape[0] != d or S.shape[1] != d:
        raise ValueError('S must be the same shape as X.')

    _, logdet_S = slogdet(S)
    _, logdet_X = slogdet(X)

    logpdf = (df/2)*logdet_S - ((df*d/2)*log(2) + multigammaln(df/2, d))
    logpdf += (-(d+df+1)/2)*logdet_X - (1/2)*trace(solve(X.T, S.T))

    return logpdf


def hz_to_erb(f):
    """
    Converts a numpy array, f, full of data measured in Hz to ERB according to
    B.C.J. Moore and B.R. Glasberg, "Suggested formulae for calculating auditory-filter bandwidths
        and excitation patterns" Journal of the Acoustical Society of America 74: 750-753, 1983.
    http://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth
    """
    return 6.23*(f**2) + 93.9*f + 28.52


def jitter_data(X, std):
    """
    Add random Gaussian noise (with standard deviation std) to X.

    Example:
        >>> import numpy as np
        >>> np.random.seed(1701)
        >>> X = np.zeros((4, 2))
        >>> X
        array([[ 0.,  0.],
               [ 0.,  0.],
               [ 0.,  0.],
               [ 0.,  0.]])
        >>> jitter_data(X, 1.0)
        array([[-2.45673077,  2.37595165],
               [-0.9699642 , -0.50216492],
               [-1.08208145, -1.5184966 ],
               [-0.70900708,  0.42912647]])

    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    return X + np.random.randn(*X.shape)*std


def what_do_you_think_about_the_stars_in_the_sky():
    that_was_an_interesting_response = "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n\
        \r░░░░▄██████████████████████▄░░░░\n░░░░█░░░░░░░░░░░░░░░░░░░░░░█░░░░\n\
        \r░░░░█░▄██████████████████▄░█░░░░\n░░░░█░█░░░░░░░░░░░░░░░░░░█░█░░░░\n\
        \r░░░░█░█░░░░░░░░░░░░░░░░░░█░█░░░░\n░░░░█░█░░█░░░░░░░░░░░░█░░█░█░░░░\n\
        \r░░░░█░█░░░░░▄▄▄▄▄▄▄▄░░░░░█░█░░░░\n░░░░█░█░░░░░▀▄░░░░▄▀░░░░░█░█░░░░\n\
        \r░░░░█░█░░░░░░░▀▀▀▀░░░░░░░█░█░░░░\n░░░░█░█░░░░░░░░░░░░░░░░░░█░█░░░░\n\
        \r░█▌░█░▀██████████████████▀░█░▐█░\n░█░░█░░░░░░░░░░░░░░░░░░░░░░█░░█░\n\
        \r░█░░█░████████████░░░░░██░░█░░█░\n░█░░█░░░░░░░░░░░░░░░░░░░░░░█░░█░\n\
        \r░█░░█░░░░░░░░░░░░░░░▄░░░░░░█░░█░\n░▀█▄█░░░▐█▌░░░░░░░▄███▄░██░█▄█▀░\n\
        \r░░░▀█░░█████░░░░░░░░░░░░░░░█▀░░░\n░░░░█░░░▐█▌░░░░░░░░░▄██▄░░░█░░░░\n\
        \r░░░░█░░░░░░░░░░░░░░▐████▌░░█░░░░\n░░░░█░▄▄▄░▄▄▄░░░░░░░▀██▀░░░█░░░░\n\
        \r░░░░█░░░░░░░░░░░░░░░░░░░░░░█░░░░\n░░░░▀██████████████████████▀░░░░\n\
        \r░░░░░░░░██░░░░░░░░░░░░██░░░░░░░░\n░░░░░░░░██░░░░░░░░░░░░██░░░░░░░░\n\
        \r░░░░░░░░██░░░░░░░░░░░░██░░░░░░░░\n░░░░░░░░██░░░░░░░░░░░░██░░░░░░░░\n\
        \r░░░░░░░▐██░░░░░░░░░░░░██▌░░░░░░░\n░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░"
    print(that_was_an_interesting_response)


def flatten_niw_model(target_model, data_model):
    """
    remove f3 components
    """
    from idsteach.models import NormalInverseWishart

    target_model['d'] = 2
    for i, params in enumerate(target_model['parameters']):
        params = list(params)
        params[0] = np.copy(params[0][:2])
        params[1] = np.copy(params[1][:2, :2])
        target_model['parameters'][i] = tuple(params)

    mu_0 = np.copy(data_model.mu_0[:2])
    lambda_0 = np.copy(data_model.lambda_0[:2, :2])

    data_model = NormalInverseWishart(**dict(mu_0=mu_0, lambda_0=lambda_0, kappa_0=1., nu_0=3.))

    return target_model, data_model


def matlab_csv_to_teacher_data(dirname):
    """
    Utility to convert data from the old matlab code to something the new hotness (this code) can
    use.
    """
    samples = np.genfromtxt(os.path.join(dirname, 'samples.csv'), dtype=float, delimiter=",")
    labels = np.genfromtxt(os.path.join(dirname, 'labels.csv'), dtype=int, delimiter=",")
    data = [None]*max(labels)  # matlab is 1-indexed, so no need to add 1
    for i, z in enumerate(labels):
        if data[z-1] is None:
            data[z-1] = np.copy(samples[i, :])
        else:
            data[z-1] = np.vstack((data[z-1], np.copy(samples[i, :])))
    return data


def multiple_pandas_to_teacher_data(dirname, remove_f3=False):
    basedir = os.path.join(dirname, '3d')
    full_data = []
    data = []
    num_runs = 10
    for d in range(1, num_runs+1):
        filename = os.path.join(basedir, '%i'%d, 'data_f3_full_%i.pkl' % (d,))
        full_data = pickle.load(open(filename, 'rb'))
        if remove_f3:
            full_data = [phoneme_data[:,:2] for phoneme_data in full_data]
        if d == 1:
            data = [phoneme_data for phoneme_data in full_data]
        else:
            for i, phoneme_data in enumerate(full_data):
                data[i] = np.vstack((data[i], phoneme_data))
    return data


def multiple_matlab_csv_to_teacher_data(short_runs_dirname):
    """
    Utility to convert data from the old matlab code to something the new hotness (this code) can
    use.
    """
    subdirname = 'Run-'
    data = None
    data_length = 0
    for i in range(10):
        dirname = os.path.join(short_runs_dirname, subdirname+str(i+1))
        run_data = matlab_csv_to_teacher_data(dirname)
        if i == 0:
            data = run_data
        else:
            for i, phoneme_data in enumerate(run_data):
                data[i] = np.vstack((data[i], phoneme_data))

        data_length += run_data[0].shape[0]

    for i, phoneme_data in enumerate(data):
        assert phoneme_data.shape[0] == data_length, "Data was imporperly constructed."

    return data


def bell_number(n):
    """
    Returns the nth bell number. Uses approximation.

    Example:
        >>> # generate bell({0,1,...,13})
        >>> [bell_number(n) for n in range(14)]
        [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597, 27644437]
    """
    return int((1/math.e) * sum([(k**n)/(math.factorial(k)) for k in range(ITERATIONS)])+.5)


def next_partition(Z, k, h):
    """
    Generates the next partition, Z, and histogram, h given the curerent partition and histogram
    given the pseudo counts, k. This function will mutate Z, k, and h.
    """
    n = len(Z)
    for i in range(n-1, 0, -1):
        if(Z[i] <= k[i-1]):
            h[Z[i]] -= 1
            Z[i] += 1

            if Z[i] == len(h):
                h.append(1)
            else:
                h[Z[i]] += 1

            k[i] = Z[i] if (k[i] <= Z[i]) else k[i]

            for j in range(i+1, n):
                h[Z[j]] -= 1
                h[Z[0]] += 1

                Z[j] = Z[0]
                k[j] = k[i]

            while h[-1] == 0:
                    del h[-1]

            return Z, k, h
    return None


def parition_generator(n):
    """
    Generates partitionings of n objects into 1 to n partitions

    Returns:
        numpy.ndarray<int>: a n-length array where entry, i, is an integer indicating to which
            partition the ith object is assigned.

    Examples:
        >>> Z = parition_generator(3)
        >>> [z for z in Z]
        [array([0, 1, 2]), array([0, 1, 2]), array([0, 1, 2]), array([0, 1, 2]), array([0, 1, 2])]
        >>> Z = parition_generator(8)
        >>> len([z for z in Z])  # this should be the same as bell_number(8)
        4140
        >>> bell_number(8)
        4140
    """
    # generator
    k = np.zeros(n, dtype=np.dtype(int))
    Z = np.zeros(n, dtype=np.dtype(int))
    h = [float(n)]
    yield(Z)
    while next_partition(Z, k, h) is not None:
        yield(Z)


def pflip(P):
    """
    Multinomial draw from a vector P of probabilities
    """
    if len(P) == 1:
        return 0

    P /= sum(P)

    assert math.fabs(1.0-sum(P)) < 10.0**(-10.0)

    p_minus = 0
    r = np.random.rand()
    for i in range(len(P)):
        P[i] += p_minus
        p_minus = P[i]
        if r < p_minus:
            return i

    raise IndexError("pflip:failed to find index")


def lcrp(N, Nk, alpha):
    """
    Returns the log probability under crp of the count vector Nk given
    concentration parameter alpha. N is the total number of entries
    """
    N = float(N)
    k = float(len(Nk))  # number of classes
    l = np.sum(gammaln(Nk))+k*log(alpha)+gammaln(alpha)-gammaln(N+alpha)
    return l


def lpflip(P):
    """
    Multinomial draw from a vector P of log probabilities
    """
    if len(P) == 1:
        return 0

    Z = logsumexp(P)
    P -= Z

    NP = np.exp(np.copy(P))

    assert math.fabs(1.0-sum(NP)) < 10.0**(-10.0)

    return pflip(NP)


def crp_gen(N, alpha):
    """
    Generates a random, N-length partition from the CRP with parameter alpha
    """
    assert N > 0
    assert alpha > 0.0
    alpha = float(alpha)

    partition = np.zeros(N, dtype=int)
    Nk = [1]
    for i in range(1, N):
        K = len(Nk)
        ps = np.zeros(K+1)
        for k in range(K):
            # get the number of people sitting at table k
            ps[k] = float(Nk[k])

        ps[K] = alpha

        ps /= (float(i)-1+alpha)

        assignment = pflip(ps)

        if assignment == K:
            Nk.append(1)
        elif assignment < K:
            Nk[assignment] += 1
        else:
            raise ValueError("invalid assignment: %i, max=%i" % (assignment, K))

        partition[i] = assignment

    assert max(partition)+1 == len(Nk)
    assert len(partition) == N
    assert sum(Nk) == N

    K = len(Nk)

    if K > 1:
        shuffle(partition)

    return np.array(partition), Nk, K

# _________________________________________________________________________________________________
# Plot utils
# `````````````````````````````````````````````````````````````````````````````````````````````````
def plot_data_2d(teacher_data, target_model):
    """
    TODO: Fill in
    """

    for i, X_k in enumerate(teacher_data):
        plt.scatter(X_k[:, 0], X_k[:, 1], c='blue', alpha=.5, zorder=5)
        mu_teacher = np.mean(X_k, axis=0)
        plt.scatter(mu_teacher[0], mu_teacher[1], c='blue', s=15**2, zorder=6)

        mu_target = target_model['parameters'][i][0]
        # import pdb; pdb.set_trace()
        plt.scatter(mu_target[0], mu_target[1], c='red', s=15**2, zorder=7)


# http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
