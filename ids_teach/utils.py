import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns  # never called explictly, but changes pyplot settings

from scipy.special import multigammaln
from numpy.linalg import slogdet
from numpy.linalg import solve
from scipy import linalg
from numpy import trace
from math import log

ITERATIONS = 1000


def dist(a, b):
    """
    Distance between two numpy arrays

    Example:
        >>> import numpy as np
        >>> a = np.array([1.2, -3.4])
        >>> b = np.array([0.0, 6.8])
        >>> dist(a, b)
        10.2878569197
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
    h = np.zeros(n, dtype=np.dtype(float))
    yield(Z)
    while next_partition(Z, k, h) is not None:
        yield(Z)


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
    sbplot = plt.gca()
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    ell = Ellipse(pos, .04*v[0], .04*v[1], 180 + angle, **kwargs)
    ell.set_clip_box(sbplot.bbox)
    ell.set_alpha(kwargs.get('alpha', 0))
    sbplot.add_artist(ell)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
