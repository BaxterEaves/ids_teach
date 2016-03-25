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

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.mixture import GMM
from sklearn import metrics
import multiprocessing as mp
from idsteach import ids
from idsteach import utils

from scipy.stats import ks_2samp
from scipy.stats import ttest_ind

from idsteach.fastniw import PyDPGMM

import pandas as pd
import numpy as np

import pickle
import random
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')
sns.set(rc={'axes.facecolor': '#cccccc', 'grid.color': '#dddddd'})

DIR = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(DIR, 'data', 'algcomp.pkl')

# theses are the algorihtms we evaluate. Note that scikit implmentations of
# vbgmm and dpgmm need some work, so I do not include them in analyses (I
# use my own implementation of dpgmm).
# TODO: Remove code for unused methods.
algorithm_list = ['logit', 'svm_linear', 'gmm', 'dpgmm']


def algcomp(opt_data, target_model, data_model, n_per_cat, n_runs,
            filename=None, do_plot=False):
    """
    Compare the performance of various algorithms on the optimized data and
    data drawn directly from the target distribution. Measures using ARI
    (adjusted Rand index), which is the gold standard. Uses the following
    algorithms:

    1. Multinomial logistic regression (trained on labeled data)
    2. Support-vector machine with linear kernel (trained on labeled data)
    3. Support-vector machine with radial basis (RBF) kernel (trained on
       labeled data)
    4. k-means (given the correct number of components)
    5. Gaussian mixture model via Expectation-maximization (given correct
       number of categories)
    6. Dirichlet process mixture model (given prior ditribution)

    Parameters
    ----------
    opt_data : list<numpy.ndarray>
        A list a data, separated by category (<teacher>.data)
    target_model : dict
        target_model used to generate opt_data
    data_model : CollapsibleDistribution
        data model with prior used by the teacher
    n_per_cat : list<int>
        number of data points to use for training and testing.
    n_runs : int
        number of times over which to average results
    filename : string
        If not None (default), saves the results to filename as a pickle of
        pandas dataframes.

    Additional arguments
    --------------------
    do_plot : bool
        do plotting? I should probably remove this because plotting from within
        multiprocessing is going to break everything.
    """
    data_model.validate_target_model(target_model)

    columns = ['mean_{ADS}', 'mean_{OPT}', 'mean_{CROSS}', 'std_{ADS}',
               'std_{OPT}', 'std_{CROSS}']
    keys = algorithm_list

    ari_orig_mean = __prep_result_holder(len(n_per_cat))
    ari_orig_std = __prep_result_holder(len(n_per_cat))
    ari_opt_mean = __prep_result_holder(len(n_per_cat))
    ari_opt_std = __prep_result_holder(len(n_per_cat))
    ari_cross_mean = __prep_result_holder(len(n_per_cat))
    ari_cross_std = __prep_result_holder(len(n_per_cat))

    data_output = dict()

    if do_plot:
        plt.figure(tight_layout=True, facecolor='white')
        plt.axis([0, 1000, 0, 1])
        plt.ion()
        plt.show()

    for j, n_per in enumerate(n_per_cat):
        print("Running evaluations for {0} examples per category for {1} "
              "runs.".format(n_per, n_runs))
        data_output[n_per] = dict()
        res_orig_n = __prep_result_holder(n_runs)
        res_opt_n = __prep_result_holder(n_runs)
        res_cross_n = __prep_result_holder(n_runs)

        # construct args. seed the rng for each processor. I was having trouble
        # with the numpy rng using the same seed across cores.
        args = []
        for _ in range(n_runs):
            args.append((opt_data, target_model, data_model, n_per,
                         np.random.randint(0, 2**32-1)))

        pool = mp.Pool()
        pool.daemon = True
        res = pool.map(__eval_all_mp, args)
        for i, (orig_res, opt_res, cross_res, z_true) in enumerate(res):
            for alg in algorithm_list:
                res_orig_n[alg]['ari'][i] = orig_res[alg][0]
                res_opt_n[alg]['ari'][i] = opt_res[alg][0]
                res_cross_n[alg]['ari'][i] = cross_res[alg][0]

                res_orig_n[alg]['z'][i] = orig_res[alg][1]
                res_opt_n[alg]['z'][i] = opt_res[alg][1]
                res_cross_n[alg]['z'][i] = cross_res[alg][1]

        print(' ')

        for alg in algorithm_list:
            ari_orig_mean[alg][j] = np.mean(res_orig_n[alg]['ari'])
            ari_orig_std[alg][j] = np.std(res_orig_n[alg]['ari'])
            ari_opt_mean[alg][j] = np.mean(res_opt_n[alg]['ari'])
            ari_opt_std[alg][j] = np.std(res_opt_n[alg]['ari'])
            ari_cross_mean[alg][j] = np.mean(res_cross_n[alg]['ari'])
            ari_cross_std[alg][j] = np.std(res_cross_n[alg]['ari'])

        # build table
        table_data = np.zeros((len(keys), len(columns)))
        for row, alg in enumerate(keys):
            table_data[row, 0] = ari_orig_mean[alg][j]
            table_data[row, 1] = ari_opt_mean[alg][j]
            table_data[row, 2] = ari_cross_mean[alg][j]

            table_data[row, 3] = ari_orig_std[alg][j]
            table_data[row, 4] = ari_opt_std[alg][j]
            table_data[row, 5] = ari_cross_std[alg][j]

        df = pd.DataFrame(table_data, index=keys, columns=columns)
        data_output[n_per]['dataframe'] = df
        data_output[n_per]['res_optimal'] = res_opt_n
        data_output[n_per]['res_original'] = res_orig_n
        data_output[n_per]['res_cross'] = res_cross_n
        data_output[n_per]['z_true'] = z_true
        print(df)

    if filename is not None:
        pickle.dump(data_output, open(filename, "wb"))


def __eval_all_mp(args):
    np.random.seed(args[-1])
    return __eval_all(*args[:-1])


def __eval_all(opt_data, target_model, data_model, n_per_cat, do_plot=False):
    """
    Creates datasets and labels and evaluates clustering performance of all
    algorithms on both the data derived from the target model as well as
    optimized data from a Teacher.

    Parameters
    ----------
    opt_data : list<numpy.ndarray>
        A list a data, separated by category (<teacher>.data)
    target_model : dict
        target_model used to generate opt_data
    data_model : CollapsibleDistribution
        data model with prior used by the teacher
    n_per_cat : int
        number of data points to use for training and testing

    Returns
    -------
    res_orig : dict
        each (key, value) pair corresponds to an algorithm (str) and a tuples
        with the  ARI value (float) that algorithm achieved and its inferred
        assignment vector res_opt (dict): each (key, value) pair corresponds to
        an algorithm (str) and a tuples with the  ARI value (float) that
        algorithm achieved and its inferred assignment vector
    """
    data_train_orig, z_train_orig = __prep_standard_data(
        target_model, data_model, n_per_cat)
    data_test_orig, z_test_orig = __prep_standard_data(
        target_model, data_model, n_per_cat)
    data_train_opt, z_train_opt = __prep_optimal_data(opt_data, n_per_cat)
    data_test_opt, z_test_opt = __prep_optimal_data(opt_data, n_per_cat)

    if do_plot:
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.scatter(data_train_orig[:, 0], data_train_orig[:, 1], alpha=.5,
                    color='red')
        plt.title('Original data (train)')

        plt.subplot(2, 2, 2)
        plt.scatter(data_train_opt[:, 0], data_train_opt[:, 1], alpha=.5,
                    color='blue')
        plt.title('Optimal data (train)')

        plt.subplot(2, 2, 3)
        plt.scatter(data_test_orig[:, 0], data_test_orig[:, 1], alpha=.5,
                    color='red')
        plt.title('Original data (test)')

        plt.subplot(2, 2, 4)
        plt.scatter(data_test_opt[:, 0], data_test_opt[:, 1], alpha=.5,
                    color='blue')
        plt.title('Optimal data (test)')
        plt.draw()

    print("<", end="")
    sys.stdout.flush()
    res_orig = __eval_set(data_train_orig, z_train_orig, data_test_orig,
                          z_test_orig, data_model)
    print("|", end="")
    sys.stdout.flush()
    res_opt = __eval_set(data_train_opt, z_train_opt, data_test_opt,
                         z_test_opt, data_model)
    print(">", end="")
    sys.stdout.flush()
    res_cross = __eval_set(data_train_opt, z_train_opt, data_test_orig,
                           z_test_orig, data_model)

    return res_orig, res_opt, res_cross, z_test_opt


def __eval_set(train_data, train_labels, test_data, test_labels, data_model):
    """
    Tests the cluster performance of all algorithms on training and testing
    data sets.  Note that test data is only used for supervised algorithms,
    which learn from labeled data.  All other algorithms are tested on their
    partitioning of the training data.

    Parameters
    ----------
    train_data : numpy.ndarray
        training data
    train_labels : numpy.ndarray
        labels of training data
    test_data : numpy.ndarray
        testing data
    test_labels : numpy.ndarray
        labels of testing data

    Returns
    -------
    res : dict
        each (key, value) pair corresponds to an algorithm (str) and a tuples
        with the  ARI value (float) that algorithm achieved and its inferred
        assignment vector.
    """

    # fixed-categoty algorithms
    K = max(train_labels)+1
    res_gmm = __eval_gmm(train_data, train_labels, K, test_data=test_data,
                         test_labels=test_labels)

    # variational algorithms
    res_dpgmm = __eval_dpgmm(train_data, train_labels, data_model, alpha=1.0,
                             test_data=test_data, test_labels=test_labels)

    # labeled-set algorithms
    res_logit = __eval_logit(train_data, train_labels, test_data, test_labels)
    res_svm_linear = __eval_svm(train_data, train_labels, test_data,
                                test_labels, kernel='linear')

    # each entry in res is an (ARI, assgnment vector) tuple
    res = {
        'logit': res_logit,
        'svm_linear': res_svm_linear,
        'gmm': res_gmm,
        'dpgmm': res_dpgmm,
    }

    return res


# _____________________________________________________________________________
# helpers
# `````````````````````````````````````````````````````````````````````````````
def __prep_result_holder(n):
    """
    Returns a dict with a n-length numpy.ndarray of zeros for each clustering
    algorithm
    """
    holder = dict()
    for alg in algorithm_list:
        holder[alg] = {
            'ari': np.zeros(n, dtype=np.dtype(float)),
            'z': [None]*n}
    return holder


def __prep_standard_data(target_model, data_model, n_per_category):
    """
    Samples n_percategory datapoints from each component in target model
    accoring to the data model defined by data_model.

    Returns
    -------
    data : numpy.ndarray
        the data set
    Z : numpy.ndarray
        the data labels
    """
    data = None
    Z = []
    for i, params in enumerate(target_model['parameters']):
        X = data_model.draw_data_from_params(*params, n=n_per_category)
        data = X if i == 0 else np.vstack((data, X))
        Z += [i]*n_per_category
    return data, Z


def __prep_optimal_data(opt_data, n_per_category):
    """
    Pulls a n_per_category subset from eaach category and stacks them into one
    array.

    Returns
    -------
    data : numpy.ndarray
        the data set
    Z : numpy.ndarray
        the data labels
    """
    n = opt_data[0].shape[0]
    data = None
    Z = []
    for i, X in enumerate(opt_data):
        rows = random.sample([idx for idx in range(n)], n_per_category)
        data = X[rows] if i == 0 else np.vstack((data, X[rows]))
        Z += [i]*n_per_category
    return data, Z


# _____________________________________________________________________________
# Algorithms w/ known number of clusters
# `````````````````````````````````````````````````````````````````````````````
def __eval_gmm(data, labels, num_components, test_data=None, test_labels=None):
    """
    Expectation-maximization with a given number of categories
    """
    gmm = GMM(n_components=num_components, n_iter=100)
    gmm.fit(data)

    if test_data is not None and test_labels is not None:
        Z = gmm.predict(test_data)
        ari = metrics.adjusted_rand_score(test_labels, Z)
    else:
        ari = metrics.adjusted_rand_score(labels, Z)
    ari = metrics.adjusted_rand_score(labels, Z)

    return ari, Z


# _____________________________________________________________________________
# Algorithms w/ labeled data
# `````````````````````````````````````````````````````````````````````````````
def __eval_logit(train_data, train_labels, test_data, test_labels):
    """
    Multinomial logistic regression.
    Trained on train_data, tested on test_data. Returns ARI(test_labels, Z)
    """
    lr = LogisticRegression()
    lr.fit(train_data, train_labels)
    Z = lr.predict(test_data)
    ari = metrics.adjusted_rand_score(test_labels, Z)
    return ari, Z


def __eval_svm(train_data, train_labels, test_data, test_labels,
               kernel='linear'):
    """
    Support-vector machine with user-specified kernel (linear, rbf, etc)
    Trained on train_data, tested on test_data. Returns ARI(test_labels, Z)
    """
    vm = svm.SVC(kernel=kernel)
    vm.fit(train_data, train_labels)
    Z = vm.predict(test_data)
    ari = metrics.adjusted_rand_score(test_labels, Z)
    return ari, Z


# ____________________________________________________________________________
# Variational algorithms
# These algorithms must be given a max number of categories, then they choose
# which categories to keep around. We need to choose high, but if we choose
# too high, we have to wait froever.  I chose n/k because I want to build in a
# little as possible.
# ````````````````````````````````````````````````````````````````````````````
def __eval_dpgmm(data, labels, data_model, alpha=1.0, test_data=None,
                 test_labels=None):
    """
    Dirichlet proccess EM for Gaussian mixture models.
    Returns ARI(test_labels, Z)
    """
    dpgmm = PyDPGMM(data_model, data, crp_alpha=alpha,
                    init_mode='single_cluster')
    Z = dpgmm.fit(n_iter=500, sm_prop=.25, sm_burn=50, num_sm_sweeps=2)
    if test_data is not None and test_labels is not None:
        Z = dpgmm.predict(test_data)
        ari = metrics.adjusted_rand_score(test_labels, Z)
    else:
        ari = metrics.adjusted_rand_score(labels, Z)

    return ari, Z


# ____________________________________________________________________________
def ari_subset(data_n, category_list):
    idx = [i for i, zi in enumerate(data_n['z_true']) if zi in category_list]
    z_true = [data_n['z_true'][i] for i in idx]
    res_optimal = dict()
    res_original = dict()
    res_cross = dict()

    for alg in algorithm_list:
        res_optimal[alg] = np.zeros(len(data_n['res_optimal'][alg]['ari']))
        res_original[alg] = np.zeros(len(data_n['res_original'][alg]['ari']))
        res_cross[alg] = np.zeros(len(data_n['res_cross'][alg]['ari']))

        for i in range(len(data_n['res_optimal'][alg]['z'])):
            z_opt = [data_n['res_optimal'][alg]['z'][i][z] for z in idx]
            z_orig = [data_n['res_original'][alg]['z'][i][z] for z in idx]
            z_cross = [data_n['res_cross'][alg]['z'][i][z] for z in idx]

            res_optimal[alg][i] = metrics.adjusted_rand_score(z_true, z_opt)
            res_original[alg][i] = metrics.adjusted_rand_score(z_true, z_orig)
            res_cross[alg][i] = metrics.adjusted_rand_score(z_true, z_cross)

    return res_optimal, res_original, res_cross


def plot_single_result(df, ptype, axes, stat_type='KS', grayscale=True):

    def get_clip(x):
        s = np.std(x)
        m = np.mean(x)
        return [m-4.0*s, m+4.0*s]

    if grayscale:
        c1, c3, c2 = sns.color_palette("Greys", 3)
        c1 = '#f8f8f8'
    else:
        c1, c3, c2 = sns.color_palette("colorblind", 3)

    for j, alg in enumerate(algorithm_list):
        ari_orig = df['res_original'][alg]['ari']
        ari_opt = df['res_optimal'][alg]['ari']

        print("KS stat, p {}".format(alg))
        ks_ari, p_ari = ks_2samp(ari_orig, ari_opt)
        print("\tADS-OPT KS: {}, p: {}".format(ks_ari, p_ari))

        if 'res_cross' in df:
            ari_cross = df['res_cross'][alg]['ari']
            ks_ari, p_ari = ks_2samp(ari_orig, ari_cross)
            print("\tADS-CROSS KS: {}, p: {}".format(ks_ari, p_ari))
            ks_ari, p_ari = ks_2samp(ari_opt, ari_cross)
            print("\tOPT-CROSS KS: {}, p: {}".format(ks_ari, p_ari))
        else:
            ari_cross = None

        if ptype == 'kde':
            clip_orig = get_clip(ari_orig)
            clip_opt = get_clip(ari_opt)

            sns.distplot(ari_orig, color=c1, ax=axes[j], vertical=True,
                         kde_kws=dict(clip=clip_orig,
                                      label=(None if j > 0 else 'ADS'),
                                      lw=2),
                         hist_kws=dict(histtype="stepfilled"))
            sns.distplot(ari_opt, color=c2, ax=axes[j], vertical=True,
                         kde_kws=dict(clip=clip_opt,
                                      label=None if j > 0 else 'Teaching',
                                      lw=2),
                         hist_kws=dict(histtype="stepfilled"))
            if ari_cross is not None:
                clip_cross = get_clip(ari_cross)
                sns.distplot(ari_cross, color=c3, ax=axes[j], vertical=True,
                             kde_kws=dict(clip=clip_cross,
                                          label=None if j > 0 else 'Transfer',
                                          lw=2),
                             hist_kws=dict(histtype="stepfilled"))

            if ari_cross is None:
                tstat, tp = ttest_ind(ari_orig, ari_opt)
                if stat_type.upper() == 'KS':
                    txt = "KS: %1.4f\np: %1.4f" % (ks_ari, p_ari)
                elif stat_type.upper() == 'T':
                    txt = "\nt: %1.4f\n p: %1.4f" % (tstat, tp)
                else:
                    raise ValueError("Invalid stat_type: {}".format(stat_type))
                axes[j].text(1, .95, txt, ha='right', va='top',
                             transform=axes[j].transAxes, color='#333333')

            y_l = axes[j].get_ylim()[0]
            y_u = axes[j].get_ylim()[1]
            ystp = (y_u - y_l)/3.
            yticks = [y_l, y_l + ystp, y_l + ystp*2, y_u]
            yticks = [round(tick, 2) for tick in yticks]
            axes[j].set_ylim([y_l, y_u])
            axes[j].set_ylabel('')
            axes[j].set_yticks(yticks)
            if j == 0:
                axes[j].set_ylabel('ARI')
                axes[j].set_xlabel('Density')
                axes[j].legend(loc=2)
        elif ptype == 'violin':
            aris = [ari_orig, ari_opt]
            names = ['ADS', 'Teaching']
            if ari_cross is not None:
                aris += [ari_cross]
                names += ['Transfer']
            sns.violinplot(aris, positions=1, ax=axes.flat[j], names=names,
                           alpha=.5)

            axes[j].set_ylabel('ARI')
        axes[j].set_title(alg.upper())
    print(' ')
    return axes


def plot_compare(filenames, Ns, ylabels=None, figwidth=8.5, stat_type='KS',
                 grayscale=False):
    if ylabels is None:
        ylabels = ['Density']*len(filenames)

    if isinstance(Ns, float):
        Ns = [Ns]*len(filenames)

    if grayscale:
        c1, c3, c2 = sns.color_palette("Greys", 3)
        c1 = '#f8f8f8'
    else:
        c1, c3, c2 = sns.color_palette("colorblind", 3)

    ptype = 'kde'

    f, axes = plt.subplots(len(filenames), len(algorithm_list),
                           figsize=(figwidth, figwidth/1.5))

    dfs = [pickle.load(open(fname, 'rb'))[n] for n, fname in
           zip(Ns, filenames)]

    axes = tuple([plot_single_result(df, ptype, axes[i, :].tolist(),
                 stat_type=stat_type, grayscale=grayscale)
                 for i, df in enumerate(dfs)])

    y_l = min([min([ax.get_ylim()[0] for ax in axis_row]) for axis_row
               in axes])
    y_u = max([max([ax.get_ylim()[1] for ax in axis_row]) for axis_row
               in axes])

    ystp = (y_u - y_l)/3.
    yticks = [y_l, y_l + ystp, y_l + ystp*2, y_u]
    yticks = [round(tick, 2) for tick in yticks]

    for row, axis_row in enumerate(axes):
        axis_row[0].set_ylabel(ylabels[row])
        for col, ax in enumerate(axis_row):
            ax.set_ylim([y_l, y_u])
            ax.set_yticks(yticks)

            if col != 0:
                ax.set_yticklabels([])
                ax.set_ylabel('')

    for col, axis_col in enumerate(zip(*axes)):
        x_l = 0
        x_u = max([ax.get_xlim()[1] for ax in axis_col])
        alg = algorithm_list[col]

        for row, ax in enumerate(axis_col):
            ax.set_xlim([x_l, x_u])
            if row != len(axis_col)-1:
                ax.set_xticklabels([])
                ax.set_xlabel('')

            if row != 0:
                ax.set_title('')

            mot = dfs[row]['res_original'][alg]['ari'].mean()
            mtt = dfs[row]['res_optimal'][alg]['ari'].mean()

            ax.plot([0, x_u], [mot, mot], lw=2, color=c1, ls='--')
            ax.plot([0, x_u], [mtt, mtt], lw=2, color=c2, ls='--')

            if 'res_cross' in dfs[row]:
                mct = dfs[row]['res_cross'][alg]['ari'].mean()
                ax.plot([0, x_u], [mct, mct], lw=2, color=c3, ls='--')

    return f, axes


def plot_result(filename, type='kde', suptitle=None, base_filename=None):
    data = pickle.load(open(filename, 'rb'))
    N = [key for key in data.keys()]
    N = sorted(N)

    for i, n in enumerate(N):
        f, axes = plt.subplots(1, len(algorithm_list), figsize=(7.5, 3.5))
        f.set_facecolor('white')
        f.tight_layout()

        axes = plot_single_result(data[n], type, axes)

        if suptitle is not None:
            plt.suptitle(suptitle)
        else:
            plt.suptitle("N=%i" % n)

        if base_filename:
            filename = base_filename + "_" + str(n) + "n.png"
            plt.savefig(filename, dpi=300)
        else:
            plt.show()


if __name__ == '__main__':
    import argparse
    from idsteach.models import NormalInverseWishart

    parser = argparse.ArgumentParser(description='Run examples')
    parser.add_argument('--num_examples', metavar='N', type=int, nargs='+',
                        help='list of number of exampler per phoneme')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='Number of runs to average over.')
    parser.add_argument('--plot_type', type=str, default='kde',
                        help="type of plot 'kde' (default) or 'violin'")
    parser.add_argument('--filename', type=str, default='alcomptest.pkl',
                        help='save as filename')
    parser.add_argument('--multirun', action='store_true', default=False,
                        help='use data from multiple sampler chains')
    parser.add_argument('--base_figname', type=str, default='alcomptest',
                        help='save figure as filename')
    parser.add_argument('--matlab_data', action='store_true',
                        help='input data is a matlab .csv rather than '
                        'pandas df')
    parser.add_argument('--flatten', action='store_true',
                        help='Remove f3 dimension (only applicable for '
                        'non-matlab data')
    parser.add_argument('--plot_only', action='store_true',
                        help='Plot existing data.')

    args = parser.parse_args()

    if not args.plot_only:
        if args.flatten and args.matlab_data:
            raise ValueError('Cannot flatten matlab data')

        target_model, labels = ids.gen_model(f3=(not args.matlab_data))
        data_model = NormalInverseWishart.with_vague_prior(target_model)

        if args.flatten:
            target_model, data_model = utils.flatten_niw_model(
                target_model, data_model)

        if args.multirun:
            if args.matlab_data:
                dirname = os.path.join(DIR, '../data', 'ml_runs')
                data = utils.multiple_matlab_csv_to_teacher_data(dirname)
            else:
                raise NotImplementedError('Must use multiple data sets when '
                                          'using pandas')
        else:
            if args.matlab_data:
                dirname = os.path.join(DIR, '../data', 'lrunml')
                data = utils.matlab_csv_to_teacher_data(dirname)
            else:
                dirname = os.path.join(DIR, '../data', 'ptn_runs')
                data = utils.multiple_pandas_to_teacher_data(
                    dirname, remove_f3=args.flatten)

        algcomp(data, target_model, data_model, args.num_examples,
                args.num_runs, filename=args.filename)

    plot_result(args.filename, args.plot_type, base_filename=args.base_figname)
