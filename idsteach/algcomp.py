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
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.mixture import GMM
from sklearn.mixture import VBGMM
from sklearn.mixture import DPGMM as DPGMM_sk
from sklearn import metrics
from math import log
import multiprocessing as mp
from idsteach import ids
from idsteach import utils

from idsteach.dpgmm import DPGMM
from idsteach.fastniw import PyDPGMM

import pandas as pd
import numpy as np

import pickle
import random
import time
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns  # never called explictly, but changes pyplot settings

# theses are the algorihtms we evaluate. Note that scikit implmentations of vbgmm and dpgmm
# need some work, so I do not include them in analyses (I use my own implementation of dpgmm).
# algorithm_list = ['logit', 'svm_linear', 'svm_rbf', 'k_means', 'gmm', 'vbgmm', 'dpgmm']
algorithm_list = ['logit', 'svm_linear', 'svm_rbf', 'k_means', 'gmm', 'dpgmm']


def algcomp(opt_data, target_model, data_model, n_per_cat, n_runs, filename=None, do_plot=False):
    """
    Compare the performance of various algorithms on the optimized data and data drawn directly
    from the target distribution. Measures using ARI (adjusted Rand index), which is the gold
    standard. Uses the following algorithms:

    1. Multinomial logistic regression (trained on labeled data)
    2. Support-vector machine with linear kernel (trained on labeled data)
    3. Support-vector machine with radial basis (RBF) kernel (trained on labeled data)
    4. k-means (given the correct number of components)
    5. Gaussian mixture model via Expectation-maximization (given correct number of categories)
    6. Dirichlet process mixture model (given prior ditribution)

    Inputs:
        opt_data (list<numpy.ndarray>): A list a data, separated by category (<teacher>.data)
        target_model (dict): target_model used to generate opt_data
        data_model (CollapsibleDistribution): data model with prior used by the teacher
        n_per_cat (list<int>): number of data points to use for training and testing. 
        n_runs (int): number of times over which to average results

    Kwargs:
        filename (string): If not None (default), saves the results to filename as a pikcle of
            pandas dataframes.
        do_plot (bool): do plotting? I should probably remove this because plotting from withing
            multiprocessing is going to break everything.

    Returns:
        None
    """
    data_model.validate_target_model(target_model)

    columns = ['mean_{ADS}', 'mean_{OPT}', 'std_{ADS}', 'std_{OPT}']
    keys = algorithm_list

    ari_orig_mean = __prep_result_holder(len(n_per_cat))
    ari_orig_std = __prep_result_holder(len(n_per_cat))
    ari_opt_mean = __prep_result_holder(len(n_per_cat))
    ari_opt_std = __prep_result_holder(len(n_per_cat))

    data_output = dict()

    if do_plot:
        plt.figure(tight_layout=True, facecolor='white')
        plt.axis([0, 1000, 0, 1])
        plt.ion()
        plt.show()

    for j, n_per in enumerate(n_per_cat):
        print("Running evaluations for {0} examples per category for {1} runs.".format(n_per, n_runs))
        data_output[n_per] = dict()
        ari_orig_n = __prep_result_holder(n_runs)
        ari_opt_n = __prep_result_holder(n_runs)

        # construct args. seed the rng for each processor. I was having trouble with the numpy rng
        # using the same seed across cores.
        args = []
        for _ in range(n_runs):
            args.append((opt_data, target_model, data_model, n_per, np.random.randint(0, 2**32-1)))

        pool = mp.Pool()
        pool.daemon = True
        res = pool.map(__eval_all_mp, args)
        for i, (ari_orig_res, ari_opt_res) in enumerate(res):
            for alg in algorithm_list:
                ari_orig_n[alg][i] = ari_orig_res[alg]
                ari_opt_n[alg][i] = ari_opt_res[alg]

        print(' ')

        for alg in algorithm_list:
            ari_orig_mean[alg][j]  = np.mean(ari_orig_n[alg])
            ari_orig_std[alg][j]  = np.std(ari_orig_n[alg])
            ari_opt_mean[alg][j]  = np.mean(ari_opt_n[alg])
            ari_opt_std[alg][j]  = np.std(ari_opt_n[alg])

        # build table
        table_data = np.zeros((len(keys), len(columns)))
        for row, alg in enumerate(keys):
            table_data[row, 0] = ari_orig_mean[alg][j]
            table_data[row, 1] = ari_opt_mean[alg][j]
            table_data[row, 2] = ari_orig_std[alg][j]
            table_data[row, 3] = ari_opt_std[alg][j]

        df = pd.DataFrame(table_data, index=keys, columns=columns)
        data_output[n_per]['dataframe'] = df
        data_output[n_per]['ari_optimal'] = ari_opt_n
        data_output[n_per]['ari_original'] = ari_orig_n

        print(df)

    if filename is not None:
        pickle.dump(data_output, open(filename, "wb"))


def __eval_all_mp(args):
    np.random.seed(args[-1])
    return __eval_all(*args[:-1])


def __eval_all(opt_data, target_model, data_model, n_per_cat, do_plot=False):
    """
    Creates datasets and labels and evaluates clustering performance of all algorithms on both the
    data derived from the target model as well as optimized data from a Teacher.

    Inputs:
        opt_data (list<numpy.ndarray>): A list a data, separated by category (<teacher>.data)
        target_model (dict): target_model used to generate opt_data
        data_model (CollapsibleDistribution): data model with prior used by the teacher
        n_per_cat (int): number of data points to use for training and testing

    Returns:
        aris_orig (dict): each  (key, value) is a algorithm (str) the ARI value achieved by that
            algorithm on the data sampled from the target model
        aris_opt (dict): each  (key, value) is a algorithm (str) the ARI value achieved by that
            algorithm on a subset of the optimzed data
    """
    data_train_orig, z_train_orig = __prep_standard_data(target_model, data_model, n_per_cat)
    data_test_orig, z_test_orig = __prep_standard_data(target_model, data_model, n_per_cat)
    data_train_opt, z_train_opt = __prep_optimal_data(opt_data, n_per_cat)
    data_test_opt, z_test_opt = __prep_optimal_data(opt_data, n_per_cat)

    if do_plot:
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.scatter(data_train_orig[:, 0], data_train_orig[:, 1], alpha=.5, color='red')
        plt.title('Original data (train)')

        plt.subplot(2, 2, 2)
        plt.scatter(data_train_opt[:, 0], data_train_opt[:, 1], alpha=.5, color='blue')
        plt.title('Optimal data (train)') 

        plt.subplot(2, 2, 3)
        plt.scatter(data_test_orig[:, 0], data_test_orig[:, 1], alpha=.5, color='red')
        plt.title('Original data (test)')

        plt.subplot(2, 2, 4)
        plt.scatter(data_test_opt[:, 0], data_test_opt[:, 1], alpha=.5, color='blue')
        plt.title('Optimal data (test)')
        plt.draw()

    print(".", end=" ")
    sys.stdout.flush()
    aris_orig = __eval_set(data_train_orig, z_train_orig, data_test_orig, z_test_orig, data_model)
    print("`", end=" ")
    sys.stdout.flush()
    aris_opt = __eval_set(data_train_opt, z_train_opt, data_test_opt, z_test_opt, data_model)

    return aris_orig, aris_opt

def __eval_set(train_data, train_labels, test_data, test_labels, data_model):
    """
    Tests the cluster performance of all algorithms on training and testing data sets.
        Note that test data is only used for supervised algorithms, which learn from labeled data.
    All other algorithms are tested on their partitioning of the training data.

    Inputs:
        train_data (numpy.ndarray): training data
        train_labels (numpy.ndarray): labels of training data
        test_data (numpy.ndarray): testing data
        test_labels (numpy.ndarray): labels of testing data

    Returns:
        ari (dict): each (key, value) pair corresponds to an algorithm (str) and the  ARI value
            (float) that algorithm achieved
    """

    # fixed-categoty algorithms
    K = max(train_labels)+1
    ari_k_means = __eval_kmeans(train_data, train_labels, K)
    ari_gmm = __eval_gmm(train_data, train_labels, K)

    # variational algorithms
    # ari_vbgmm = __eval_vbgmm(train_data, train_labels, alpha=1.0)
    ari_dpgmm = __eval_dpgmm(train_data, train_labels, data_model, alpha=1.0)

    # labeled-set algorithms
    ari_logit = __eval_logit(train_data, train_labels, test_data, test_labels)
    ari_svm_linear = __eval_svm(train_data, train_labels, test_data, test_labels, kernel='linear')
    ari_svm_rbf = __eval_svm(train_data, train_labels, test_data, test_labels, kernel='rbf')

    ari = {
        'logit': ari_logit,
        'svm_linear': ari_svm_linear,
        'svm_rbf': ari_svm_rbf,
        'k_means': ari_k_means,
        'gmm': ari_gmm,
        # 'vbgmm': ari_vbgmm,
        'dpgmm': ari_dpgmm,
    }

    return ari


# __________________________________________________________________________________________________
# helpers
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def __prep_result_holder(n):
    """ 
    Returns a dict with a n-length numpy.ndarray of zeros for each clustering algorithm
    """
    holder = dict()
    for alg in algorithm_list:
        holder[alg] = np.zeros(n, dtype=np.dtype(float))    
    return holder


def __prep_standard_data(target_model, data_model, n_per_category):
    """
    Samples n_percategory datapoints from each component in target model accoring to the data model
    defined by data_model.

    Returns:
        data (numpy.ndarray): the data set
        Z (numpy.ndarray): the data labels
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
    Pulls a n_per_category subset from eaach category and stacks them into one array.

    Returns:
        data (numpy.ndarray): the data set
        Z (numpy.ndarray): the data labels
    """
    n = opt_data[0].shape[0]
    data = None
    Z = []
    for i, X in enumerate(opt_data):
        rows = random.sample([idx for idx in range(n)], n_per_category)
        data = X[rows] if i == 0 else np.vstack((data, X[rows]))
        Z += [i]*n_per_category
    return data, Z


# __________________________________________________________________________________________________
# Algorithms w/ known number of clusters
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def __eval_kmeans(data, labels, num_clusters):
    """
    Cluster data into num_clusters cluster. Returns a n-length array of cluster assignments.
    """
    km = KMeans(n_clusters=num_clusters)
    km.fit(data)
    Z = km.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari


def __eval_gmm(data, labels, num_components):
    """
    Expectation-maximization with a given number of categories
    """
    gmm = GMM(n_components=num_components, n_iter=100)
    gmm.fit(data)
    Z = gmm.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari


# __________________________________________________________________________________________________
# Algorithms w/ labeled data
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def __eval_logit(train_data, train_labels, test_data, test_labels):
    """
    Multinomial logistic regression. 
    Trained on train_data, tested on test_data. Returns ARI(test_labels, Z)
    """
    lr = LogisticRegression()
    lr.fit(train_data, train_labels)
    Z = lr.predict(test_data)
    ari = metrics.adjusted_rand_score(test_labels, Z)
    return ari


def __eval_svm(train_data, train_labels, test_data, test_labels, kernel='linear'):
    """
    Support-vector machine with user-specified kernel (linear, rbf, etc)
    Trained on train_data, tested on test_data. Returns ARI(test_labels, Z)
    """
    vm = svm.SVC(kernel=kernel)
    vm.fit(train_data, train_labels)
    Z = vm.predict(test_data)
    ari = metrics.adjusted_rand_score(test_labels, Z)
    return ari


# __________________________________________________________________________________________________
# Variational algorithms
# These algorithms must be given a max number of categories, then they choose which categories
# to keep around. We need to choose high, but if we choose too high, we have to wait froever.
# I chose n/k because I want to build in a little as possible.
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def __eval_vbgmm(data, labels, alpha=1.0):
    """
    Variational inferrence for Gaussian mixture models. Based on dirichlet process.
    Returns ARI(test_labels, Z)
    """
    # k = int(data.shape[0]/float(max(labels)))
    k = 24
    vbgmm = VBGMM(alpha=alpha, n_iter=10000, n_components=k, covariance_type='full')
    vbgmm.fit(data)
    Z = vbgmm.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    # import pdb; pdb.set_trace()
    return ari


def __eval_dpgmm(data, labels, data_model, alpha=1.0):
    """
    Dirichlet proccess EM for Gaussian mixture models. 
    Returns ARI(test_labels, Z)
    """
    dpgmm = PyDPGMM(data_model, data, crp_alpha=alpha, init_mode='single_cluster')
    Z = dpgmm.fit(n_iter=200, sm_prop=.25, sm_burn=50, num_sm_sweeps=2)
    # dpgmm = DPGMM(data_model, data, crp_alpha=alpha, init_mode='single_cluster')
    # Z = dpgmm.fit(n_iter=200, sm_prop=.8, sm_burn=10)
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari

# def __eval_dpgmm(data, labels, data_model, alpha=1.0):
#     """
#     Dirichlet proccess EM for Gaussian mixture models. 
#     Returns ARI(test_labels, Z)
#     """
#     data_cpy = np.copy(data)
#     data_cpy -= np.mean(data_cpy, axis=0)
#     data_cpy /= np.max(data_cpy)
#     dpgmm = DPGMM_sk(n_components=min(20, data_cpy.shape[0]), alpha=1.0,  n_iter=10000)
#     dpgmm.fit(data_cpy)
#     Z = dpgmm.predict(data_cpy)
#     ari = metrics.adjusted_rand_score(labels, Z)
#     return ari


def plot_result(filename, type='kde', suptitle=None, base_filename=None):
    data = pickle.load(open(filename, 'rb'))
    N = [key for key in data.keys()]
    N = sorted(N)
    
    n_sbplts_x = 3
    n_sbplts_y = 2

    def get_clip(x):
        s = np.std(x)
        m = np.mean(x)
        return [m-4.0*s, m+4.0*s]

    for i, n in enumerate(N):
        # f, axes = plt.subplots(n_sbplts_y, n_sbplts_x, sharex=True, sharey=True, figsize=(4, 8))
        f, axes = plt.subplots(n_sbplts_y, n_sbplts_x, figsize=(20, 12))
        f.set_facecolor('white')
        c1, c2, c3 = sns.color_palette("Set1", 3)

        for j, alg in enumerate(algorithm_list):
            ari_orig = data[n]['ari_original'][alg]
            ari_opt = data[n]['ari_optimal'][alg]

            if type == 'kde':
                clip_orig = get_clip(ari_orig)
                clip_opt = get_clip(ari_opt)

                # sns.kdeplot(ari_orig, shade=True, color=c1, ax=axes.flat[j], clip=clip_orig, label='original')
                # sns.kdeplot(ari_opt, shade=True, color=c2, ax=axes.flat[j], clip=clip_opt, label='optimal')
                sns.distplot(ari_orig, color=c1, ax=axes.flat[j], 
                    kde_kws=dict(clip=clip_orig, label='original', lw=3),
                    hist_kws=dict(histtype="stepfilled"))
                sns.distplot(ari_opt, color=c2, ax=axes.flat[j],
                    kde_kws=dict(clip=clip_opt, label='optimized', lw=3),
                    hist_kws=dict(histtype="stepfilled"))

                axes.flat[j].set_xlabel('ARI')
                axes.flat[j].set_ylabel('Density')
            elif type == 'violin':
                sns.violinplot([ari_orig, ari_opt], positions=[1, 2], ax=axes.flat[j],
                    names=['original','optimized'], alpha=.5)

                axes.flat[j].set_ylabel('ARI')

            axes.flat[j].set_title(alg.upper())

        if suptitle is not None:
            plt.suptitle(suptitle)
        else:
            plt.suptitle("N=%i" % n)

        if base_filename:
            filename = base_filename + "_" + str(s) + "n.png"
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

# _________________________________________________________________________________________________
# Entry point
# `````````````````````````````````````````````````````````````````````````````````````````````````
if __name__ == '__main__':
    import argparse
    from idsteach.teacher import Teacher
    from idsteach.models import NormalInverseWishart
    import numpy as np

    parser = argparse.ArgumentParser(description='Run examples')
    parser.add_argument('--num_examples', metavar='N', type=int, nargs='+',help='list of number of exampler per phoneme')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of runs to average over.')
    parser.add_argument('--plot_type', type=str, default='kde', help="type of plot 'kde' (default) or 'violin'")
    parser.add_argument('--filename', type=str, default='alcomptest.pkl', help='save as filename')
    parser.add_argument('--multirun', action='store_true', default=False, help='use data from multiple sampler chains')
    parser.add_argument('--base_figname', type=str, default='alcomptest', help='save figure as filename')

    args = parser.parse_args()

    target_model, labels = ids.gen_model()
    data_model = NormalInverseWishart.with_vague_prior(target_model)
    if args.multirun:
        dirname = '../data'
        data = utils.multiple_matlab_csv_to_teacher_data(dirname)
    else:
        dirname = os.path.join('../data', 'lrunml')
        data = utils.matlab_csv_to_teacher_data(dirname)
    
    algcomp(data, target_model, data_model, args.num_examples, args.num_runs, filename=args.filename)
    plot_result(args.filename, args.plot_type, base_filename=args.figname)
