from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.mixture import GMM
from sklearn.mixture import VBGMM
from sklearn.mixture import DPGMM

import pandas as pd
from sklearn import metrics
from math import log

import pickle
import random
import sys

algorithm_list = ['logit', 'svm_linear', 'svm_rbf', 'k_means', 'gmm', 'vbgmm', 'dpgmm']


def algcomp(opt_data, target_model, data_model, n_per_cat, n_times, filename=None):
    """
    TODO: docstring
    """
    data_model.validate_target_model(target_model)

    columns = ['mean_{ADS}', 'mean_{OPT}', 'std_{ADS}', 'std_{OPT}']
    keys = algorithm_list

    ari_orig_mean = __prep_result_holder(len(n_per_cat))
    ari_orig_std = __prep_result_holder(len(n_per_cat))
    ari_opt_mean = __prep_result_holder(len(n_per_cat))
    ari_opt_std = __prep_result_holder(len(n_per_cat))

    df_output = dict()

    for j, n_per in enumerate(n_per_cat):
        print("Running evaluations for {} examples per category.".format(n_per))
        ari_orig_n = __prep_result_holder(n_times)
        ari_opt_n = __prep_result_holder(n_times)

        for i in range(n_times):
            ari_orig_res, ari_opt_res = eval_all(opt_data, target_model, data_model, n_per)
            # print("--run {0} or {1} complete".format(i+1, n_times), end=" ")
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
        df_output[n_per] = df
        # print(df.to_latex())
        print(df)

    if filename is not None:
        pickle.dump(df_output, open(filename, "wb"))


def eval_all(opt_data, target_model, data_model, n_per_cat):
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
    print(".", end=" ")
    sys.stdout.flush()
    data_train_orig, z_train_orig = __prep_standard_data(target_model, data_model, n_per_cat)
    data_test_orig, z_test_orig = __prep_standard_data(target_model, data_model, n_per_cat)
    aris_orig = eval_set(data_train_orig, z_train_orig, data_test_orig, z_test_orig)

    print("`", end=" ")
    sys.stdout.flush()
    data_train_opt, z_train_opt = __prep_optimal_data(opt_data, n_per_cat)
    data_test_opt, z_test_opt = __prep_optimal_data(opt_data, n_per_cat)
    aris_opt = eval_set(data_train_opt, z_train_opt, data_test_opt, z_test_opt)

    return aris_orig, aris_opt

def eval_set(train_data, train_labels, test_data, test_labels):
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
    ari_k_means = eval_kmeans(train_data, train_labels, K)
    ari_gmm = eval_gmm(train_data, train_labels, K)

    # variational algorithms
    ari_vbgmm = eval_vbgmm(train_data, train_labels, alpha=1.0)
    ari_dpgmm = eval_dpgmm(train_data, train_labels, alpha=1.0)

    # labeled-set algorithms
    ari_logit = eval_logit(train_data, train_labels, test_data, test_labels)
    ari_svm_linear = eval_svm(train_data, train_labels, test_data, test_labels, kernel='linear')
    ari_svm_rbf = eval_svm(train_data, train_labels, test_data, test_labels, kernel='rbf')

    ari = {
        'logit': ari_logit,
        'svm_linear': ari_svm_linear,
        'svm_rbf': ari_svm_rbf,
        'k_means': ari_k_means,
        'gmm': ari_gmm,
        'vbgmm': ari_vbgmm,
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
def eval_kmeans(data, labels, num_clusters):
    """
    Cluster data into num_clusters cluster. Returns a n-length array of cluster assignments.
    """
    km = KMeans(n_clusters=num_clusters)
    km.fit(data)
    Z = km.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari


def eval_gmm(data, labels, num_components):
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
def eval_logit(train_data, train_labels, test_data, test_labels):
    """
    Multinomial logistic regression. 
    Trained on train_data, tested on test_data. Returns ARI(test_labels, Z)
    """
    lr = LogisticRegression()
    lr.fit(train_data, train_labels)
    Z = lr.predict(test_data)
    ari = metrics.adjusted_rand_score(test_labels, Z)
    return ari


def eval_svm(train_data, train_labels, test_data, test_labels, kernel='linear'):
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
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def eval_vbgmm(data, labels, alpha=1.0):
    """
    Variational inferrence for Gaussian mixture models. Based on dirichlet process.
    Returns ARI(test_labels, Z)
    """
    # These algorithms must be given a max number of categories, then they choose which categories
    # to keep around. We need to choose high, but if we choose too high, we have to wait froever.
    # I chose n/k because I want to build in a little as possible.
    k = int(data.shape[0]/float(max(labels)))
    vbgmm = VBGMM(alpha=alpha, n_iter=100, n_components=k, covariance_type='full')
    vbgmm.fit(data)
    Z = vbgmm.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    # import pdb; pdb.set_trace()
    return ari


def eval_dpgmm(data, labels, alpha=1.0):
    """
    Dirichlet proccess EM for Gaussian mixture models. 
    Returns ARI(test_labels, Z)
    """
    k = int(data.shape[0]/float(max(labels)))
    dpgmm = DPGMM(alpha=alpha, n_iter=100, n_components=10)
    dpgmm.fit(data)
    Z = dpgmm.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari


# _________________________________________________________________________________________________
# Entry point
# `````````````````````````````````````````````````````````````````````````````````````````````````
if __name__ == '__main__':
    from ids_teach.teacher import Teacher
    from ids_teach.models import NormalInverseWishart
    import numpy as np

    # FIXME: replace w/ actual data when we have it
    cov_a = np.eye(2)*.4
    cov_b = np.eye(2)*.4
    cov_c = np.eye(2)*.4
    mean_a = np.array([-2.0, 0.0])
    mean_b = np.array([2.0, 0.0])
    mean_c = np.array([0.0, 1.6])

    target_model = {
        'd': 2,
        'parameters': [
            (mean_a, cov_a),
            (mean_b, cov_b),
            (mean_c, cov_c),
            ],
        'assignment': np.array([0, 1, 2], dtype=np.dtype(int))
    }

    data_model = NormalInverseWishart.with_vague_prior(target_model)

    n = 500
    t = Teacher(target_model, data_model, crp_alpha=1.0, t_std=1, fast_niw=True)
    t.mh(n, burn=500, lag=10, plot_diagnostics=False)

    algcomp(t.data, target_model, data_model, [10, 25, 50, 100], 100, filename='alcomp_test.pkl')