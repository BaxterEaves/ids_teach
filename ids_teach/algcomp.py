from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.mixture import GMM
from sklearn.mixture import VBGMM
from sklearn.mixture import DPGMM

from sklearn import metrics

import random


def eval_all(data, labels):

    # labeled data algorithm
    training_prop = .5
    ari_logit = eval_logit(data, labels, training_prop)
    ari_svm_linear = eval_svm(data, labels, training_prop, kernel='linear')
    ari_svm_rbf = eval_svm(data, labels, training_prop, kernel='rbf')

    # fixed-categoty algorithms
    K = max(labels)+1
    ari_k_means = eval_kmeans(data, labels, K)
    ari_gmm = eval_gmm(data, labels, K)

    # variational algorithms
    ari_vbgmm = eval_vbgmm(data, labels, alpha=1.0)
    ari_dpgmm = eval_dpgmm(data, labels, alpha=1.0)

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
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def _get_training_set(data, labels, training_prop):
    n = data.shape[0]
    n_train = round(n*training_prop)

    training_set_indices = random.sample([i for i in range(n), n_train])
    grading_set_indices = [i for i in range(n) if i not in training_set_indices]
    data_train = data[training_set_indices, :]
    training_labels = labels[training_set_indices]
    grading_labels = labels[grading_set_indices]
    data_grade = data[grading_set_indices, :]

    return data_train, training_labels, data_grade, grading_labels


# __________________________________________________________________________________________________
# Algorithms w/ known number of clusters
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def eval_kmeans(data, labels, num_clusters):
    """
    Cluster data into num_clusters cluster. Returns a n-length array of cluster assignments.
    """
    km = KMeans(k=num_clusters)
    km.fit()
    Z = km.predict()
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari


def eval_gmm(data, labels, num_components):
    gmm = GMM(n_components=num_components)
    gmm.fit(data)
    Z = gmm.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari


# __________________________________________________________________________________________________
# Algorithms w/ labeled data
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def eval_logit(data, labels, training_prop):
    """
    Predict labels (clusters) of the data given labels.

    Args:
        data (numpy.ndarray): an n by num_features data set
        labels (numpy.ndarray): an n-length array of object labels. labels[i] is the label of
            data[i]
        training_prop (float): the proportion of the data the model is trained on
    """
    data_t, labels_t, data_g, labels_g = _get_training_set(data, labels, training_prop)

    lr = LogisticRegression()
    lr.fit(data_t, labels_t)
    Z = lr.predict(data_g)
    ari = metrics.adjusted_rand_score(labels_g, Z)
    return ari


def eval_svm(data, labels, training_prop, kernel='linear'):
    data_t, labels_t, data_g, labels_g = _get_training_set(data, labels, training_prop)
    vm = svm.SVC(kernel=kernel)
    vm.fit(data_t, labels_t)
    Z = vm.predict(data_g)
    ari = metrics.adjusted_rand_score(labels_g, Z)
    return ari


# __________________________________________________________________________________________________
# Variational algorithms
# ``````````````````````````````````````````````````````````````````````````````````````````````````
def eval_vbgmm(data, labels, alpha=1.0):
    vbgmm = VBGMM(alpha=alpha)
    vbgmm.fit(data)
    Z = vbgmm.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari


def eval_dpgmm(data, labels, alpha=1.0):
    dpgmm = DPGMM(alpha=alpha)
    dpgmm.fit(data)
    Z = dpgmm.predict(data)
    ari = metrics.adjusted_rand_score(labels, Z)
    return ari
