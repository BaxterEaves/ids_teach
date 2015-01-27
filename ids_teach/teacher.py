import numpy as np
import multiprocessing as mp

from ids_teach import utils
from ids_teach import models
import copy
import time

from ids_teach.cppext import niw_module as niwm
from progressbar import ProgressBar
from progressbar import ETA
from progressbar import RotatingMarker
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
import seaborn as sns


# _________________________________________________________________________________________________
# `````````````````````````````````````````````````````````````````````````````````````````````````
class Teacher(object):
    '''
    Teacher produces data optimized to teach a target categorization model to a
    naive learner (under infinite Gaussian mixture model).

    Attributes:
        _use_mp (bool): calculate moraginal likelihood w/ using multiprocessing
        _num_procs (int): number of processors avalibe for multiprocessing
        t_std (float): Proposal distribution standard deviation. Data is jittered via Gaussian
            nosie i.e. X += randn(X.shape)*t_std
        target (dict): The model the teacher teaches.
        data_model (CollapsibleDistribution): The model the data follow
        crp_alpha (float): Concentration parameter for CRP
    '''
    def __init__(self, target, data_model, crp_alpha, t_std=50, use_mp=False, fast_niw=True):
        n = len(target['assignment'])
        if n > 12:
            raise ValueError("n is too large")

        # TODO: figure out why this doesn't work
        # if not isinstance(data_model, models.CollapsibleDistribution):
        #     raise TypeError("model ({}) bust be models.CollapsibleDistribution")

        # TODO: More input validation

        self._use_mp = use_mp
        self._num_procs = mp.cpu_count()
        self._fast_niw = fast_niw

        self.t_std = t_std
        self.target = target
        self.data_model = data_model
        self.crp_alpha = crp_alpha
        self._d = target['d']
        self._n = n

        # Set up the partitons for work-sharing. Colelct partition seeds at equal interval for each
        # thread so each thread can generate its own partitions.
        #   NOTE: this is not the best way to do this because the latter partition will have more
        # categories which will increase the run time of _log_marginal and _log_crp. It remains an
        # emperical questions as to whther this is faster than passing arround shuffled lists to
        # each processor. My huch is that the i/o hit would be worse---hence why i did it this way.
        # Who knows though.
        k = np.zeros(n, dtype=np.dtype(int))
        z = np.zeros(n, dtype=np.dtype(int))
        h = [float(self._n)]
        bn = utils.bell_number(self._n)
        if self._use_mp:
            pbar = ProgressBar(maxval=bn)
            print('Pre-generating partitions for multiprocessing...')
            self.pool = mp.Pool()
            div = round(bn/self._num_procs)
            dividers = [div*r for r in range(self._num_procs)] + [-1]

            self.ks = []
            self.zs = []
            self.hs = []
            self.ns = [div]*(self._num_procs-1) + [bn-div*(self._num_procs-1)]

            i = 0
            pbar.start()
            while True:
                if i == dividers[0]:
                    self.ks.append(np.copy(k))
                    self.zs.append(np.copy(z))
                    self.hs.append(copy.copy(h))
                    del dividers[0]

                Z_k_h = utils.next_partition(z, k, h)

                if Z_k_h is None:
                    break
                Z, k, h = Z_k_h
                i += 1
                pbar.update(i)
            pbar.finish()
            print('Done.')
        else:
            self.ks = [k]
            self.zs = [z]
            self.hs = [h]
            self.ns = [bn]
            self.pool = None

        # set up acceptance ratio and such
        self.data = None
        self.logps = []
        self.acceptance = []

        H = np.zeros(len(self.target['parameters']))
        for z in self.target['assignment']:
            H[z] += 1
        self.p_crp_true = models.do_log_crp(H, self._n, self.crp_alpha)

        # generate som random start data
        self.X = data_model.draw_data_from_prior(self._n)
        self.logp = self.evaluate_probability(self.X)

    def evaluate_probability(self, X):
        """
        Evaluate the probability of the data, X, given the optimal IGMM teacher model
        """

        numer = 0

        for z, theta in enumerate(self.target['parameters']):
            Y = X[np.nonzero(self.target['assignment'] == z)[0], :]
            numer += self.data_model.log_likelihood(Y, *theta) + self.data_model.log_prior(*theta)
        numer += self.p_crp_true

        if self._use_mp:
            # TODO: abstract out do_marginal calculation
            args = []
            if self._fast_niw:
                for i in range(self._num_procs):
                    args.append((X, self.data_model.lambda_0, self.data_model.mu_0,
                                 self.data_model.kappa_0, self.data_model.nu_0, self.crp_alpha,
                                 self.zs[i], self.ks[i], self.hs[i], self.ns[i]))
                to_sum = self.pool.map(niwm.niw_mmml_mp, args)
            else:
                for i in range(self._num_procs):
                    args.append((self.data_model, X, self.zs[i], self.ks[i], self.hs[i],
                                 self.crp_alpha, self.ns[i]))

                sum_parts = self.pool.map(models.mp_log_marginals, args)
                to_sum = np.zeros(self._num_procs)
                for i, part in enumerate(sum_parts):
                    to_sum[i] = logsumexp(part)

            denom = logsumexp(to_sum)

        else:
            if self._fast_niw:
                denom = niwm.niw_mmml(X, np.copy(self.data_model.lambda_0),
                                      np.copy(self.data_model.mu_0), float(self.data_model.kappa_0),
                                      float(self.data_model.nu_0), float(float(self.crp_alpha)),
                                      np.copy(self.zs[0]), np.copy(self.ks[0]), self.hs[0],
                                      self.ns[0])
            else:
                to_sum = models.do_log_marginals(self.data_model, X, np.copy(self.zs[0]),
                                                 np.copy(self.ks[0]),  copy.copy(self.hs[0]),
                                                 self.crp_alpha, self.ns[0])
                denom = logsumexp(to_sum)

        return numer-denom

    def mh(self, n, burn=200, lag=50, plot_diagnostics=False):
        """
        Use Metropolis-Hastings sampling to generate data samples
        """
        total_iters = 0
        widgets = [ETA(), ' ', RotatingMarker() ]
        pbar = ProgressBar(widgets=widgets, maxval=burn+lag*n)
        pbar.start()
        for _ in range(burn):
            # sample new data
            X_prime = utils.jitter_data(np.copy(self.X), self.t_std)
            logp_prime = self.evaluate_probability(X_prime)

            if np.log(np.random.rand()) < logp_prime - self.logp:
                self.__accept_data(X_prime, logp_prime)
            total_iters += 1
            pbar.update(total_iters)

        if plot_diagnostics:
            plt.figure(tight_layout=True, facecolor='white')
            plt.axis([0, 1000, 0, 1])
            plt.ion()
            plt.show()


        for itr in range(n):
            # sample new data
            for _ in range(lag):
                X_prime = utils.jitter_data(np.copy(self.X), self.t_std)
                logp_prime = self.evaluate_probability(X_prime)

                if np.log(np.random.rand()) < logp_prime - self.logp:
                    self.__accept_data(X_prime, logp_prime)
                total_iters += 1
                pbar.update(total_iters)

            self.__collect_data()
            self.__calibrate_proposal_distribution()

            if plot_diagnostics and itr > 1:
                plt.cla()
                utils.plot_data_2d(self.data, self.target)
                plt.draw()
                # time.sleep(0.05)

        pbar.finish()

    def clear_data(self):
        """
        Clear collected data and associated logps
        """
        self.data = []
        self.logps = []

    def __accept_data(self, X_prime, logp_prime):
        """
        Update data sample and logp given new values
        """
        self.X = X_prime
        self.logp = logp_prime

    def __collect_data(self):
        """
        Bin the current data and logp values.
        """
        if self.data is None:
            self.data = [None for _ in range(len(self.target['parameters']))]
            for i, k in enumerate(self.target['assignment']):
                if self.data[k] is None:
                    self.data[k] = np.copy(self.X[i, :])
                else:
                    self.data[k] = np.vstack((self.data[k], np.copy(self.X[i, :])))
        else:
            for i, k in enumerate(self.target['assignment']):
                self.data[k] = np.vstack((self.data[k], np.copy(self.X[i, :])))

        self.logps.append(self.logp)

    def __calibrate_proposal_distribution(self):
        """
        Adjust jump standard deviation so that the acceptance rate will stay in an acceptable range.
        """
        pass
