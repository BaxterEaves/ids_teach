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

import numpy as np
import multiprocessing as mp

from idsteach import utils
from idsteach import models
import pickle
import copy

from idsteach.bhc import bhc
from idsteach import fastniw as fniw
from progressbar import ProgressBar
from progressbar import ETA
from progressbar import RotatingMarker
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
import seaborn as sns


NIW_ML_ENUMERATE = 0
NIW_ML_BHC = 1
NIW_ML_PGIBBS = 2

APPROX_KEY = {'bhc': NIW_ML_BHC, 'pgibbs': NIW_ML_PGIBBS}


class Teacher(object):
    """
    Teacher produces data optimized to teach a target categorization model to a
    naive learner (under infinite Gaussian mixture model).

    Attributes
    ----------
    _use_mp : bool
        calculate moraginal likelihood w/ using multiprocessing
    _num_procs : int
        number of processors avalibe for multiprocessing
    t_std : float
        Proposal distribution standard deviation. Data is jittered via Gaussian
        noise i.e. X += randn(X.shape)*t_std
    target : dict
        The model the teacher teaches.
    data_model : CollapsibleDistribution
        The model the data follow
    crp_alpha : float
        Concentration parameter for CRP
    """
    def __init__(self, target, data_model, crp_alpha, t_std=50, use_mp=False,
                 fast_niw=True, approx=False, yes=False):
        """
        FIXME: Fill in
        """
        n = len(target['assignment'])
        if n > 12 and not (yes or approx):
            raise ValueError("**WARNING: n is very large. Exiting. Bypass "
                             "with yes=True")

        # TODO: More input validation
        if not isinstance(data_model, models.CollapsibleDistribution):
            raise TypeError('data_model should be type '
                            'models.CollapsibleDistribution')

        data_model.validate_target_model(target)

        self._use_mp = use_mp
        self._num_procs = mp.cpu_count()
        self._fast_niw = fast_niw
        if approx is not False:
            self._approx = APPROX_KEY[approx.lower()]

        if self._use_mp and self._approx:
            print('No multiprocessing needd for approximation. '
                  'setting use_mp=False.')
            self._use_mp = False

        if n < 6 and self._use_mp:
            print("**WARNING: The target conditions favor serial execution. "
                  " Switching off multiprocessing.")
            self._use_mp = False

        self.t_std = t_std
        self.target = target
        self.data_model = data_model
        self.crp_alpha = crp_alpha
        self._d = target['d']
        self._n = n

        # Set up the partitons for work-sharing. Colelct partition seeds at
        # equal interval for each thread so each thread can generate its own
        # partitions.
        # NOTE: this is not the best way to do this because the latter
        # partition will have more categories which will increase the run time
        # of _log_marginal and _log_crp. It remains an emperical questions as
        # to whther this is faster than passing arround shuffled lists to
        # each processor. My huch is that the i/o hit would be worse---hence
        # why i did it this way...who knows though?
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
            self._num_procs = 1
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

        # generate some random start data from the original model
        self.X = np.zeros((self._n, self._d))
        for i, k in enumerate(self.target['assignment']):
            mu_k, sigma_k = self.target['parameters'][k]
            self.X[i, :] = np.random.multivariate_normal(mu_k, sigma_k)
        self.logp = self.evaluate_probability(self.X)

    def evaluate_probability(self, X):
        """
        Evaluate the probability of the data, X, given the optimal IGMM teacher
        model
        """
        # in python 3, map returns a generator--we use a lambda to make it a
        # list
        if self._use_mp:
            mapper = self.pool.map
        else:
            mapper = lambda f, args: [res for res in map(f, args)]

        numer = 0
        for z, theta in enumerate(self.target['parameters']):
            Y = X[np.nonzero(self.target['assignment'] == z)[0], :]
            # the prior cancels out in the acceptance ratio so there is no need
            # to calculate it.
            numer += self.data_model.log_likelihood(Y, *theta)
            # \ + self.data_model.log_prior(*theta)
        numer += self.p_crp_true

        args = []
        if self._fast_niw:
            for i in range(self._num_procs):
                args.append((X, self.data_model.lambda_0, self.data_model.mu_0,
                             self.data_model.kappa_0, self.data_model.nu_0,
                             self.crp_alpha, self.zs[i], self.ks[i],
                             self.hs[i], self.ns[i]))
            to_sum = mapper(fniw.niw_mmml_mp, args)

            denom = logsumexp(to_sum)
        elif self._approx == NIW_ML_BHC:
            _, denom = bhc(X, self.data_model, self.crp_alpha)
        elif self._approx == NIW_ML_PGIBBS:
            denom = fniw.pgibbs_estimator(X, self.data_model, self.crp_alpha,
                                          n_samples=1000)
        else:
            for i in range(self._num_procs):
                args.append((self.data_model, X, copy.copy(self.zs[i]),
                             copy.copy(self.ks[i]), copy.copy(self.hs[i]),
                             self.crp_alpha, self.ns[i]))
            sum_parts = mapper(models.mp_log_marginals, args)
            to_sum = np.zeros(self._num_procs)
            for i, part in enumerate(sum_parts):
                to_sum[i] = logsumexp(part)

            denom = logsumexp(to_sum)

        return numer-denom

    def mh(self, n, burn=200, lag=50, plot_diagnostics=False,
           datum_jitter=False):
        """
        Use Metropolis-Hastings (MH) sampling to generate data samples

        Parameters
        ----------
        n : int
            Number of samples to collect.
        burn : int
            Number of initial samples to disregard.
        lag : int
            Number of transitions to perform between sample collections.
        plot_diagnostics : bool , optional
            Do real-time plot of data and scores over time.
        datum_jitter : bool, optional
            If False (default), at each iteration, the proposal distribution
            perturbs the entire dataset; otherwise perturbs a single randomly-
            selected datum.
        """
        total_iters = 0
        widgets = [ETA(), ' ', RotatingMarker()]
        pbar = ProgressBar(widgets=widgets, maxval=burn+lag*n)
        pbar.start()
        for _ in range(burn):
            # sample new data
            X_prime = utils.jitter_data(np.copy(self.X), self.t_std,
                                        datum_jitter)
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
                X_prime = utils.jitter_data(np.copy(self.X), self.t_std,
                                            datum_jitter)
                logp_prime = self.evaluate_probability(X_prime)

                if np.log(np.random.rand()) < logp_prime - self.logp:
                    self.__accept_data(X_prime, logp_prime)
                total_iters += 1
                pbar.update(total_iters)

            self.__collect_data()
            # self.__calibrate_proposal_distribution()

            if plot_diagnostics and itr > 1:
                plt.clf()
                plt.subplot(1, 2, 1)
                utils.plot_data_2d(self.data, self.target)
                plt.subplot(1, 2, 2)
                plt.plot(self.logps)
                plt.draw()

        pbar.finish()

    def clear_data(self):
        """
        Clear collected data and associated logps
        """
        self.data = []
        self.logps = []

    def get_stacked_data(self):
        """
        Returns the data as a single numpy array along with a n*k-length list
        labeling each datum
        """
        Z = [0]*self._n
        data = np.copy(self.data[0])

        for i in range(1, len(self.data)):
            Z += [i]*self._n
            data = np.vstack((data, np.copy(self.data[i])))

        return data, Z

    def save(self, filename, labels=None):
        """Dump the main data to a pickle"""
        to_save = {
            'data': self.data,
            'target': self.target,
            'labels': labels,
            'X': self.X,
            'logp': self.logp
        }
        pickle.dump(to_save, open(filename, "wb"))

    def load(self, filename, labels=None):
        """
        Load in data and resume from previous state.
        """
        raise NotImplementedError

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
                    self.data[k] = np.vstack((self.data[k],
                                              np.copy(self.X[i, :])))
        else:
            for i, k in enumerate(self.target['assignment']):
                self.data[k] = np.vstack((self.data[k], np.copy(self.X[i, :])))

        self.logps.append(self.logp)

    def __calibrate_proposal_distribution(self):
        """
        Adjust jump standard deviation so that the acceptance rate will stay in
        an acceptable range.
        """
        raise NotImplementedError
