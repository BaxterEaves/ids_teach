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

from math import log
import numpy as np
import random
import copy
import sys

from scipy.misc import logsumexp
from idsteach.utils import lpflip
from idsteach.utils import lcrp
from idsteach.utils import crp_gen

def check_partition_integrity(Z, Nk):
    """
    Makes sure the the partition information is healthy and each piece agrees.
    """
    errors = ''
    # if K != len(Nk):
    #     errors += 'number of bins in Nk ({0}) is not equal to K ({1})\n'.format(Nk, K)
    if 0 in Nk:
        errors += 'Nk ({}) has a zero entry.\n'.format(Nk)
    if len(Z) != sum(Nk):
        errors += 'Nk ({0}) should have a count for every element in Z ({1}).\n'.format(Z, Nk)
    if max(Z) != len(Nk)-1:
        errors += 'max Z ({0}) should be one less than the number of bins in Nk ({1}).\n'.format(max(Z), Nk)
    for k, num_k in enumerate(Nk):
        if len(np.nonzero(Z==k)[0]) != num_k:
            errors += 'Z ({0}), does not add to Nk ({1})\n'.format(Z, Nk)
            break

    if errors:
        print(errors)
        return False
    else:
        return True


class DPGMM(object):
    """
    A CPR Gaussian mixture model using Gibbs sweeps.
    """

    def __init__(self, data_model, data, crp_alpha=1.0, init_mode='from_prior'):
        self.data_model = data_model
        self.data = data
        self.crp_alpha = crp_alpha 

        self._n = data.shape[0]
        self._rows = [i for i in range(self._n)]

        if init_mode == 'from_prior':
            self.Z, self.Nk, self.K = crp_gen(self._n, self.crp_alpha)
        elif init_mode == 'single_cluster':
            self.Z = np.zeros(self._n, dtype=np.dtype(int))
            self.Nk = [self._n]
            self.K = 1

    def score_data(self, Z):
        log_score = 0
        for k in range(max(Z)+1):
            X_k = self.data[np.nonzero(Z == k)[0], :]
            log_score += self.data_model.log_marginal_likelihood(X_k)
        return log_score

    def __restricted_init(self, Z, Nk, k1, k2, sweep_indices):
        
        idx_1 = [k1]
        idx_2 = [k2]

        Nk[k1] = 1
        Nk[k2] = 1

        for row in sweep_indices:
            Y = self.data[row, :]
            X_1 = self.data[idx_1, :]
            X_2 = self.data[idx_2, :]
            try:
                lp_k1 = log(float(Nk[k1])) + self.data_model.log_posterior_predictive(Y, X_1)
                lp_k2 = log(float(Nk[k2])) + self.data_model.log_posterior_predictive(Y, X_2)
            except:
                import pdb; pdb.set_trace()
            
            C = logsumexp([lp_k1, lp_k2])
            
            if log(random.random()) < lp_k1 - C:
                Nk[k1] += 1
                Z[row] = k1
                idx_1 = np.append(idx_1, [row])
            else:
                Nk[k2] += 1
                Z[row] = k2
                idx_2 = np.append(idx_2, [row])

        assert check_partition_integrity(Z, Nk), '__restricted_init: partition check failed'
        return Z, Nk

    def __restricted_gibbs_sweep(self, Z, Nk, k1, k2, sweep_indices, Z_final=None):
        """
        FIXME: DOCSTRING
        """
        lp_split = 0.0

        hypothetical = (Z_final is not None)

        for row in sweep_indices:
            Y = self.data[row, :]
            k_a = Z[row]

            if k_a == k1:
                lcrp_1 = log(Nk[k1]-1.0)
                lcrp_2 = log(float(Nk[k2]))
            else:
                lcrp_1 = log(float(Nk[k1]))
                lcrp_2 = log(Nk[k2]-1.0)

            idx_k1 = [i for i in np.nonzero(Z == k1)[0] if i != row]
            idx_k2 = [i for i in np.nonzero(Z == k2)[0] if i != row]

            lp_k1 = self.data_model.log_posterior_predictive(Y, self.data[idx_k1, :]) + lcrp_1
            lp_k2 = self.data_model.log_posterior_predictive(Y, self.data[idx_k2, :]) + lcrp_2

            C = logsumexp([lp_k1, lp_k2])

            if hypothetical:
                k_b = Z_final[row]
            else:
                k_b = k1 if log(random.random()) < lp_k1 - C else k2

            if k_a != k_b:
                Z[row] = k_b
                Nk[k_a] -= 1
                Nk[k_b] += 1


            if k_b == k1:
                lp_split += lp_k1 - C
            else:
                lp_split += lp_k2 - C

        assert check_partition_integrity(Z, Nk), '__restricted_gibbs_sweep: partition check failed'

        if not hypothetical:
            return Z, Nk, lp_split
        else:
            return lp_split


    def __transition_row_gibbs(self, row):
        """
        Run a gibbs sweep on a given row

        TODO: Explain gibbs

        Inputs:
            row (int): The row to transition
        """
        Y = self.data[row, :]
        k_a = self.Z[row]
        is_singleton = (self.Nk[k_a] == 1)

        ps = np.copy(self.Nk)
        if not is_singleton:
            ps = np.append(ps, [self.crp_alpha])
        else:
            ps[k_a] = self.crp_alpha
        ps = np.log(ps)
        
        if not is_singleton:
            idx = [i for i in np.nonzero(self.Z == k_a)[0].tolist() if i != row]
            ps[k_a] += self.data_model.log_posterior_predictive(Y, self.data[idx, :])
        else:
            ps[k_a] += self.data_model.log_marginal_likelihood(Y)

        for k_b in range(self.K):
            if k_b != k_a:
                idx = np.nonzero(self.Z == k_b)[0].tolist()
                X = self.data[idx, :]
                ps[k_b] += self.data_model.log_posterior_predictive(Y, X)

        if not is_singleton:
            ps[-1] += self.data_model.log_marginal_likelihood(Y)

        k_b = lpflip(ps)
        assert k_b <= self.K, '__transition_row_gibbs: The proposed component exceeds self.K'

        # cleanup
        if k_b != k_a:
            self.Z[row] = k_b
            if is_singleton:
                self.Nk[k_b] += 1
                del self.Nk[k_a]
                # keep Z ordered and gapless
                self.Z[np.nonzero(self.Z >= k_a)] -= 1
            else:
                self.Nk[k_a] -= 1
                if k_b == self.K:
                    self.Nk.append(1)
                else:
                    self.Nk[k_b] += 1

            self.K = len(self.Nk)

        assert check_partition_integrity(self.Z, self.Nk),\
            '__transition_row_gibbs: partition check failed'

    def __transition_split_merge(self, num_sm_sweeps=5):
        """
        Split-merge transition kernel

        TODO: split-merge

        Inputs:
            num_sm_sweeps (int): Number of restricted gibbs sweeps to do for split-merge proposals
        """
        # pick two different rows
        row_1, row_2 = random.sample(range(self._n), 2)

        k1 = self.Z[row_1]
        k2 = self.Z[row_2]

        sweep_indices = [i for i in range(self._n) if (self.Z[i] in [k1, k2]) and (i not in [row_1, row_2])]

        # create launch state
        Z_launch = copy.copy(self.Z)
        Nk_launch = copy.copy(self.Nk)

        k1_launch = k1
        if k1 == k2:
            k1_launch = self.K
            Z_launch[row_1] = k1_launch
            Nk_launch[k1] -= 1
            Nk_launch.append(1)

        k2_launch = k2

        assert 0 not in Nk_launch, '__transition_split_merge: Zero entry in Nk_launch'

        Z_launch, Nk_launch = self.__restricted_init(Z_launch, Nk_launch, k1_launch, k2_launch,
                                                     sweep_indices)
        Q_launch = 0
        Q = 0
        for sweep in range(num_sm_sweeps):
            Z_launch, Nk_launch, Q = self.__restricted_gibbs_sweep(Z_launch, Nk_launch, k1_launch,
                                                                   k2_launch, sweep_indices)

        if k1 == k2:
            # split proposal

            L = self.score_data(Z_launch)-self.score_data(self.Z)
            P = lcrp(self._n, Nk_launch, self.crp_alpha) - lcrp(self._n, self.Nk, self.crp_alpha)

            if log(random.random()) < P+L-Q:
                self.Z = Z_launch
                self.Nk = Nk_launch
                self.K = len(Nk_launch)

            assert check_partition_integrity(self.Z, self.Nk),\
                '__transition_split_merge (split): partition check failed'
        else:
            # merge proposal
            Z_merge = copy.copy(self.Z)
            Nk_merge = copy.copy(self.Nk)
            k1_merge = k2
            k2_merge = k2

            Z_merge[row_1] = k2
            for idx in sweep_indices:
                Z_merge[idx] = k2

            Nk_merge[k2] += Nk_merge[k1]
            del Nk_merge[k1]
            Z_merge[np.nonzero(Z_merge > k1)[0]] -= 1

            # if k2_merge > k1:
            #     k2_merge -= 1
    
            Q = self.__restricted_gibbs_sweep(Z_launch, Nk_launch, k1_launch, k2_launch,
                                              sweep_indices, Z_final=self.Z)

            L = self.score_data(Z_merge)-self.score_data(self.Z)
            P = lcrp(self._n, Nk_merge, self.crp_alpha) - lcrp(self._n, self.Nk, self.crp_alpha)

            if log(random.random()) < P+Q+L:
                self.Z = Z_merge
                self.Nk = Nk_merge
                self.K = len(Nk_merge)

                assert check_partition_integrity(self.Z, self.Nk),\
                    '__transition_split_merge (merge accept): partition check failed'
            else:
                assert check_partition_integrity(self.Z, self.Nk),\
                    '__transition_split_merge (merge reject): partition check failed'

    def __transition(self, sm_prop=0.0, num_sm_sweeps=5):
        """ 
        Transition the component assignment. 

        Kwargs:
            sm_prop (float<0,1>):  proportion of iterations that use split-merge
            num_sm_sweeps (int): Number of restricted gibbs sweeps to do for split-merge proposals
        """
        if random.random() < sm_prop:
            self.__transition_split_merge(num_sm_sweeps)
        else:
            random.shuffle(self._rows)
            for row in self._rows:
                self.__transition_row_gibbs(row)

    def fit(self, n_iter, sm_prop=.1, num_sm_sweeps=5, sm_burn=10):
        """
        Runs the sampler

        Inputs:
            n_iter (float): number of iterations to run

        Kwargs:
            sm_prop (float<0,1>): proportion of iterations that use split-merge
            num_sm_sweeps (int): Number of restricted gibbs sweeps to do for split-merge proposals
            sm_burn (int): Use split-merge for the first sm_burn iterations
        """
        # init from the prior
        for i in range(n_iter):
            do_sm_prop = 1.0 if i < sm_burn else sm_prop
            self.__transition(sm_prop=do_sm_prop, num_sm_sweeps=num_sm_sweeps)

        return self.Z

    def predict(self, X):
        X = np.array(X)
        if X.shape[1] != self.data.shape[1]:
            raise ValueError('X must have the dimensionality as the training data.')

        log_crp = np.log(np.array(self.Nk))-log(self._n)

        n_X = X.shape[0]
        Z_X = np.zeros(n_X, dtype=np.dtype(int))

        for row in range(n_X):
            x = X[row, :]
            ps = np.copy(log_crp)
            for k in range(self.K):
                Y = self.data[np.nonzero(self.Z==k)[0], :]
                ps[k] += self.data_model.log_posterior_predictive(x, Y)
            Z_X[row] = np.argmax(ps)

        return Z_X

    def score(self, X):
        """ log predictive probability of each X under the model """
        X = np.array(X)
        if X.shape[1] != self.data.shape[1]:
            raise ValueError('X must have the dimensionality as the training data.')

        n_X = X.shape[0]

        log_crp = np.log(np.array(self.Nk + [self.crp_alpha]))-log(self._n+self.crp_alpha)
        log_ps = np.zeros((n_x, self.K+1))

        for k in range(self.K):
            log_ps[:, k] = log_crp[k]
            Y = self.data[np.nonzero(self.Z==k)[0], :]
            for row in range(n_X):
                x = X[row, :]
                log_ps[row, k] += self.data_model.log_posterior_predictive(x, Y)

        # must account for probability of X being in a singleton
        for row in range(n_X):
            x = X[row, :]
            log_ps[row, -1] += self.data_model.log_marginal_likelihood(x) + log_crp[-1]

        return logsumexp(log_ps, axis=1)

