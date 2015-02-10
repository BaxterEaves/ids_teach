import utils
import numpy as np
import random

from ids_teach.utils import lpflip


class DPGMM(object):
    """
    A non-terrible version of CPR Gaussian mixture models.
    """

    def __init__(self, data_model, data, crp_alpha=1.0):
        self.data_model = data_model
        self.data = data
        self.crp_alpha = crp_alpha 

        self._n = data.shape[0]
        self._rows = [i for i in range(self._n)]

        self.Z, self.Nk, self.K = utils.crp_gen(self._n, self.crp_alpha)

    def __transition(self):
        random.shuffle(self._rows)
        for row in self._rows:
            Y = self.data[row, :]
            k_a = self.Z[row]
            is_singleton = (self.Nk[k_a] == 1)

            ps = np.copy(self.Nk)
            if not is_singleton:
                ps = np.append(ps, [self.crp_alpha])
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
            assert k_b <= self.K

            # cleanup
            if k_b != k_a:
                self.Z[row] = k_b
                if is_singleton:
                    self.Nk[k_b] += 1
                    del self.Nk[k_a]
                    # keep Z ordered and gapless
                    self.Z[np.nonzero(self.Z >= k_a)] -= 1
                else:
                    if k_b == self.K:
                        self.Nk[k_a] -= 1
                        self.Nk.append(1)
                    else:
                        self.Nk[k_a] -= 1
                        self.Nk[k_b] += 1

                self.K = len(self.Nk)

            assert max(self.Z) == len(self.Nk)-1
            assert sum(self.Nk) == self._n

    def fit(self, n_iter=200):
        """ Inits the state and runs the gibbs sampler """
        # init from the prior
        for _ in range(n_iter):
            self.__transition()

        return self.Z
