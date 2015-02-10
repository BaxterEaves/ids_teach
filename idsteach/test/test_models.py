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


import unittest

import numpy as np

from idsteach.models import NormalInverseWishart as NIW


class TestNormalInverseWishartValues(unittest.TestCase):
    def setUp(self):
        self.generic_params = dict(
            mu_0=np.zeros(2),
            lambda_0=np.array([[1, 0], [0, 1]]),
            kappa_0=1,
            nu_0=2
        )
        self.random_params = dict(
            mu_0=np.array([-0.124144348216312, 1.48969760778546]),
            lambda_0=np.array([[0.226836817541677, -0.0200753958619398],
                              [-0.0200753958619398, 0.217753683861863]]),
            kappa_0=2.03620546457332,
            nu_0=2.273220391735,
        )

        self.generic_niw = NIW(**self.generic_params)
        self.random_niw = NIW(**self.random_params)

        self.singledata = np.array([[3.57839693972576, 0.725404224946106]])
        self.multidata = np.array([[3.57839693972576, 0.725404224946106],
                                  [2.76943702988488, -0.0630548731896562],
                                  [-1.34988694015652, 0.714742903826096],
                                  [3.03492346633185, -0.204966058299775]])

    def test_generic_prior_value(self):
        logp = self.generic_niw.log_marginal_likelihood(self.multidata)
        self.assertAlmostEqual(-16.3923777220275, logp)

    def test_single_value_generic_prior(self):
        logp = self.generic_niw.log_marginal_likelihood(self.singledata)
        self.assertAlmostEqual(-5.5861321608291, logp)

    def test_random_prior_value(self):
        logp = self.random_niw.log_marginal_likelihood(self.multidata)
        self.assertAlmostEqual(-19.5739755706395, logp)

    def test_single_value_random_prior(self):
        logp = self.random_niw.log_marginal_likelihood(self.singledata)
        self.assertAlmostEqual(-6.60964751885643, logp)

    def test_log_liklihood_values_generic_params(self):
        params = (self.generic_params['mu_0'], self.generic_params['lambda_0'])
        logpdf_s = NIW.log_likelihood(self.singledata, *params)
        logpdf_m = NIW.log_likelihood(self.multidata, *params)

        self.assertAlmostEqual(logpdf_s, -8.50344504031352)
        self.assertAlmostEqual(logpdf_m, -23.6468667799676)

    def test_predictive_probability(self):
        logp = self.random_niw.log_posterior_predictive(self.singledata, self.multidata)
        self.assertAlmostEqual(-3.25801708275319, logp)

if __name__ == '__main__':
    unittest.main()
