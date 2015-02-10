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


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from idsteach import ids
from idsteach import utils
from idsteach.teacher import Teacher
from idsteach.models import NormalInverseWishart

import pickle
import time


def rebuild_data(filename=None, f3=False):
    """
    FIXME: Docstring
    """
    data_out = dict()
    # long run
    target_model, labels = ids.gen_model(which_phonemes=ids.all_the_vowels, f3=f3)
    # target_model, labels = ids.gen_model(which_phonemes=ids.corner_vowels)
    data_model = NormalInverseWishart.with_vague_prior(target_model)
    t = Teacher(target_model, data_model, 1.0, t_std=40, use_mp=True, fast_niw=True)
    t.mh(2000, burn=1000, lag=100)

    data_out['labels'] = labels
    data_out['target'] = target_model
    data_out['long_run_data'] = t.data
    data_out['short_runs_data'] = []

    for _ in range(10):
        t = Teacher(target_model, data_model, 1.0, t_std=40, use_mp=True, fast_niw=True)
        t.mh(1000, burn=300, lag=20)
        data_out['short_runs_data'].append(t.data)

    if filename is None:
        filename = str(int(time.time()*100)) + '.pkl'

    pickle.dump(data_out, open(filename, "wb"))


def reproduce_paper(filename=None):
    """
    Loads in data and reporoduces the manuscript figures.
    """
    if filename is None:
        filename = 'default_data.pkl'

    dataset = pickle.load(open(filename, "rb"))
    short_runs_data = dataset['short_runs_data']
    long_run_data = dataset['long_run_data']
    # short_run_z = dataset['short_run_z']
    # long_run_z = dataset['long_run_z']

    target = dataset['target']
    labels = dataset['labels']

    plt.figure(tight_layout=True, facecolor='white')

    plt.subplot(3, 1, 1)
    ids.plot_phoneme_models(long_run_data, target, labels)
    plt.subplot(3, 1, 2)
    ids.plot_phoneme_articulation(short_runs_data, target, labels)
    plt.subplot(3, 1, 3)
    ids.plot_phoneme_variation(short_runs_data, target, labels)
    plt.show()


def example():
    """
    Runs a live example showing how incresing variance can occur as a result of teaching
    """
    n = 500
    cov_a = np.array([[3, 0], [0, 1]], dtype=np.dtype(float))
    cov_b = np.array([[1, 0], [0, 3]], dtype=np.dtype(float))
    mean_a = np.array([0.0, 0.0])
    mean_b = np.array([0.0, 0.0])

    target_model = {
        'd': 2,
        'parameters': [
            (mean_a, cov_a),
            (mean_b, cov_b),
            ],
        'assignment': np.array([0, 0, 1, 1], dtype=np.dtype(int))
    }

    prior = {
        'nu_0': 3,
        'kappa_0': 1,
        'mu_0': np.zeros(2),
        'lambda_0': np.eye(2)
    }

    data_model = NormalInverseWishart(**prior)
    t = Teacher(target_model, data_model, 1.0, t_std=1, fast_niw=True)
    t.mh(n, burn=500, lag=20, plot_diagnostics=False)

    X_orig = np.vstack((np.random.multivariate_normal(mean_a, cov_a, n),
                        np.random.multivariate_normal(mean_b, cov_b, n)))
    X_opt, _ = t.get_stacked_data()

    #plot
    plt.figure(tight_layout=True, facecolor='white')
    plt.scatter(X_opt[:, 0], X_opt[:, 1], color='blue', alpha=.5, label='optimized')
    plt.scatter(X_orig[:, 0], X_orig[:, 1], color='red', alpha=.5, label='original')
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run examples')
    parser.add_argument('--example', action='store_true', help='Do cross example')
    parser.add_argument('--paperfigs', action='store_true', help='reproduce manuscript figures')
    parser.add_argument('--rebuild', action='store_true', help='rebuild full dataset')
    parser.add_argument('--filename', type=str, default=None, help='data file name')

    args = parser.parse_args()

    if args.rebuild:
        print("Reproducing figures from loaded data...")
        # TODO: give user the option to load in their own data
        rebuild_data(args.filename)

    if args.example:
        print("Running cross example...")
        example()

    if args.paperfigs:
        print("Reproducing figures from loaded data...")
        # TODO: give use the option to load in their own data
        reproduce_paper(args.filename)

    # utils.what_do_you_think_about_the_stars_in_the_sky()
