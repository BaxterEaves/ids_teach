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
from idsteach.utils import multiple_pandas_to_teacher_data
from idsteach.experiments.exp_learntime import ari_over_time_violin
from idsteach.experiments.exp_algcomp import plot_compare

import pickle
import time
import os

sns.set_context('paper')

DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, 'data')


def rebuild_data(filename=None, f3=False):
    """
    FIXME: Docstring
    """
    data_out = dict()
    target_model, labels = ids.gen_model(which_phonemes=ids.all_the_vowels,
                                         f3=f3)
    data_model = NormalInverseWishart.with_vague_prior(target_model)
    t = Teacher(target_model, data_model, 1.0, t_std=40, use_mp=True,
                fast_niw=True)
    t.mh(2000, burn=1000, lag=100)

    data_out['labels'] = labels
    data_out['target'] = target_model
    data_out['long_run_data'] = t.data
    data_out['short_runs_data'] = []

    for _ in range(10):
        t = Teacher(target_model, data_model, 1.0, t_std=40, use_mp=True,
                    fast_niw=True)
        t.mh(1000, burn=300, lag=20)
        data_out['short_runs_data'].append(t.data)

    if filename is None:
        filename = str(int(time.time()*100)) + '.pkl'

    pickle.dump(data_out, open(filename, "wb"))


def reproduce_paper():
    """
    Loads in data and reporoduces the manuscript figures.
    """
    dirname = os.path.join(DATADIR, 'ptn_runs')
    tdatlst_f3 = [multiple_pandas_to_teacher_data(dirname)]
    tdatlst_f2 = [multiple_pandas_to_teacher_data(dirname, remove_f3=True)]

    target_f3, labels = ids.gen_model(f3=True)
    target_f2, _ = ids.gen_model(f3=False)

    # Phoneme models and samples
    # --------------------------
    plt.figure(tight_layout=True, facecolor='white', figsize=(9.5, 6.34,))
    sns.set(rc={'axes.facecolor': '#aaaaaa', 'grid.color': '#bbbbbb'})

    plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    plt.title('A')
    ids.plot_phoneme_models(tdatlst_f3[0], target_f3, labels,
                            formants=[0, 1], nstd=2,
                            legend=True, grayscale=True)
    plt.subplot2grid((2, 3), (0, 2))
    plt.title('B')
    ids.plot_phoneme_models(tdatlst_f3[0], target_f3, labels, formants=[1, 2],
                            nstd=2, grayscale=True)
    plt.subplot2grid((2, 3), (1, 2))
    plt.title('C')
    ids.plot_phoneme_models(tdatlst_f3[0], target_f3, labels, formants=[0, 2],
                            nstd=2, grayscale=True)

    plt.savefig('fig-1.png', dpi=300)

    # Articulation
    # ------------
    sns.set(rc={'axes.facecolor': '#e8e8e8', 'grid.color': '#ffffff'})
    f = plt.figure(tight_layout=True, facecolor='white', figsize=(9.5, 4.75,))

    ax1 = f.add_subplot(2, 1, 1)
    plt.title('A')
    indices = ids.plot_phoneme_articulation(tdatlst_f3, target_f3, labels,
                                            ax=ax1)
    ax2 = f.add_subplot(2, 1, 2)
    plt.title('B')
    ids.plot_phoneme_articulation(tdatlst_f2, target_f2, labels, ax=ax2,
                                  indices=indices)
    ax2.set_ylim(ax1.get_ylim())

    plt.savefig('fig-2.png', dpi=300)

    # Variation (F1, F2, F3)
    # ---------------------
    plt.figure(tight_layout=True, facecolor='white', figsize=(9.5, 3.5,))
    sns.set(rc={'axes.facecolor': '#e8e8e8'})
    ids.plot_phoneme_variation(tdatlst_f3, target_f3, labels)
    plt.savefig('fig-3.png', dpi=300)

    # ARI for different learning algorithms
    ari_file_f3 = os.path.join(DATADIR, 'omnirunf3.pkl')
    ari_file_f2 = os.path.join(DATADIR, '2FLearning_500ex.pkl')
    ari_ylabels = ["ARI (F1, F2, F3)", "ARI (F1, F2)"]
    sns.set(rc={'axes.facecolor': '#cccccc', 'grid.color': '#bbbbbb'})
    plot_compare([ari_file_f3, ari_file_f2], [500, 500], ari_ylabels,
                 grayscale=True)
    plt.savefig('fig-4.png', dpi=300)

    # DPGMM ARI as a funciton of the number of samples
    # ------------------------------------------------
    f = plt.figure(tight_layout=True, facecolor='white', figsize=(7.5, 3.5,))
    ari_over_time_violin()
    plt.savefig('fig-5.png', dpi=300)


def example():
    """
    Runs a live example showing how incresing variance can occur as a result
    of teaching
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

    plt.figure(tight_layout=True, facecolor='white')
    plt.scatter(X_opt[:, 0], X_opt[:, 1], color='royalblue', alpha=.5,
                label='optimized')
    plt.scatter(X_orig[:, 0], X_orig[:, 1], color='crimson', alpha=.5,
                label='original')
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run examples')
    parser.add_argument('--example', action='store_true',
                        help='Do cross example')
    parser.add_argument('--paperfigs', action='store_true',
                        help='reproduce manuscript figures')
    parser.add_argument('--rebuild', action='store_true',
                        help='rebuild full dataset')
    parser.add_argument('--filename', type=str, default=None,
                        help='data file name')

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
        reproduce_paper()

    utils.what_do_you_think_about_the_stars_in_the_sky()
