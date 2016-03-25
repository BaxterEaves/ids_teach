
import os
import gzip
import pandas
import pickle

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')

DIR = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(DIR, 'data', 'algcomp-2048.pkl.gz')


def ari_over_time_violin(algcomp_filename=FILENAME, log_x_scale=False,
                         alg='dpgmm', ax=None, set_ylim=False):
    data = pickle.load(gzip.GzipFile(algcomp_filename, 'r'))
    c1, c3, c2 = sns.color_palette("Greys", 3)

    if ax is None:
        ax = plt.gca()

    to_df = []
    for n in sorted(data.keys()):
        to_df += [{'n': int(n), 'ari': ari, 'type': 'Teaching'} for ari in
                  data[n]['res_optimal'][alg]['ari']]
        to_df += [{'n': int(n), 'ari': ari, 'type': 'ADS'} for ari in
                  data[n]['res_original'][alg]['ari']]
        to_df += [{'n': int(n), 'ari': ari, 'type': 'Transfer'} for ari in
                  data[n]['res_cross'][alg]['ari']]
    df = pandas.DataFrame(to_df)

    sns.violinplot(data=df, x='n', y='ari', hue='type',
                   hue_order=['ADS', 'Transfer', 'Teaching'], ax=ax,
                   palette='Greys', scale='width', linewidth=.5)
    ax.set_xlabel('Number of examples per phoneme')
    ax.set_ylabel('ARI (%s)' % (alg.upper(),))
    return ax


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot ARI distribution over time')
    parser.add_argument('-f', '--filename', type=str, default=FILENAME,
                        help='data input filename (pickle)')
    parser.add_argument('-a', '--algorithm', type=str, nargs='+',
                        help='the algorithm key')
    args = parser.parse_args()

    figwidth = 7.5

    f, axes = plt.subplots(len(args.algorithm), 1, facecolor='white')
    if len(args.algorithm) == 1:
        axes = [axes]
    for i, alg in enumerate(args.algorithm):
        ax = ari_over_time_violin(args.filename, alg=alg, ax=axes[i],
                                  set_ylim=len(args.algorithm) > 1,
                                  log_x_scale=True)
    f.set_size_inches(7.5, 3.5*len(args.algorithm))

    plt.savefig('ari-over-ex_{}.png'.format(args.algorithm), dpi=300)
