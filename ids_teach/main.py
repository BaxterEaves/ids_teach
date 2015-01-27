from ids_teach import ids
from ids_teach import utils
from ids_teach.teacher import Teacher
from ids_teach.models import NormalInverseWishart

import matplotlib.pyplot as plt
import seaborn as sns


def main():

    Xs = []

    target, labels = ids.gen_model(ids.all_the_vowels[2:])
    for _ in range(10):
        niw = NormalInverseWishart.with_vague_prior(target)
        t = Teacher(target, niw, 1.0, use_mp=True, fast_niw=True)

        t.mh(1000, burn=200, lag=10, plot_diagnostics=False)

        X = t.data
        Xs.append(X)

    plt.close()
    plt.ioff()
    plt.figure(tight_layout=True, facecolor='white')
    ids.plot_phoneme_models(X, target, labels, err_interval=.5)
    plt.show()

    plt.figure(tight_layout=True, facecolor='white')
    ids.plot_phoneme_articulation(Xs, target, labels)
    plt.show()


if __name__ == '__main__':
    main()
