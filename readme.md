# Code for `Title`
This is the code used to generate data in the paper `cite`

## Insallation
The installation proceedure has been tested only on Mac OSX Yosimite.
### Requirements (OSX)

For the python parts, run

```bash
$ pip install -r requirements.txt
```
For the C++ parts you will need the [Armadillo linear algebra library](http://arma.sourceforge.net/download.html). You can install it with [homebrew](http://brew.sh/).

```bash
$ brew update
$ brew install armadillo
```

### Testing

To run tests of the python code

```bash
$ cd ids_teach/tests
$ py.test

```

To run (adhoc) tests of the C++ code:

```bash
$ cd cpp
$ make runtest
```

## Use

An example can be found in `main.py`.

```python
from ids_teach import ids
from ids_teach import utils
from ids_teach.teacher import Teacher
from ids_teach.models import NormalInverseWishart

# generate corner-vowel-only model
target, labels = ids.gen_model(ids.corner_vowels)
# create a data model, give it to the teacher and generate optimized teaching data
niw = NormalInverseWishart.with_vague_prior(target)
t = Teacher(target, niw, crp_alpha=1.0)
# plot_diagnostics produces plots in real time
t.mh(250, burn=10, lag=10, plot_diagnostics=True)
```

 You should see something like this:
 ![Corner vowel analysis](figure_1.png)


**NOTE**: If you intend to recreate the entire experiment (by providing `ids.gen_model` with `ids.all_the_vowels`), be prepared to devote several hours---or more, depending on your machine---to computation.
