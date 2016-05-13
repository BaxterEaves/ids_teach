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


from setuptools import setup
from setuptools import find_packages
from distutils.core import Extension
from pip.req import parse_requirements

import os

cmdclass = dict()

NO_CYTHON = os.environ.get('NO_CYTHON', False)
if NO_CYTHON:
    print('Using pre-built cython files.')
    ext = '.cpp'
else:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    cmdclass['build_ext'] = build_ext
    print('Using cython.')
    ext = '.pyx'

extensions = [Extension("idsteach.fastniw",
                        sources=["idsteach/fastniw"+ext, "src/dpgmm.cpp"],
                        libraries=['armadillo'],
                        extra_compile_args=['-std=c++11',
                                            # '-DNIDSDEBUG',
                                            '-DARMA_NO_DEBUG'],
                        include_dirs=['src'],
                        language="c++")]

if not NO_CYTHON:
    extensions = cythonize(extensions)

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='idsteach',
    version='0.1.0-dev',
    author='Baxter S. Eaves Jr.',
    url='TBA',
    long_description='TBA.',
    install_requires=reqs,
    package_dir={'idsteach': 'idsteach/'},
    packages=find_packages(exclude=['testdpgmm.py']),
    ext_modules=extensions,
    cmdclass=cmdclass
)
