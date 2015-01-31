from setuptools import setup
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from pip.req import parse_requirements


install_reqs = parse_requirements('requirements.txt')
reqs = [str(ir.req) for ir in install_reqs]

niw_ext = Extension("ids_teach.niw_module", 
                    sources=["ids_teach/cppext/niw_module.pyx"],
                    extra_compile_args=['-std=c++11'],
                    extra_link_args=['-O2', '-larmadillo'],
                    include_dirs=['cpp'],
                    language="c++")

setup(
    name='IDS teach',
    version='0.1',
    author='Baxter S. Eaves Jr.',
    url='TBA',
    long_description='TBA.',
    install_requires=reqs,
    package_dir={'ids_teach':'ids_teach/'},
    ext_modules=[niw_ext],
    cmdclass={'build_ext': build_ext},
)
