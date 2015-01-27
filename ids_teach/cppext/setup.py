from distutils.core import setup, Extension
from Cython.Distutils import build_ext

setup(
    name='niw_module',
    ext_modules=[
        Extension(
            "niw_module", 
            sources=["niw_module.pyx"],
            extra_compile_args=['-std=c++11'],
            extra_link_args=['-O2', '-larmadillo'],
            include_dirs=['../../cpp', '/usr/local/include'],
            language="c++",
        )
    ],
    cmdclass={'build_ext': build_ext},
)
