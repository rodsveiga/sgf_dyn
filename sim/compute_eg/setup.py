from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='compute_eg',
    ext_modules=cythonize('compute_eg.pyx'),
    zip_safe=False,
    include_dirs=[np.get_include()]
)