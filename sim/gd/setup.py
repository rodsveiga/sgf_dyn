from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='trainGD',
    ext_modules=cythonize('trainGD.pyx'),
    zip_safe=False,
    include_dirs=[np.get_include()]
)