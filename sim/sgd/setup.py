from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='trainSGD',
    ext_modules=cythonize('trainSGD.pyx'),
    zip_safe=False,
    include_dirs=[np.get_include()]
)