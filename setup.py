from distutils.core import setup
import numpy as np
from Cython.Build import cythonize

setup(
    name='pykonal',
    ext_modules=cythonize("pykonal.pyx", annotate=True),
    include_dirs=[np.get_include()]
)
