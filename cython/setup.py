from distutils.core import setup
import numpy as np
from Cython.Build import cythonize

setup(name='Hello world app',
      ext_modules=cythonize("*.pyx", annotate=True),
      include_dirs=[np.get_include()]
      )
