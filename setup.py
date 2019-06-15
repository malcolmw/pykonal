'''
setup.py adapted from https://github.com/kennethreitz/setup.py
'''
import io
import numpy as np
import os
from distutils.core import setup
from Cython.Build import cythonize


# Package meta-data.
name            = 'pykonal'
description     = 'Solver for the Eikonal equation in 3D Cartesian coordiantes.'
url             = 'https://github.com/malcolmw/pykonal'
email           = 'malcolm.white@.usc.edu'
author          = 'Malcolm C. A. White'
requires_python = '>=3'
packages        = ['pykonal']
package_data    = {'pykonal': ['data/marmousi_2d.npz']}
required        = ['cython', 'numpy', 'scipy']
extras          = {'tests': ['nose']}
ext_modules     = cythonize('pykonal/pykonal.pyx')
include_dirs    = [np.get_include()]
license         = 'GNU GPLv3'

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = description

# Load the package's __version__.py module as a dictionary.
about = {}
project_slug = name.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, '__version__.py')) as f:
    exec(f.read(), about)

# Where the magic happens:
setup(
    name=name,
    version=about['__version__'],
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=author,
    author_email=email,
    python_requires=requires_python,
    url=url,
    packages=packages,
    package_data=package_data,
    ext_modules=ext_modules,
    include_dirs=include_dirs,
    install_requires=required,
    extras_require=extras,
    license=license,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
