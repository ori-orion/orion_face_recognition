#!/usr/bin/env python

import setuptools

from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    #packages=setuptools.find_packages(where="src"),#['flib'],
    packages=['flib','actionserver_flib'],
    package_dir={'': 'src'},
    # package_dir={'flib': 'src/flib', 'actionserver_flib':'src/actionserver_flib'},
)

setup(**setup_args)
