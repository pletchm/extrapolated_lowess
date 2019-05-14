#!/usr/bin/env python
from setuptools import find_packages, setup


setup(name='extrapolated_lowess',
      description='Extrapolated LOWESS',
      author='Martin Pletcher',
      author_email='martypletcher@gmail.com',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
          'numpy',
          'scipy'
          ],
      zip_safe=True,
      )
