#!/usr/bin/env python
# encoding: utf-8


from setuptools import setup, find_packages
from clusim import __package__, __description__, __version__


setup(name=__package__,
      version=__version__,
      description='Clustering simliarity',
      long_description=__description__,
      classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      keywords="clustering cluster comparison similarity data science network science network community",
      url="https://github.com/ajgates42/clusim",
      author = 'Alex Gates <ajgates42@gmail.com> \n YY Ahn <yyahn@iu.edu>',
      license="MIT",
      packages = find_packages(),
      install_requires=[
            'numpy',
            'scipy',
            'networkx',
            'mpmath',
            'python-igraph'
      ],
      include_package_data=True
      )
