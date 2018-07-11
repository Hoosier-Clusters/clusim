#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from clusim import __package__, __description__, __version__


def readme():
    with open('README.md') as f:
        return f.read()


setup(name=__package__,
      version=__version__,
      description='Clustering simliarity',
      long_description=__description__,
      url="https://github.com/ajgates42/clusim",
      author='Alex Gates <ajgates42@gmail.com>',
      license="MIT",
      packages=['clusim.clugen', 'clusim.clusimelement', 'clusim.clustering',
                'clusim.plotutils', 'clusim.sim', 'clusim.utils'],
      install_requires=['numpy',
                        'scipy',
                        'networkx',
                        'mpmath',
                        ],
      include_package_data=True
      )
