#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup

setup(name='clusim', 
      version='0.1', 
      description = 'Clustering simliarity', 
      author = 'Alex Gates <xxx@xxx.com>', 
      packages = ['clusim.clugen', 'clusim.clusimelement', 'clusim.clustering', 
                  'clusim.plotutils', 'clusim.sim', 'clusim.utils'],
)
