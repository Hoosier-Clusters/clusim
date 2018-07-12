#!/usr/bin/env python
# encoding: utf-8

<<<<<<< HEAD
from setuptools import setup, find_packages
=======
from distutils.core import setup
>>>>>>> 4c6251836689be3e15e93fbe8b5da10123a9b5fa
from clusim import __package__, __description__, __version__
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}


def readme():
    with open('README.md') as f:
        return f.read()


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
<<<<<<< HEAD
      author = 'Alex Gates <ajgates42@gmail.com>',
      license="MIT", 
      packages = find_packages(),
      install_requires=[
		'numpy',
		'scipy',
		'networkx',
            'mpmath'
	],
	include_package_data=True,
      command_options={
        'build_sphinx': {'source_dir': ('setup.py', 'doc')} }
)
=======
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
>>>>>>> 4c6251836689be3e15e93fbe8b5da10123a9b5fa
