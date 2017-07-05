#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='dlibfaceextractor',
      version='0.0.1',
      description='Face extractor based on dlib',
      author='sergem',
      author_email='sbmatyunin@gmail.com',
      url='https://serge-m.github.io',
      install_requires=required,
      packages=['dlibfaceextractor'],
     )


