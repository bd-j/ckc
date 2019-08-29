#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import subprocess
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

    
setup(
    name="ckc",
    url="https://github.com/cconroy20/h3",
    version="0.1",
    author="",
    author_email="benjamin.johnson@cfa.harvard.edu",
    packages=["ckc"],
    license="LICENSE",
    description="C3K stellar library utilities",
    #long_description=open("README.md").read(),
    #package_data={"": ["README.md", "LICENSE"]},
    #scripts=glob.glob("scripts/*.py"),
    include_package_data=True,
    install_requires=["numpy", "astropy", "h5py"],
)
