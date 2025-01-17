from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'Implementation of Nested Sampling with Barriers'
LONG_DESCRIPTION = 'This is an implementation of Nested Sampling with Barriers, a modified version of the Nested Sampling algorithm by Skilling. It can be used to fit Bayesian models.'

# Setting up
setup(
    name="barriersampling",
    version=VERSION,
    author="Farin Lippmann",
    author_email="<toucanmeister@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'jax', 'scipy', 'matplotlib'],
    keywords=['python', 'statistics', 'sampling', 'bayes', 'optimization']
)