#!/usr/bin/env python3

from setuptools import setup, find_packages

from decision_trees import __version__

setup(
    name='DecisionTrees',
    version=__version__,
    packages=find_packages(),

    entry_points={
        'console_scripts': (
            'DecisionTrees = decision_trees.__main__:cli',
        )
    },

    install_requires=[
        'scikit-learn <0.20',
        'scikit-image <0.14',
        'numpy <2.0',
        'matplotlib <3.0',
        'click <7.0',
    ],
)
