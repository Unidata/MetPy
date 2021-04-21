# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Setup script for installing MetPy custom flake8 plugin."""

from setuptools import setup

setup(
    name='flake8-metpy',
    version='1.0',
    license='BSD 3 Clause',
    py_modules=['flake8_metpy'],
    install_requires=['flake8 > 3.0.0'],
    entry_points={
        'flake8.extension': ['METPY00 = flake8_metpy:MetPyChecker'],
    },
)
