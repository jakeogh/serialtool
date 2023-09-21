# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

import fastentrypoints

dependencies = [
    "pyserial",
    "click",
    "clicktool @ git+https://git@github.com/jakeogh/clicktool",
    "timestamptool @ git+https://git@github.com/jakeogh/timestamptool",
]

config = {
    "version": "0.1",
    "name": "serialtool",
    "url": "https://github.com/jakeogh/serialtool",
    "license": "ISC",
    "author": "Justin Keogh",
    "author_email": "github.com@v6y.net",
    "description": "common functions for serial communication",
    "long_description": __doc__,
    "packages": find_packages(exclude=["tests"]),
    "package_data": {"serialtool": ["py.typed"]},
    "include_package_data": True,
    "zip_safe": False,
    "platforms": "any",
    "install_requires": dependencies,
    "entry_points": {
        "console_scripts": [
            "serialtool=serialtool.serialtool:cli",
        ],
    },
}

setup(**config)
