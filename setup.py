#!/usr/bin/env python 3.8
import io
import os
import re

from pip._internal.req import parse_requirements
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Find requirements from requirements.txt
requirements = list(parse_requirements("requirements.txt", session="hack"))
try:
    requirements = [str(ir.req) for ir in requirements]
except BaseException:
    requirements = [str(ir.requirement) for ir in requirements]


setup_info = dict(
    # Metadata
    name="torchrecsys",
    version=find_version("torchrecsys", "__init__.py"),
    author="Jaime Ferrando Huertas",
    author_email="fhjaime96@gmail.com",
    description="Lightweight pytorch library to build recommender systems.",
    python_requires=">=3.6",
    # Package info
    packages=find_packages(exclude=[".vscode", "build_tools", "docs", "tests"]),
    zip_safe=True,
    install_requires=requirements,
)

setup(**setup_info)
