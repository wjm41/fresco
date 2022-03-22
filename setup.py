import os
import codecs
from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="fresco",
    version=get_version("fresco/__init__.py"),
    description="Code for performing virtual screening by applying unsupervised machine learning on crystal structures of fragment-protein complexes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wjm41/frag-pcore-screen",
    packages=['fresco'],
    author="W. McCorkindale",
    license="MIT License"
)
