from setuptools import find_packages, setup

setup(
    name="fresco",
    version="0.1.0",
    description="Code for performing virtual screening by applying unsupervised machine learning on crystal structures of fragment-protein complexes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wjm41/frag-pcore-screen",
    packages=find_packages(),
    author="W. McCorkindale",
    license="MIT License"
)
