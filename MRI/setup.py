from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="thingsmri",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    author="Oliver Contier",
    author_email="contier@cbs.mpg.de",
    description="Python code for analyzing the things-fMRI dataset",
)