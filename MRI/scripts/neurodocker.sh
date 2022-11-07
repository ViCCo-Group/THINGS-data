#!/usr/bin/env bash

# Generate a docker container that supports FreeSurfer and FSL
neurodocker generate docker --base=debian:stretch --pkg-manager=apt \
  --fsl version=6.0.3 method=binaries \
  --freesurfer version=6.0.0 method=binaries --copy license.txt /opt/freesurfer-6.0.0/ \
  --install gcc g++ graphviz tree nano less git \
  --user=root \
  --miniconda miniconda_version=4.3.31 create_env="thingsmrienv" pip_install="setuptools git+https://github.com/nipy/nipype.git pybids" activate=true \
  > Dockerfile

docker build --rm -t preproc .
