#!/usr/bin/env bash

source ~/.bashrc
conda activate thingsmri_env

thingsmridir="$(pwd)/../../../../"
subject=$1
nprocs=50

docker run -it --rm --mount \
  type=bind,source="${thingsmridir}",target=/thingsmri \
  preproc \
  python /thingsmri/bids/code/things/mri/reconall.py "${subject}" "${nprocs}"
