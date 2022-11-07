#!/usr/bin/env bash

subject=$1
nprocs=$2
maxmem=15 # 400

rawdatadir="$(pwd)/../../../rawdata"
derivsdir="$(pwd)/../../../../test_fprep_upsampling/derivatives"
workdir="$(pwd)/../../../../fmriprep_wdir"

docker run -ti --rm \
  --memory="$maxmem""g" \
  -v "$rawdatadir":/data:ro \
  -v "$derivsdir":/out \
  -v "$workdir":/work \
  -v "$(pwd)":/licensedir \
  poldracklab/fmriprep:20.2.0 \
  --participant-label "$subject" \
  --t nsd --fs-no-reconall \
  --output-spaces T1w func \
  --bold2t1w-dof 9 \
  --nprocs "$nprocs" --mem "$maxmem""GB" \
  --fs-license-file /licensedir/license.txt \
  --bids-filter-file /licensedir/filtercfg.json \
  -w /work /data /out participant
