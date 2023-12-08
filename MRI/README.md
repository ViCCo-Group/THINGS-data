# THINGS-fMRI

This repository contains code for running the analyses presented in the [THINGS-data manuscript](https://doi.org/10.1101/2022.07.22.501123).

## Installation

The python code can be installed from this repository with `pip`:

```
# create a new environment
conda create -n thingsdata python==3.9
conda activate thingsdata
# install the python modules for analyzing the fMRI data
pip install -e .
```

## External requirements

Some of the analyses run non-python neuroimaging software, such as [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/), [ANTS](https://stnava.github.io/ANTs/), and [FSL](https://fsl.fmrib.ox.ac.uk/). 


## Jupyter notebooks

-  [fmri_usage.ipynb](notebooks/fmri_usage.ipynb): Examples on how to interact with the fMRI data in general, such as: a) loading the single trial responses from the table or the volumetric data, b) using the brain masks to convert data between these two formats, c) plotting data on the cortical flat maps.
-  [animacy_size.ipynb](notebooks/animacy_size.ipynb): Demonstration for fitting an encoding model of object animacy and size to the single trial fMRI responses.

## Python modules

- [reconall.py](src/reconall.py): Python scrpit wrapping [FreeSurfer reconall](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all). 
- [scenePRF_Fix.py](src/scenePRF_Fix.py): Experiment code to run used to run the PRF experiment in psychopy.
- [prf.py](src/prf.py): Analysis code for running a population receptive field model in [AFNI](https://afni.nimh.nih.gov/), refining the retinotopic estimates with [neuropythy](https://github.com/noahbenson/neuropythy), and generating ROIs for retinotopic brain areas.
- [localizerGLM_FSL.py](src/localizerGLM_FSL.py): Analysis code for the object category functional localizer, running [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) and [nipype](https://nipy.org/packages/nipype/index.html).
- [melodic.py](src/melodic.py): Analysis code for the ICA denoising procedure. This includes: a) running ICA ([FSL MELODIC](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC)) on the preprocessed functional MRI data, b) calculating a list of features which characterize each IC, c) generating visualizations for raters to label, d) output ICA noise regressors based on feature thresholds. 
- [betas.py](src/betas.py): Procedure for estimating single trial responses from the preprocessed volumetric time series data.
- [anc.py](src/anc.py): Estimation of noise ceilings in single-trial response estimates.
- [mds_betas.py](src/mds_betas.py): Script for visualizing similarity structure in LOC responses via multidimensional scaling (grouped by object categories). 
- [utils.py](src/utils.py), [glm.py](src/glm.py), [dataset](src/dataset.py): Miscellaneous helper functions used by the other modules.

## Bash scripts

- [neurodocker.sh](scripts/neurodocker.sh): Recipe for a docker container running FSL and FreeSurfer.  
- [reconall.sh](scripts/reconall.sh): Run FreeSurfer recon-all through the neurodocker container. 
- [run_fmriprep.sh](scripts/run_fmriprep.sh): [fMRIPrep](https://fmriprep.org/en/stable/) command.