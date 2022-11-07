# THINGS-data

[THINGS-data](https://doi.org/10.1101/2022.07.22.501123) is a collection of large-scale datasets for the study of natural object representations in brain and behavior. It includes functional magnetic resonance imaging (fMRI) data, magnetoencephalographic (MEG) recordings, and 4.70 million similarity judgments in response to thousands of images from the [THINGS object concept and image database](https://doi.org/10.1371/journal.pone.0223792).

> 🔄 Early access
>
> The THINGS-data manuscript is currently under review. All of the data will be made publicly available upon acceptance of the manuscript and the relevant links will be included in this readme. If you are interested in early access, get in touch (hebart [at] cbs.mpg.de)! Currently, early access is restricted to the figshare collection.


# Download

## Download from figshare

THINGS-data is hosted as a collection of data objects on figshare. You can browse the collection and download individual parts which are relevant for your research. 

By default, clicking on the desired data object will prompt a browser download. If you plan to download larger data objects such as the raw MEG or fMRI datasets, it might make sense to start this process in the command line. Simply right-click on the “Download” button and copy the link address. Executing the following code in the command line to begin the download process for that file. 
```
wget https://figshare.com/copied/link/address
```
For longer downloads, it might make sense to run this process in the background with tools such as `screen` or `tmux`.

## Download from OpenNeuro

The raw fMRI and MEG datasets are available on [OpenNeuro](https://openneuro.org). You can find general explanations on how to download data from OpenNeuro in the official [documentation](https://docs.openneuro.org/user-guide).

## Download from OSF

The behavioral dataset containing 4.7 million human similarity judgements is available on OSF and can be downloaded directly via your web browser.


# Repository contents

The analysis code in this repository is structured into three sub-folders for the three data modalities: [MRI](MRI), [MEG](MEG), and [behavior](behavior).

# How to cite
```
@article {Hebart2022.07.22.501123,
	author = {Hebart, M.N. and Contier, O. and Teichmann, L. and Rockter, A.H. and Zheng, C.Y. and Kidder, A. and Corriveau, A. and Vaziri-Pashkam, M. and Baker, C.I.},
	title = {THINGS-data: A multimodal collection of large-scale datasets for investigating object representations in brain and behavior},
	elocation-id = {2022.07.22.501123},
	year = {2022},
	doi = {10.1101/2022.07.22.501123},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/07/23/2022.07.22.501123},
	eprint = {https://www.biorxiv.org/content/early/2022/07/23/2022.07.22.501123.full.pdf},
	journal = {bioRxiv}
}
```