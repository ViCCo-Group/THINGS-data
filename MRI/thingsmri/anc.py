"""
Calculate the noise ceiling in single trial response estimates.

Usage:
python anc.py <bids_path> <subject_ID> <betas_path> <output_path>

Example:
python anc.py 01 /home/user/thingsmri /home/user/thingsmri/betas_vol /home/user/thingsmri/noiseceiling
"""

import os
import sys
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
from scipy.stats import zscore
from nilearn.image import new_img_like

sys.path.append(os.getcwd())
from betas import list_stb_outputs_for_mcnc


def kknc(data: np.ndarray, n: int or None = None, ignore_nans=False):
    """
    Calculate the noise ceiling reported in the NSD paper (Allen et al., 2021)

    Arguments:
        data: np.ndarray
            Should be shape (ntargets, nrepetitions, nobservations)
        n: int or None
            Number of trials averaged to calculate the noise ceiling. If None, n will be the number of repetitions.
        ignore_nans: bool
            If True, ignore nans in data normalization and variance calculation.
    returns:
        nc: np.ndarray of shape (ntargets)
            Noise ceiling without considering trial averaging.
        ncav: np.ndarray of shape (ntargets)
            Noise ceiling considering all trials were averaged.
    """
    if not n:
        n = data.shape[-2]
    nanpol = "omit" if ignore_nans else "propagate"
    normalized = zscore(data, axis=-1, nan_policy=nanpol)
    if ignore_nans:
        normalized = np.nan_to_num(normalized)
        noisesd = np.sqrt(np.nanmean(np.nanvar(normalized, axis=-2, ddof=1), axis=-1))
    else:
        noisesd = np.sqrt(np.mean(np.var(normalized, axis=-2, ddof=1), axis=-1))
    sigsd = np.sqrt(np.clip(1 - noisesd**2, 0.0, None))
    ncsnr = sigsd / noisesd
    nc = 100 * ((ncsnr**2) / ((ncsnr**2) + (1 / n)))
    return nc


def calc_kknc_singletrialbetas(
    sub: str,
    bidsroot: str,
    betas_basedirs: list,
    out_dirs: list,
    ns: list,
) -> None:
    """
    Calculate KKNC for different versions of our single trial betas.
    Example:
        sub = '01'
        bidsroot = '/path/to/bids/dataset'
        ns = [1, 12]
        stb_derivnames = ['derivatives/betas_vol/']
        out_derivnames = ['derivatives/nc/']
        calc_kknc_singletrialbetas(sub, bidsroot, stb_derivnames, out_derivnames, ns)
    """
    assert len(betas_basedirs) == len(out_dirs)
    for betas_basedir, outdir in tqdm(
        zip(betas_basedirs, out_dirs),
        desc="Iterating through betas versions",
        total=len(out_dirs),
    ):
        assert os.path.exists(betas_basedir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        betas, example_img = list_stb_outputs_for_mcnc(
            sub, bidsroot, betas_basedir=betas_basedir
        )
        ncs = [kknc(betas, n) for n in tqdm(ns, desc="calculating noise ceilings")]
        for n, nc in tqdm(zip(ns, ncs), desc="saving output files", total=len(ncs)):
            img = new_img_like(example_img, nc)
            outfile = pjoin(outdir, f"sub-{sub}_kknc_n-{n}.nii.gz")
            img.to_filename(outfile)


if __name__ == "__main__":
    sub, bidsroot, betaspath, outpath = sys.argv[1], sys.argv[2]
    ns = [1, 12]
    betas_basedirs = [betaspath]
    out_dirs = [outpath]
    calc_kknc_singletrialbetas(sub, bidsroot, betas_basedirs, out_dirs, ns)
