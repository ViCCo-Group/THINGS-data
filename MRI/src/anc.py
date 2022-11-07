"""
Calculate analytical noise ceiling.
"""

import os
import sys
from os.path import join as pjoin
from os.path import pardir
from tqdm import tqdm
import numpy as np
from scipy.stats import zscore
from nilearn.image import concat_imgs, mean_img, threshold_img, math_img, index_img, new_img_like

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
    nanpol = 'omit' if ignore_nans else 'propagate'
    normalized = zscore(data, axis=-1, nan_policy=nanpol)
    if ignore_nans:
        normalized = np.nan_to_num(normalized)
        noisesd = np.sqrt(np.nanmean(np.nanvar(normalized, axis=-2, ddof=1), axis=-1))
    else:
        noisesd = np.sqrt(np.mean(np.var(normalized, axis=-2, ddof=1), axis=-1))
    sigsd = np.sqrt(np.clip(1 - noisesd ** 2, 0., None))
    ncsnr = sigsd / noisesd
    nc = 100 * ((ncsnr ** 2) / ((ncsnr ** 2) + (1 / n)))
    return nc


def calc_kknc_singletrialbetas(
        sub: str,
        bidsroot: str,
        stb_derivnames: list,
        out_derivnames: list,
        ns: list,
) -> None:
    """
    Calculate KKNC for different versions of our single trial betas.
    Example:
        sub = sys.argv[1]
        bidsroot = pjoin(pardir, pardir, pardir, pardir)
        ns = [1, 3, 12]
        stb_derivnames = [
            'derivatives/betas_loo/on_residuals/assume_hrf_unregularized',
            'derivatives/betas_loo/on_residuals/unregularized',
            'derivatives/betas_loo/on_residuals/regularized',
            'derivatives/betas_loo/on_residuals/scalematched',
        ]
        out_derivnames = [f.replace('/betas_loo/', '/kknc/model-stb/') for f in stb_derivnames]
        calc_kknc_singletrialbetas(sub, bidsroot, stb_derivnames, out_derivnames, ns)
    """
    assert len(stb_derivnames) == len(out_derivnames)
    for stb_derivname, out_derivname in tqdm(zip(stb_derivnames, out_derivnames),
                                             desc='Iterating through betas versions', total=len(stb_derivnames)):
        betas_basedir = pjoin(bidsroot, 'derivatives', stb_derivname)
        assert os.path.exists(betas_basedir)
        outdir = pjoin(bidsroot, 'derivatives', out_derivname)
        if not os.path.exists(outdir):
            print(f'making output directory: {outdir}')
            os.makedirs(outdir)
        print(f'loading betas: {stb_derivname}')
        betas, example_img = list_stb_outputs_for_mcnc(sub, bidsroot, betas_basedir=betas_basedir)
        ncs = [kknc(betas, n) for n in tqdm(ns, desc='calculating noise ceilings')]
        for n, nc in tqdm(zip(ns, ncs), desc='saving output files', total=len(ncs)):
            img = new_img_like(example_img, nc)
            outfile = pjoin(bidsroot, 'derivatives', out_derivname, f'sub-{sub}_kknc_n-{n}.nii.gz')
            img.to_filename(outfile)


if __name__ == '__main__':
    sub = sys.argv[1]
    bidsroot = pjoin(pardir, pardir, pardir, pardir)
    ns = list(range(1, 13))
    stb_derivnames = [
        'betas_loo/on_residuals/assume_hrf_unregularized',
        'betas_loo/on_residuals/unregularized',
        'betas_loo/on_residuals/regularized',
        'betas_loo/on_residuals/scalematched',
        'betas_loo/simultaneous_fit/assume_hrf_unregularized',
        'betas_loo/simultaneous_fit/unregularized',
        'betas_loo/simultaneous_fit/regularized',
        'betas_loo/simultaneous_fit/scalematched',
        'betas_loo/orthogonalized/assume_hrf_unregularized',
        'betas_loo/orthogonalized/unregularized',
        'betas_loo/orthogonalized/regularized',
        'betas_loo/orthogonalized/scalematched',
        'betas_loo/no_icadenoising/assume_hrf_unregularized',
        'betas_loo/no_icadenoising/unregularized',
        'betas_loo/no_icadenoising/regularized',
        'betas_loo/no_icadenoising/scalematched',
        'betas_loo/aroma/on_residuals/assume_hrf_unregularized',
    ]
    out_derivnames = [f.replace('betas_loo/', 'kknc/model-stb_nanvar/') for f in stb_derivnames]
    calc_kknc_singletrialbetas(sub, bidsroot, stb_derivnames, out_derivnames, ns)
