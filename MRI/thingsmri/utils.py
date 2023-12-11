#! /usr/env/python

import copy
import glob
import io
import json
import os
import subprocess
from os.path import abspath
from os.path import join as pjoin
from os.path import pardir

import numpy as np
import nilearn
import pandas as pd

import cmasher as cmr
from scipy.io import loadmat
import requests
import scipy
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from nilearn.image import resample_to_img, math_img, smooth_img
from nilearn.masking import intersect_masks, apply_mask
from nipype.interfaces.ants.resampling import ApplyTransforms
from scipy.signal import periodogram
from scipy.stats import sem
import matplotlib.pylab as pl
from sklearn.linear_model import LinearRegression

from thingsmri.dataset import ThingsMRIdataset


def save_dict_as_json(d, fname, mode="w"):
    with open(fname, mode) as fp:
        json.dump(d, fp)


def add_rectag_to_BIDSJson(bidsjson, rec_tag):
    """
    Get a BIDSJson file as dict and add the 'reconstruction' value (e.g. for de/refacing)
    Returns a dict with the new json contents and a path to the new json file with the _rec-{rec_tag} part filled in.
    """
    json_dict = bidsjson.get_dict()
    json_dict["reconstruction"] = rec_tag
    # TODO: Make work for other modalities than T1w
    json_path = bidsjson.path.replace("_T1w.json", "_rec-{}_T1w.json".format(rec_tag))
    return json_dict, json_path


def modify_BIDSImage(bidsimage, bidslayout, updates: dict):
    """
    Take a BIDSImage object returned by a pybids query, edit its entities, and reconstruct new file path.
    """
    newimage = copy.copy(bidsimage)
    ent_dict = dict(newimage.entities)
    ent_dict.update(updates)
    newimage.path = bidslayout.build_path(ent_dict)
    return newimage


def get_slice_timings(
    bids_root: str, subject: str, tr: float = 1.5, only_things_sessions: bool = False
):
    """
    Get double-nested list of slice timings per run per session.
    All timings are divided by TR to conform to FSL's slicetimer.
    Also returns the session_names as str.
    """
    thingsmri = ThingsMRIdataset(bids_root, validate=False)
    session_names = (
        thingsmri.things_sessions if only_things_sessions else thingsmri.sessions
    )
    slice_timings = [
        np.array(  # list of arrays of different length
            [
                json.get_dict()["SliceTiming"]
                for json in thingsmri.layout.get(
                    subject=subject, suffix="bold", extension="json", session=ses
                )
            ]
        )
        / float(tr)
        for ses in session_names
    ]
    return slice_timings, session_names


def write_slice_timings(
    slice_timings: list,
    session_names: list,
    tmpdir: str,
    extension: str = "txt",
    row_wise=False,
):
    """
    Writes slice timings to temporary text files in tmpdir for use with FSL's slicetimer.
    """
    txtpaths_all = []
    # remove new line marker to save in a row (e.g. for pnm) instead of column (e.g. for stc)
    nl = " " if row_wise else "\n"
    for ses_name, ses_slices in zip(session_names, slice_timings):
        txtpaths_ses = [
            abspath(
                pjoin(
                    tmpdir, f"slicetimes_ses-{ses_name}_run-{run_i + 1:02}.{extension}"
                )
            )
            for run_i in range(len(ses_slices))
        ]
        txtpaths_all.append(txtpaths_ses)
        for fname, run_slices in zip(txtpaths_ses, ses_slices):
            np.savetxt(fname, run_slices, fmt="%10.5f", newline=nl)
    return txtpaths_all


def df_from_url(url: str, sep: str, header):
    s = requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode("utf-8")), sep=sep, header=header)


def load_spose_dims(
    dims_url: str = "https://osf.io/4pgk8/download",
    names_url: str = "https://osf.io/sz2ux/download",
) -> tuple:
    """
    Load spose restuls:
        - dimensions as array of shape (1854, 49)
        - names as pd.DataFrame with columns [Word Wordnet Synset	Wordnet ID	Wordnet ID2	Wordnet ID3	uniqueID]
    """
    dims_df = df_from_url(dims_url, sep=" ", header=None)
    dims_df = dims_df.dropna(how="all", axis=1)
    dims_arr = np.array(dims_df)
    names_df = df_from_url(names_url, sep="\t", header="infer")
    return dims_arr, names_df


def load_spose_labels(
    labels_mat: str = "/Users/olivercontier/Desktop/pycortex_scripts/labels_short.mat",
):
    labels = loadmat(labels_mat)["labels_short"][0]
    labels = [a[0] for a in labels]
    return labels


def load_clip_preds(
    clippred_dir: str = pjoin(pardir, pardir, "external_libraries", "clip_preds")
):
    """
    Load the image-wise predictions for behavioral (spose) dimensions based on CLIP penultimate layer.
    Returns:
        preds: np.ndarray (26107, 49)
            predicted behavioral dimension weights for each image
        fnames: np.ndarray (26107,), dtype=str
            image file names, i.e. ['./images/aardvark/aardvark_01b.jpg', './images/aardvark/aardvark_02s.jpg', ...]
        concept_inds: np.ndarray (26107,), dtype=int
            indices ranging from 0 to 1853, matching one object concept to each image.
    """
    preds_txt = pjoin(clippred_dir, "predicts_CLIP_ViT_penult_full_things.txt")
    fnames_txt = pjoin(clippred_dir, "file_names_CLIP_ViT_penult_full_things.txt")
    targets_npy = pjoin(clippred_dir, "targets_CLIP_ViT_penult_full_things.npy")
    preds = np.loadtxt(preds_txt)
    fnames = np.loadtxt(fnames_txt, dtype=str)
    concept_inds = np.load(targets_npy)
    return preds, fnames, concept_inds


def project_glasser(
    glasser_atlas: str = pjoin(
        pardir,
        pardir,
        pardir,
        pardir,
        "atlases",
        "glasser_vol",
        "HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz",
    ),
    fmriprepdir: str = pjoin(pardir, pardir, "derivatives", "fmriprep"),
    outdir: str = pjoin(pardir, pardir, "derivatives", "glasser_projected"),
):
    """
    Project the Atlas from Glasser et al. 2016 to anatomical space of all subjects.
    Requires ANTs to be installed.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for sub in ["01", "02", "03"]:
        fmriprepdir_sub = pjoin(fmriprepdir, f"sub-{sub}")
        t1w = pjoin(
            fmriprepdir_sub,
            "anat",
            f"sub-{sub}_acq-prescannormalized_rec-pydeface_desc-preproc_T1w.nii.gz",
        )
        transform = pjoin(
            fmriprepdir_sub,
            "anat",
            f"sub-{sub}_acq-prescannormalized_rec-pydeface_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
        )
        apply = ApplyTransforms(
            input_image=glasser_atlas,
            reference_image=t1w,
            interpolation="NearestNeighbor",
            transforms=[transform],
            output_image=pjoin(outdir, f"sub-{sub}_glasser.nii.gz"),
        )
        print(f"projecting subject {sub}")
        apply.run()
    return None


def project_julian(
    parcel_basedir: str = pjoin("/LOCAL/ocontier/atlases", "julian_parcels"),
    fmriprepdir: str = pjoin(pardir, pardir, "derivatives", "fmriprep"),
    outdir: str = pjoin(
        pardir, pardir, "derivatives", "julian_parcels", "julian_parcels_projected"
    ),
) -> dict:
    """
    Project localizer parcels from Julian et al. (2012) to subjects' anatomical spaces.
    Returns outfiles as a nested dict:
        {'01':
        {'face_parcels': ['../../derivatives/julian_parcels/sub-01/face_parcels/sub-01_lOFA.nii.gz',
        '../../derivatives/julian_parcels/sub-01/face_parcels/sub-01_lSTS.nii.gz',
        ...],
        'scene_parcels': ['../../derivatives/julian_parcels/sub-01/scene_parcels/sub-01_rTOS.nii.gz',
        '../../derivatives/julian_parcels/sub-01/scene_parcels/sub-01_rRSC.nii.gz',
        ...]},
        '02': {...}}
    """
    outfiles = {}
    for sub in ["01", "02", "03"]:
        outfiles[sub] = {}
        fmriprepdir_sub = pjoin(fmriprepdir, f"sub-{sub}")
        t1w = pjoin(
            fmriprepdir_sub,
            "anat",
            f"sub-{sub}_acq-prescannormalized_rec-pydeface_desc-preproc_T1w.nii.gz",
        )
        transform = pjoin(
            fmriprepdir_sub,
            "anat",
            f"sub-{sub}_acq-prescannormalized_rec-pydeface_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
        )
        for parcel_class in [
            "face_parcels",
            "scene_parcels",
            "body_parcels",
            "object_parcels",
        ]:
            outfiles[sub][parcel_class] = []
            parcel_subdir = pjoin(parcel_basedir, parcel_class)
            out_subdir = pjoin(outdir, f"sub-{sub}", parcel_class)
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)
            imgs = glob.glob(pjoin(parcel_subdir, "*.img"))
            for img in imgs:
                outfile = pjoin(
                    out_subdir,
                    f"sub-{sub}_" + img.split("/")[-1].replace(".img", ".nii.gz"),
                )
                outfiles[sub][parcel_class].append(outfile)
                print("starting with ", outfile)
                apply = ApplyTransforms(
                    input_image=img,
                    reference_image=t1w,
                    interpolation="Linear",
                    transforms=[transform],
                    output_image=outfile,
                )
                apply.run()
    return outfiles


def intersect_localizer_with_parcels(
    sub: str = "01",
    parcels_basedir: str = pjoin(
        pardir, pardir, "derivatives", "julian_parcels", "julian_parcels_projected"
    ),
    localizer_results_dir: str = pjoin(pardir, pardir, "derivatives", "localizer"),
    out_basedir: str = pjoin(
        pardir, pardir, "derivatives", "julian_parcels", "julian_parcels_intersected"
    ),
    parcel_subdirs: tuple = (
        "face_parcels",
        "scene_parcels",
        "body_parcels",
        "object_parcels",
    ),
    contrast_subdirs: tuple = ("FFA_obj", "PPA_obj", "EBA_obj", "LOC_alt"),
    thr_julianroi: float = 0.5,  # threshold to binarize the resampled version of the Julian parcels
    thr_contrast: float = 3.7,
) -> dict:
    assert len(parcel_subdirs) == len(contrast_subdirs)
    outfiles = {"sub": sub}
    for parcel_subdir, cont_subdir in zip(parcel_subdirs, contrast_subdirs):
        outfiles[parcel_subdir] = []
        # get contrast and roi files, binarize contrast
        cont_file = pjoin(
            localizer_results_dir,
            f"sub-{sub}",
            f"contrast-{cont_subdir}",
            "zstat1_threshold.nii.gz",
        )
        cont_bin = math_img(f"img>{thr_contrast}", img=cont_file)
        roi_files = glob.glob(
            pjoin(parcels_basedir, f"sub-{sub}", parcel_subdir, "*.nii.gz")
        )
        assert len(roi_files) > 0
        for roi_file in roi_files:
            print("intersecting ", roi_file, " with contrast ", cont_file)
            # resample roi to functional space, binarize, intersect
            roi_res = resample_to_img(roi_file, cont_file, interpolation="linear")
            roi_bin = math_img(f"img>{thr_julianroi}", img=roi_res)
            intersected = intersect_masks([cont_bin, roi_bin], threshold=1)
            # save to file
            out_subdir = pjoin(out_basedir, f"sub-{sub}", parcel_subdir)
            outfile = pjoin(out_subdir, roi_file.split("/")[-1])
            outfiles[parcel_subdir].append(outfile)
            if not os.path.exists(pjoin(out_subdir)):
                os.makedirs(out_subdir)
            intersected.to_filename(outfile)
    return outfiles


def get_surfdata(subject: str, bidsroot: str) -> dict:
    # get all the anatomical surfaces for this subject
    anatdir = pjoin(bidsroot, "derivatives", "fmriprep", f"sub-{subject}", "anat")
    reconalldir = pjoin(bidsroot, "derivatives", "reconall", f"sub-{subject}")
    return {
        "pial_left": pjoin(
            anatdir,
            f"sub-{subject}_acq-prescannormalized_rec-pydeface_hemi-L_pial.surf.gii",
        ),
        "pial_right": pjoin(
            anatdir,
            f"sub-{subject}_acq-prescannormalized_rec-pydeface_hemi-R_pial.surf.gii",
        ),
        "infl_left": pjoin(
            anatdir,
            f"sub-{subject}_acq-prescannormalized_rec-pydeface_hemi-L_inflated.surf.gii",
        ),
        "infl_right": pjoin(
            anatdir,
            f"sub-{subject}_acq-prescannormalized_rec-pydeface_hemi-R_inflated.surf.gii",
        ),
        "sulc_left": pjoin(reconalldir, "surf", f"lh.sulc"),
        "sulc_right": pjoin(reconalldir, "surf", f"rh.sulc"),
        "wm_left": pjoin(reconalldir, "surf", "lh.white"),
        "wm_right": pjoin(reconalldir, "surf", "rh.white"),
    }


def load_category_df(
    cat_tsv: str = "/LOCAL/ocontier/thingsmri/thingsmri-metadata/Categories_final_20200131.tsv",
) -> pd.DataFrame:
    return pd.read_csv(cat_tsv, sep="\t").drop(
        columns="Definition (from WordNet, Google, or Wikipedia)"
    )


def psc(a: np.array, timeaxis: int = 0) -> np.array:
    """rescale array with fmri data to percent signal change (relative to the mean of each voxel time series)"""
    return 100 * ((a / a.mean(axis=timeaxis)) - 1)


def get_scicol(scicol_dir):
    """Get Scientific Color maps"""
    col_dirs = glob.glob(pjoin(scicol_dir, "[!+]*"))
    col_names = [cd.split("/")[-1] for cd in col_dirs]
    cdict = {}
    for cd, cn in zip(col_dirs, col_names):
        txt = glob.glob(pjoin(cd, "*.txt"))[0]
        cdict[cn] = LinearSegmentedColormap.from_list(cn, np.loadtxt(txt))
    return cdict


def get_render3_cmap(
    render3_file: str = "/LOCAL/ocontier/thingsmri/bids/code/external_libraries/render3.cmap",
):
    """Get FSL color map called 'render3' from a file taken from the FSL application contents."""
    return ListedColormap(np.loadtxt(render3_file))


def get_CMasher_as_listedcmap(cmr_name="iceburn", steps=500):
    """Get a colormap from the CMasher library and turn it into a matplotlib ListedColormap"""
    colors = cmr.take_cmap_colors(f"cmr.{cmr_name}", steps)
    return ListedColormap(colors, name=f"{cmr_name}")


def get_RdBu_r_alpha_cmap():
    """
    Modulate matplotlib RdBu_r color map by alpha (symmetrically)
    """
    cmap = pl.cm.RdBu_r
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap[:, -1] = np.hstack(
        [np.linspace(1, 0, int(cmap.N / 2)), np.linspace(0, 1, int(cmap.N / 2))]
    )
    return ListedColormap(my_cmap)


def xy_to_eccrad(x, y):
    """
    Convert two arrays of x and y coordinates to eccentricity, and various polar angle measure conventions.
    """
    ecc = np.sqrt(x**2 + y**2)
    pa = np.rad2deg(np.arctan2(y, x))
    return ecc, pa


def xy_to_various_angles(x, y):
    """
    Convert two arrays of x and y coordinates to eccentricity, and various polar angle measure conventions.
    Arrays can have any shape, since all computations are performed elementwise.
    returns
        ecc: eccentricity
        pa: signed polar angle (-180 to 180) with origin 0,1 ('east'), counterclockwise. Negative values are in the
            lower quadrants.
        compass: clockwise angles (0 to 360) with origin 1,0 ('north), clockwise.
        signed_n2s: signed north-to-south angles (-180 to 180) with origin 1,0 ('north'), clockwise. Negative values are
                    in the left quadrants.
        abs_n2s: unsigned (absolute) north-to-south angles (0 to 180). For use in neuropythy.
    """
    ecc = np.sqrt(x**2 + y**2)
    pa = np.rad2deg(np.arctan2(y, x))
    compass = (450 - pa) % 360
    signed_n2s = np.copy(compass)
    signed_n2s[np.where(signed_n2s > 180)] = (
        signed_n2s[np.where(signed_n2s > 180)] - 360
    )
    abs_n2s = np.abs(signed_n2s)
    return ecc, pa, compass, signed_n2s, abs_n2s


def ci_array(a, confidence=0.95, alongax=0):
    """Returns tuple of upper and lower CI for mean along some axis in multidimensional array"""
    m, se = np.mean(a), sem(a, axis=alongax)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, a.shape[alongax] - 1)
    return m - h, m + h


def call_vol2surf(sub: str, volumefile: str, outfile: str, outdir: str):
    """Produces two outfiles (lh.outfile.mgz, rh.outfile.mgz)"""
    for hemi in ["lh", "rh"]:
        subprocess.run(
            [
                "mri_vol2surf",
                "--src",
                volumefile,
                "--out",
                pjoin(outdir, f"{hemi}.{outfile}.mgz"),
                "--regheader",
                f"sub-{sub}",
                "--hemi",
                hemi,
            ]
        )


def calc_hfc(timeseries: np.ndarray, tr: float = 1.5):
    """Calculate high frequency content for time series data. Tr can generally mean sampling rate in seconds."""
    nf = (1.0 / tr) * 0.5  # nyquist
    freqs, power = periodogram(timeseries, fs=1.0 / tr)
    relcumsum = np.cumsum(power) / power.sum()
    freqind = np.argmin(np.absolute(relcumsum - 0.5))
    hfc = freqs[freqind] / nf
    return hfc


def pearsonr_nd(arr1: np.ndarray, arr2: np.ndarray, alongax: int = 0) -> np.ndarray:
    """
    Pearson correlation between respective variables in two arrays.
    arr1 and arr2 are 2d arrays. Rows correspond to observations, columns to variables.
    Returns:
        correlations: np.ndarray (shape nvariables)
    """
    # center each feature
    arr1_c = arr1 - arr1.mean(axis=alongax)
    arr2_c = arr2 - arr2.mean(axis=alongax)
    # get sum of products for each voxel (numerator)
    numerators = np.sum(arr1_c * arr2_c, axis=alongax)
    # denominator
    arr1_sds = np.sqrt(np.sum(arr1_c**2, axis=alongax))
    arr2_sds = np.sqrt(np.sum(arr2_c**2, axis=alongax))
    denominators = arr1_sds * arr2_sds
    return numerators / denominators


def r2_ndarr(x, y, alongax=-1):
    """Calculate the coefficient of determination in y explained by x"""
    ssres = np.nansum(np.square(y - x), axis=alongax)
    sstot = np.nansum(
        np.square(y - np.expand_dims(y.mean(axis=alongax), axis=alongax)), axis=alongax
    )
    return 100 * (1 - np.nan_to_num(ssres) / np.nan_to_num(sstot))


def vcorrcoef(X, y):
    """
    Calculate correlation between a vector y (size 1 x k) and each row in a matrix X (size N x k).
    Returns a vector of correlation coefficients r (size N x 1).
    """
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    return r


def get_hrflib(
    hrflib_url: str,
    rescale_amplitude: bool = True,
    resample_to_tr: bool = False,
    tr: float = 1.5,
    dtype=np.single,
) -> np.ndarray:
    """
    Get HRF library from Kendrick Kay's github repository.
    optionally rescale amplitudes of all HRFs to 1 (recommended) and resample to a specific TR (not recommended).
    """
    hrflib = np.array(df_from_url(hrflib_url, sep="\t", header=None))
    if resample_to_tr:  # resample to our TR
        sampleinds = np.arange(0, hrflib.shape[0], tr * 10, dtype=np.int16)
        hrflib = hrflib[sampleinds, :]
    if rescale_amplitude:  # rescale all HRFs to a peak amplitude of 1
        hrflib = hrflib / np.max(hrflib, axis=0)
    return hrflib.astype(dtype)


def regress_out(
    x: np.ndarray,
    y: np.ndarray,
    dtype=np.single,
    lr_kws: dict = dict(copy_X=True, fit_intercept=True, normalize=True, n_jobs=-1),
) -> np.ndarray:
    reg = LinearRegression(**lr_kws)
    reg.fit(x, y)
    resid = y - reg.predict(x)
    return resid.astype(dtype)


def spearman_brown(corrs: np.ndarray):
    """Spearman-Brown correction for split-half reliability. <corrs> can be any shape."""
    return (2 * corrs) / (1 + corrs)


def match_scale(
    original: np.ndarray, target: np.ndarray, alongax: int = 0
) -> np.ndarray:
    """
    Rescale 'original' to match the mean and variance of 'target'.
    The result will be the values of 'original', distributed with  mean and variance of 'target'.
    Both input arrays can have any shape, 'alongax' specifies the axis along which mean and SD is considered.
    """
    matched = (original / original.std(axis=alongax)) * target.std(axis=alongax)
    return matched



def get_prf_rois(sub, bidsroot, prf_derivname) -> dict:
    """
    Get file names of early visual ROIs deliniated with neuropythy.
    """
    rois = {}
    prf_dir = pjoin(bidsroot, "derivatives", prf_derivname, f"sub-{sub}")
    for va in range(1, 4):
        rois[f"V{va}"] = pjoin(prf_dir, f"resampled_va-{va}_interp-nn.nii.gz")
    rois["hV4"] = pjoin(prf_dir, f"resampled_va-4_interp-nn.nii.gz")
    rois["VO1"] = pjoin(prf_dir, f"resampled_va-5_interp-nn.nii.gz")
    rois["VO2"] = pjoin(prf_dir, f"resampled_va-6_interp-nn.nii.gz")
    rois["LO1 (prf)"] = pjoin(prf_dir, f"resampled_va-7_interp-nn.nii.gz")
    rois["LO2 (prf)"] = pjoin(prf_dir, f"resampled_va-8_interp-nn.nii.gz")
    rois["TO1"] = pjoin(prf_dir, f"resampled_va-9_interp-nn.nii.gz")
    rois["TO2"] = pjoin(prf_dir, f"resampled_va-10_interp-nn.nii.gz")
    rois["V3b"] = pjoin(prf_dir, f"resampled_va-11_interp-nn.nii.gz")
    rois["V3a"] = pjoin(prf_dir, f"resampled_va-12_interp-nn.nii.gz")
    return rois


def get_category_rois(sub, bidsroot, julian_derivname) -> dict:
    """
    Get file names of category seletive ROIS determined with a GLM on the localizer data.
    """
    julian_dir = pjoin(bidsroot, "derivatives", julian_derivname, f"sub-{sub}")
    rois = {}
    roinames = [
        "lEBA",
        "rEBA",
        "lFFA",
        "rFFA",
        "lOFA",
        "rOFA",
        "lSTS",
        "rSTS",
        "lPPA",
        "rPPA",
        "lRSC",
        "rRSC",
        "lTOS",
        "rTOS",
    ]  # LOC added manually below
    for roiname in roinames:
        found = glob.glob(pjoin(julian_dir, "*", f"*{roiname}*"))
        if found:
            rois[roiname] = found[0]
    for hemi in ["l", "r"]:  # LOC is in different directory
        rois[f"{hemi}LOC"] = pjoin(
            julian_dir.replace("_edited", "_intersected"),
            "object_parcels",
            f"sub-{sub}_{hemi}LOC.nii.gz",
        )
    return rois


def get_all_roi_files(sub, bidsroot, prf_derivname, julian_derivname) -> dict:
    """
    Returns a dict with roinames as keys and file names as values.
    category ROIs are separate per hemisphere, PRF rois are bihemispheric.
    """
    prf_rois = get_prf_rois(sub, bidsroot, prf_derivname)
    cat_rois = get_category_rois(sub, bidsroot, julian_derivname)
    # combine two dictionaries
    rois = {**prf_rois, **cat_rois}
    return rois


def get_unionroi_categoryselective(
    sub, bidsroot, julian_derivname="julian_parcels/julian_parcels_edited"
) -> nilearn.image:
    """
    Get a union mask including all category selective ROIs (FFA, PPA, etc.) and LOC.
    """
    # get category selective areas as a mask
    julian_dir = pjoin(bidsroot, "derivatives", julian_derivname, f"sub-{sub}")
    roi_files = glob.glob(pjoin(julian_dir, "*", "*.nii.gz"))
    loc_files = glob.glob(
        pjoin(julian_dir.replace("edited", "intersected"), "object_parcels", "*.nii.gz")
    )
    roi_files += loc_files
    mask = intersect_masks(roi_files, threshold=0, connected=False)
    return mask


def apply_mask_smoothed(nifti, mask, smoothing=0.0, dtype=np.single):
    return apply_mask(smooth_img(nifti, smoothing), mask, dtype=dtype)
