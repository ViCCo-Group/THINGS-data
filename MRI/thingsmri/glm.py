import os
from os.path import join as pjoin
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import load_img, concat_imgs
from nilearn.masking import apply_mask, intersect_masks
from scipy.stats import zscore
from tqdm import tqdm

from thingsmri.dataset import ThingsMRIdataset
from thingsmri.utils import psc


def get_nuisance_df(noiseregs, nuisance_tsv, include_all_aroma=False):
    """Make pd.DataFrame based on list of desired noise regressors and a nuisance_tsv file returned by fmriprep"""
    noiseregs_copy = noiseregs[:]
    nuisance_df = pd.read_csv(nuisance_tsv, sep="\t")
    if include_all_aroma:
        noiseregs_copy += [c for c in nuisance_df.columns if "aroma" in c]
    nuisance_df = nuisance_df[noiseregs_copy]
    if "framewise_displacement" in noiseregs_copy:
        nuisance_df["framewise_displacement"] = nuisance_df[
            "framewise_displacement"
        ].fillna(0)
    return nuisance_df


def df_to_boxcar_design(
    design_df: pd.DataFrame, frame_times: np.ndarray, add_constant: bool = False
) -> pd.DataFrame:
    """
    Make boxcar design matrix from data frame with one regressor for each trial_type (and no constant).
    CAVEAT: nilearn sorts the conditions alphabetically, not by onset.
    """
    dropcols = [] if add_constant else ["constant"]
    trialtypes = design_df["trial_type"].unique().tolist()
    designmat = make_first_level_design_matrix(
        frame_times=frame_times,
        events=design_df,
        hrf_model=None,
        drift_model=None,
        high_pass=None,
        drift_order=None,
        oversampling=1,
    ).drop(columns=dropcols)
    return designmat[trialtypes]


def load_masked(bold_file, mask, rescale="psc", dtype=np.single):
    if rescale == "psc":
        return np.nan_to_num(psc(apply_mask(bold_file, mask, dtype=dtype)))
    elif rescale == "z":
        return np.nan_to_num(
            zscore(apply_mask(bold_file, mask, dtype=dtype), nan_policy="omit", axis=0)
        )
    elif rescale == "center":
        data = np.nan_to_num(apply_mask(bold_file, mask, dtype=dtype))
        data -= data.mean(axis=0)
    else:
        return apply_mask(bold_file, mask, dtype=dtype)


class THINGSGLM(object):
    """
    Parent class for different GLMs to run on the things mri dataset,
    mostly handling IO.
    """

    def __init__(
        self,
        bidsroot: str,
        subject: str,
        out_deriv_name: str = "glm",
        noiseregs: list = [
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "framewise_displacement",
        ],
        acompcors: bool or int = 10,
        include_all_aroma: bool = False,
        # include_manual_ica: bool = False,
        hrf_model: str or None = "spm + derivative",
        noise_model: str = "ols",
        high_pass: float = 0.01,
        sigscale_nilearn: bool or int or tuple = False,
        standardize: bool = True,
        verbosity: int = 3,
        nruns_perses: int = 10,
        nprocs: int = 1,
        lowmem=False,
        ntrs: int = 284,
        tr: float = 1.5,
        drift_model: str = "cosine",
        drift_order: int = 4,
        fwhm: bool or None = None,
        overwrite: bool = False,
        stc_reftime: float = 0.75,
    ):
        self.bidsroot = os.path.abspath(bidsroot)
        self.include_all_aroma = include_all_aroma
        # self.include_manual_ica = include_manual_ica
        self.subject = subject
        self.out_deriv_name = out_deriv_name
        self.verbosity = verbosity
        self.lowmem = lowmem
        self.nprocs = nprocs
        self.acompcors = acompcors
        self.tr = tr
        self.ntrs = ntrs
        self.nruns_perses = nruns_perses
        self.high_pass = high_pass
        self.hrf_model = hrf_model
        self.noise_model = noise_model
        self.drift_model = drift_model
        self.drift_order = drift_order
        self.sigscale_nilearn = sigscale_nilearn
        self.standardize = standardize
        self.fwhm = fwhm
        self.stc_reftime = (
            stc_reftime  # fmriprep interpolates to mean of all slice times
        )
        self.overwrite = overwrite
        self.ds = ThingsMRIdataset(self.bidsroot)
        self.n_sessions = len(self.ds.things_sessions)
        self.nruns_total = self.n_sessions * self.nruns_perses
        self.subj_prepdir = pjoin(
            bidsroot, "derivatives", "fmriprep", f"sub-{self.subject}"
        )
        self.subj_outdir = pjoin(
            bidsroot, "derivatives", out_deriv_name, f"sub-{self.subject}"
        )
        self.icalabelled_dir = pjoin(
            bidsroot, "derivatives", "ICAlabelled", f"sub-{self.subject}"
        )
        if not os.path.exists(self.subj_outdir):
            os.makedirs(self.subj_outdir)
        if acompcors:
            noiseregs += [f"a_comp_cor_{i:02}" for i in range(self.acompcors)]
        self.noiseregs = noiseregs
        self.frame_times_tr = (
            np.arange(0, self.ntrs * self.tr, self.tr) + self.stc_reftime
        )
        # get image dimensions
        example_img = load_img(
            self.ds.layout.get(
                session="things01", extension="nii.gz", suffix="bold", subject="01"
            )[0].path
        )
        self.nx, self.ny, self.nz, self.ntrs = example_img.shape
        self.n_samples_total, self.nvox_masked, self.union_mask = None, None, None

    def _get_events_files(self):
        return self.ds.layout.get(task="things", subject=self.subject, suffix="events")

    def _get_bold_files(self):
        bold_files = [
            pjoin(
                self.subj_prepdir,
                f"ses-{sesname}",
                "func",
                f"sub-{self.subject}_ses-{sesname}_task-things_run-{run_i + 1}_space-T1w_desc-preproc_bold.nii.gz",
            )
            for sesname in self.ds.things_sessions
            for run_i in range(10)
        ]
        for boldfile in bold_files:
            assert os.path.exists(boldfile), f"\nboldfile not found:\n{boldfile}\n"
        return bold_files

    def _get_masks(self):
        masks = [
            pjoin(
                self.subj_prepdir,
                f"ses-{sesname}",
                "func",
                f"sub-{self.subject}_ses-{sesname}_task-things_run-{run_i + 1}_space-T1w_desc-brain_mask.nii.gz",
            )
            for sesname in self.ds.things_sessions
            for run_i in range(10)
        ]
        for mask in masks:
            assert os.path.exists(mask), f"\nmask not found:\n{mask}\n"
        return masks

    def _get_nuisance_tsvs(self):
        nuisance_tsvs = [
            pjoin(
                self.subj_prepdir,
                f"ses-{sesname}",
                "func",
                f"sub-{self.subject}_ses-{sesname}_task-things_run-{run_i + 1}_desc-confounds_timeseries.tsv",
            )
            for sesname in self.ds.things_sessions
            for run_i in range(10)
        ]
        for tsv in nuisance_tsvs:
            assert os.path.exists(tsv), f"\nnuisance tsv not found:\n{tsv}\n"
        return nuisance_tsvs

    def _get_ica_txts(self):
        ica_txts = [
            pjoin(
                self.icalabelled_dir,
                f"ses-{sesname}",
                f"sub-{self.subject}_ses-{sesname}_task-things_run-{run_i + 1:02d}.txt",
            )
            for sesname in self.ds.things_sessions
            for run_i in range(10)
        ]
        for txt in ica_txts:
            assert os.path.exists(txt), f"\nica tsv not found:\n{txt}\n"
        return ica_txts

    def get_inputs(self):
        event_files = self._get_events_files()
        bold_files = self._get_bold_files()
        masks = self._get_masks()
        nuisance_tsvs = self._get_nuisance_tsvs()
        assert (
            len(event_files) == len(bold_files) == len(nuisance_tsvs) == len(masks)
        ), f"\ninputs have unequal length\n"
        return event_files, bold_files, nuisance_tsvs, masks

    def add_union_mask(self, masks):
        """Create a union mask based on the run-wise brain masks"""
        self.union_mask = intersect_masks(masks, threshold=0)

    def vstack_data_masked(
        self, bold_files, rescale_runwise="psc", rescale_global="off", dtype=np.single
    ):
        arrs = Parallel(n_jobs=20)(
            delayed(load_masked)(bf, self.union_mask, rescale_runwise, dtype)
            for bf in bold_files
        )
        if rescale_global == "psc":
            data = np.nan_to_num(psc(np.vstack(arrs)))
        elif rescale_global == "z":
            data = np.nan_to_num(zscore(np.vstack(arrs), nan_policy="omit", axis=0))
        elif rescale_global:
            data = np.vstack(arrs)
            data -= data.mean(axis=0)
        else:
            data = np.vstack(arrs)
        self.nvox_masked = data.shape[1]
        return data.astype(dtype)

    def load_data_concat_volumes(self, bold_files):
        print("concatinating bold files")
        bold_imgs = [
            nib.Nifti2Image.from_image(load_img(b))
            for b in tqdm(bold_files, "loading nifti files")
        ]
        return concat_imgs(bold_imgs, verbose=self.verbosity)

    def init_glm(self, mask):
        print(f"instantiating model with nprocs: {self.nprocs}")
        return FirstLevelModel(
            minimize_memory=self.lowmem,
            mask_img=mask,
            verbose=self.verbosity,
            noise_model=self.noise_model,
            t_r=self.tr,
            standardize=self.standardize,
            signal_scaling=self.sigscale_nilearn,
            n_jobs=self.nprocs,
            smoothing_fwhm=self.fwhm,
        )
