"""
Code for estimating retinotopic parameters with AFNIs circular population
receptive field mapping and neuropythy to further refine these estimates and
delineate retinotopic regions.

Requires a preprocessed version of the dataset stored in bidsdata/derivatives/fmriprep. Further non-python dependencies: FSL, AFNI

Usage:
python prf.py <subject_ID> <bids_path> <afni_output_path>
"""

import os
import subprocess
import sys
import warnings
from os import pardir
from os.path import abspath
from os.path import join as pjoin
from shutil import copyfile

import nibabel as nib
import numpy as np
from nilearn.image import load_img, new_img_like, index_img, resample_to_img
from nilearn.masking import intersect_masks, apply_mask, unmask
from nipype.interfaces.fsl.maths import TemporalFilter
from nipype.pipeline.engine import MapNode
from scipy.stats import zscore
from tqdm import tqdm

from thingsmri.utils import xy_to_eccrad, xy_to_various_angles, call_vol2surf


class ThingsPrfAfni:
    """
    Runs AFNIs PRF analysis on the THINGS-fMRI data.
    Requires fMRIPREP output of localizer sessions and afni brik files specifying the stimulus.
    Preprocessing (in addition to fmriprep) includes temporal filtering of individual runs, averaging them,
    and zscoring the resulting average time series. This pipeline creates a temporary working directory in the parent
    directory of the bids dataset. The results are saved to the derivatives section of the bids dataset.

    Example:
        prfdata = ThingsPrfAfni(bidsroot=pjoin(pardir, pardir, pardir), subject='01)
        prfdata.run()
    """

    def __init__(
        self,
        bidsroot,
        subject,
        stimulus_brik,
        stimulus_head,
        conv_fname,
        sesmap={"01": "localizer2", "02": "localizer2", "03": "localizer1"},
        fmriprep_deriv_name="fmriprep",
        out_deriv_name="prf_afni",
        preproc_hpf=100.0,
        stimdur: float = 3.0,
    ):
        self.bidsroot = abspath(bidsroot)
        self.subject = subject
        self.sesmap = sesmap
        self.conv_fname = conv_fname
        self.stimbrik = stimulus_brik
        self.stimhead = stimulus_head
        self.wdir = pjoin(self.bidsroot, pardir, "prf_afni_wdir", f"sub-{subject}")
        self.outdir = pjoin(
            self.bidsroot, "derivatives", out_deriv_name, f"sub-{subject}"
        )
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.data = pjoin(self.wdir, "data.nii.gz")
        if not os.path.exists(self.wdir):
            os.makedirs(self.wdir)
        else:
            warnings.warn(
                f"working directory already exists. Files may be overwritten in\n{self.wdir}"
            )
        self.copy_stimbrik()
        self.nruns = 4
        self.tr = 1.5
        self.stimwidth_pix = 990
        self.ppd = 90
        self.hpf = preproc_hpf
        self.ses = sesmap[subject]
        self.fmriprep_deriv_name = fmriprep_deriv_name
        self.fmriprep_dir = pjoin(
            self.bidsroot,
            "derivatives",
            fmriprep_deriv_name,
            f"sub-{subject}",
            f"ses-{self.ses}",
            "func",
        )
        self.boldfiles_fprep, self.maskfiles_fprep = self.get_file_names()
        self.umask_f = pjoin(self.wdir, "unionmask.nii.gz")
        self.stmidur = stimdur
        self.buck_map = {
            "amp": 0,
            "x": 1,
            "y": 2,
            "sigma": 3,
            "rsquared": 10,
        }  # meaning of sub bricks in afni output

    def get_file_names(self):
        boldfiles_fprep = [
            pjoin(
                self.fmriprep_dir,
                f"sub-{self.subject}_ses-{self.ses}_task-pRF_run-{runi}_space-T1w_desc-preproc_bold.nii.gz",
            )
            for runi in range(1, self.nruns + 1)
        ]
        maskfiles_fprep = [
            pjoin(
                self.fmriprep_dir,
                f"sub-{self.subject}_ses-{self.ses}_task-pRF_run-{runi}_space-T1w_desc-brain_mask.nii.gz",
            )
            for runi in range(1, self.nruns + 1)
        ]
        for l in [boldfiles_fprep, maskfiles_fprep]:
            for e in l:
                if not os.path.exists(e):
                    raise IOError(f"Could not find file\n{e}")
        return boldfiles_fprep, maskfiles_fprep

    def preproc_data(self):
        # temporal filtering with fsl
        tf = MapNode(
            TemporalFilter(highpass_sigma=self.hpf / self.tr),
            name="TemporalFilter",
            iterfield=["in_file"],
        )
        tf.inputs.in_file = self.boldfiles_fprep
        tf.run()
        filterd_boldfiles = tf.get_output("out_file")
        # create a brain mask
        umask = intersect_masks(self.maskfiles_fprep, threshold=0)
        umask.to_filename(self.umask_f)
        # load in numpy, average, z score
        bold_arrs = [
            apply_mask(bp, umask)
            for bp in tqdm(filterd_boldfiles, desc="loading runs for averaging")
        ]
        data_arr = zscore(np.mean(np.stack(bold_arrs), axis=0), axis=0)
        # save to working directory
        data_img = unmask(data_arr, umask)
        data_img.to_filename(self.data)

    def copy_stimbrik(self):
        """to working directory"""
        for f in [self.stimbrik, self.stimhead]:
            copyfile(f, pjoin(self.wdir, os.path.basename(f)))

    def set_afni_environ(self):
        os.environ["AFNI_CONVMODEL_REF"] = self.conv_fname
        os.environ["AFNI_MODEL_PRF_STIM_DSET"] = self.stimhead.replace("HEAD", "")
        os.environ["AFNI_MODEL_PRF_ON_GRID"] = "YES"
        os.environ["AFNI_MODEL_DEBUG"] = "3"

    def run_afni(self):
        os.chdir(self.wdir)
        subprocess.run(["3dcopy", self.data, "AFNIpRF+orig."])
        subprocess.run(["3dcopy", self.umask_f, "automask+orig."])
        self.set_afni_environ()
        subprocess.run(
            [
                "3dNLfim",
                "-input",
                "AFNIpRF+orig.",
                "-mask",
                "automask+orig.",
                "-noise",
                "Zero",
                "-signal",
                "Conv_PRF",
                "-sconstr",
                "0",
                "-20.0",
                "20.0",
                "-sconstr",
                "1",
                "-1.0",
                "1.0",
                "-sconstr",
                "2",
                "-1.0",
                "1.0",
                "-sconstr",
                "3",
                "0.0",
                "1.0",
                "-BOTH",
                "-nrand",
                "10000",
                "-nbest",
                "5",
                "-bucket",
                "0",
                "Buck.PRF",
                "-snfit",
                "snfit.PRF",
                "-TR",
                str(self.tr),
                "-float",
            ]
        )
        for k, v in self.buck_map.items():
            subprocess.run(
                ["3dAFNItoNIFTI", "-prefix", f"{k}.nii.gz", f"Buck.PRF+orig.[{v}]"]
            )

    def unit_to_dva(self, in_file, out_file):
        """Take a nifti that encodes x,y, or sigma in unit measures (0-1) and convert to degree visual angle"""
        in_img = nib.load(in_file)
        arr = in_img.get_fdata()
        dva_arr = (arr * self.stimwidth_pix) / self.ppd
        out_img = nib.Nifti1Image(dva_arr, affine=in_img.affine)
        out_img.to_filename(out_file)
        return dva_arr

    def convert_afni_output(self):
        # convert to dva, overwriting afni output which is in unit measures
        os.chdir(self.wdir)
        for stat in ["x", "y", "sigma"]:
            _ = self.unit_to_dva(f"{stat}.nii.gz", f"{stat}.nii.gz")
        x_img, y_img = nib.load("x.nii.gz"), nib.load("y.nii.gz")
        x_arr, y_arr = x_img.get_fdata(), y_img.get_fdata()
        # save eccentricity and polar angle directly to derivatives
        ecc_arr, pa_arr = xy_to_eccrad(x_arr, y_arr)
        for arr, name in zip([ecc_arr, pa_arr], ["ecc", "pa"]):
            img = nib.Nifti1Image(arr, affine=x_img.affine)
            img.to_filename(pjoin(self.outdir, f"{name}.nii.gz"))

    def copy_results(self):
        """From working directory to bids derivatives"""
        for k in self.buck_map.keys():
            copyfile(pjoin(self.wdir, f"{k}.nii.gz"), pjoin(self.outdir, f"{k}.nii.gz"))
        for prefix in ["snfit.PRF+orig", "Buck.PRF+orig"]:
            for suffix in ["HEAD", "BRIK"]:
                copyfile(
                    pjoin(self.wdir, ".".join([prefix, suffix])),
                    pjoin(self.outdir, ".".join([prefix, suffix])),
                )

    def run(self):
        if not os.path.exists(self.data):
            self.preproc_data()
        if not os.path.exists(pjoin(self.wdir, "Buck.PRF+orig.BRIK")):
            self.run_afni()
        self.convert_afni_output()
        self.copy_results()


def list_afni_outputs(sub, afnioutdir):
    """Get file names of afni_prf output"""
    return dict(
        ecc=pjoin(afnioutdir, f"sub-{sub}", "ecc.nii.gz"),
        rsquared=pjoin(afnioutdir, f"sub-{sub}", "rsquared.nii.gz"),
        sigma=pjoin(afnioutdir, f"sub-{sub}", "sigma.nii.gz"),
        x=pjoin(afnioutdir, f"sub-{sub}", "x.nii.gz"),
        y=pjoin(afnioutdir, f"sub-{sub}", "y.nii.gz"),
    )


def run_neuropythy(
    sub,
    afnioutir,
    base_outdir,
    base_wdir,
    meanruns_dir,
    reconall_dir,
    flip_y=True,
):
    wdir = pjoin(base_wdir, f"sub-{sub}")
    outdir = pjoin(base_outdir, f"sub-{sub}")
    for d in [wdir, outdir]:
        if not os.path.exists(d):
            os.makedirs(d)
    afniresults = list_afni_outputs(sub, afnioutir)
    x_arr = load_img(afniresults["x"]).get_fdata()
    y_arr = load_img(afniresults["y"]).get_fdata()
    if flip_y:
        y_arr *= -1
    _, _, _, _, pa_arr = xy_to_various_angles(x_arr, y_arr)
    pa_img = new_img_like(load_img(afniresults["x"]), pa_arr)
    pa_img.to_filename(pjoin(wdir, "pa.nii.gz"))
    afniresults["pa"] = pjoin(wdir, "pa.nii.gz")
    # project afni results to surface
    for inname, outname in zip(
        ["pa", "rsquared", "ecc", "sigma"],
        ["prf_angle", "prf_vexpl", "prf_eccen", "prf_radius"],
    ):
        call_vol2surf(sub, afniresults[inname], outname, wdir)
    # run neuropythy
    subprocess.run(
        [
            "python",
            "-m",
            "neuropythy",
            "register_retinotopy",
            f"sub-{sub}",
            "--verbose",
            f"--lh-eccen={pjoin(wdir, 'lh.prf_eccen.mgz')}",
            f"--rh-eccen={pjoin(wdir, 'rh.prf_eccen.mgz')}",
            f"--lh-angle={pjoin(wdir, 'lh.prf_angle.mgz')}",
            f"--rh-angle={pjoin(wdir, 'rh.prf_angle.mgz')}",
            f"--lh-weight={pjoin(wdir, 'lh.prf_vexpl.mgz')}",
            f"--rh-weight={pjoin(wdir, 'rh.prf_vexpl.mgz')}",
            f"--lh-radius={pjoin(wdir, 'lh.prf_radius.mgz')}",
            f"--rh-radius={pjoin(wdir, 'rh.prf_radius.mgz')}",
        ]
    )
    # copy and reorient the neuropythy output
    os.chdir(pjoin(reconall_dir, f"sub-{sub}", "mri"))
    for param in ["angle", "eccen", "sigma", "varea"]:
        subprocess.run(
            [
                "mri_convert",
                f"inferred_{param}.mgz",
                pjoin(outdir, f"inferred_{param}_fsorient.nii.gz"),
            ]
        )
        subprocess.run(
            [
                "fslreorient2std",
                pjoin(outdir, f"inferred_{param}_fsorient.nii.gz"),
                pjoin(outdir, f"inferred_{param}.nii.gz"),
            ]
        )
    # resample binary visual ROIs to functional space
    ref_f = pjoin(meanruns_dir, f"sub-{sub}.nii.gz")
    ref_img = index_img(ref_f, 0)
    va_img = load_img(pjoin(outdir, "inferred_varea.nii.gz"))
    va = va_img.get_fdata()
    for roival in range(1, 13):
        roi_arr = np.zeros(va.shape)
        roi_arr[np.where(va == roival)] = 1.0
        roi_img = new_img_like(va_img, roi_arr)
        res_img = resample_to_img(roi_img, ref_img, interpolation="nearest")
        res_img.to_filename(pjoin(outdir, f"resampled_va-{roival}_interp-nn.nii.gz"))
        # also save linear interpolation for convenience
        lres_img = resample_to_img(roi_img, ref_img, interpolation="linear")
        lres_img.to_filename(
            pjoin(outdir, f"resampled_va-{roival}_interp-linear.nii.gz")
        )

    # resample all other outputs too
    for param in ["angle", "eccen", "sigma", "varea"]:
        interp = "nearest" if param == "varea" else "linear"
        res_img = resample_to_img(
            pjoin(outdir, f"inferred_{param}.nii.gz"), ref_img, interpolation=interp
        )
        res_img.to_filename(pjoin(outdir, f"resampled_{param}.nii.gz"))


def threshold_vareas_by_eccentricity(
    meanruns_dir="/Users/olivercontier/bigfri/scratch/bids/derivatives/mean_runs",
    max_eccen=15.55,
    npt_bdir="/Users/olivercontier/bigfri/scratch/bids/derivatives/prf_neuropythy",
    out_bdir="/Users/olivercontier/bigfri/scratch/bids/derivatives/prf_neuropythy/fixated",
):
    """
    because neuropythy infers receptive field parameters outside the stimulated region of the visual field
    """
    for sub in tqdm(range(1, 4), desc="subjects"):
        npt_outdir = pjoin(npt_bdir, f"sub-0{sub}")
        thresh_outdir = pjoin(out_bdir, f"sub-0{sub}")
        if not os.path.exists(thresh_outdir):
            os.makedirs(thresh_outdir)
        # anatomical resolution, but reoriented to standard (unlike fsorient)
        eccen_orig_f = pjoin(npt_outdir, "inferred_eccen.nii.gz")
        eccen_orig_img = load_img(eccen_orig_f)
        eccen = eccen_orig_img.get_fdata()
        periphery_mask = eccen > max_eccen
        # reference image for resampling
        ref_f = pjoin(meanruns_dir, f"sub-0{sub}.nii.gz")
        ref_img = index_img(ref_f, 0)
        # threshold vareas
        va_img = load_img(pjoin(npt_outdir, "inferred_varea.nii.gz"))
        va = va_img.get_fdata()
        va[periphery_mask] = 0.0
        va_thr_img = new_img_like(va_img, va)
        va_thr_img.to_filename(pjoin(thresh_outdir, "inferred_varea.nii.gz"))
        # resample to functional space and save rois individually
        for roival in tqdm(range(1, 13), desc="rois", leave=False):
            roi_arr = np.zeros(va.shape)
            roi_arr[np.where(va == roival)] = 1.0
            roi_img = new_img_like(va_img, roi_arr)
            res_img = resample_to_img(roi_img, ref_img, interpolation="nearest")
            res_img.to_filename(
                pjoin(thresh_outdir, f"resampled_va-{roival}_interp-nn.nii.gz")
            )
            lres_img = resample_to_img(roi_img, ref_img, interpolation="linear")
            lres_img.to_filename(
                pjoin(thresh_outdir, f"resampled_va-{roival}_interp-linear.nii.gz")
            )
    return None


if __name__ == "__main__":
    import sys

    sub, bidsroot, afni_inputs_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    # run PRF estimation with AFNI
    conv_fname = pjoin(afni_inputs_dir, "conv.ref.spmg1_manual.1D")
    stimulus_brik = pjoin(afni_inputs_dir, "stim.308.LIA.bmask.resam+orig.BRIK")
    stimulus_head = pjoin(afni_inputs_dir, "stim.308.LIA.bmask.resam+orig.HEAD")

    afniprf = ThingsPrfAfni(
        bidsroot,
        sub,
        stimulus_brik,
        stimulus_head,
        conv_fname,
    )
    afniprf.run()
