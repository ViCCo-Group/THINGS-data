"""
Implementation of the ICA denoising procedure.

- run_melodic_wf runs the additional preprocesisng and ICA.
- EvalMelodic is used to calculate each ICs features and generate visual reports for the raters.
- write_noise_tsvs then produces the final noise regressors based on the chosen thresholds.

This module is meant for documentation not intended for re-running since it relies on labels given by two human raters.
"""

import os
from os import pardir
from os.path import abspath
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
from matplotlib.gridspec import GridSpec
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import load_img, threshold_img, math_img, resample_to_img
from nilearn.masking import intersect_masks
from nipype.interfaces.fsl.maths import TemporalFilter
from nipype.interfaces.fsl.model import MELODIC
from nipype.interfaces.fsl.preprocess import SUSAN
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import MapNode, Workflow, Node
from numpy.random import choice as rndchoice
from scipy.ndimage.morphology import binary_erosion
from scipy.signal import periodogram
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from thingsmri.dataset import ThingsMRIdataset
from thingsmri.utils import calc_hfc, get_render3_cmap


def grab_data(subject, bidsroot, fmriprepdir, derivdir, space="func_preproc"):
    ds = ThingsMRIdataset(bidsroot)
    bidsobjs = ds.layout.get(subject=subject, suffix="bold", extension=".nii.gz")
    space_str = {"T1w": "_space-T1w", "func_preproc": ""}
    assert space in space_str
    boldfiles = [
        abspath(
            pjoin(
                fmriprepdir,
                f"ses-{bidsobj.entities['session']}",
                "func",
                f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_run-{bidsobj.entities['run']}{space_str[space]}_desc-preproc_bold.nii.gz",
            )
        )
        for bidsobj in bidsobjs
    ]
    masks = [
        abspath(
            pjoin(
                fmriprepdir,
                f"ses-{bidsobj.entities['session']}",
                "func",
                f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_run-{bidsobj.entities['run']}{space_str[space]}_desc-brain_mask.nii.gz",
            )
        )
        for bidsobj in bidsobjs
    ]
    outdirs = [
        abspath(
            pjoin(
                derivdir,
                f"space-{space}",
                f"sub-{subject}",
                f"ses-{bidsobj.entities['session']}",
                f"sub-{subject}_ses-{bidsobj.entities['session']}_task-{bidsobj.entities['task']}_run-{bidsobj.entities['run']}_melodic",
            )
        )
        for bidsobj in bidsobjs
    ]
    for o in outdirs:
        if not os.path.exists(o):
            os.makedirs(o)
    return boldfiles, masks, outdirs


def calc_susan_thresh(boldfile, maskfile, timeax=0, median_factor=0.75):
    """
    Calculate the median value within brainmask and multiply with fixed factor to get an estimate of the contrast
    between background and brain for FSL's SUSAN.
    """
    from nilearn.masking import apply_mask
    import numpy as np

    data = apply_mask(boldfile, maskfile)
    med = np.median(data.mean(axis=timeax))
    del data  # prevent memory overuse
    return med * median_factor


def run_melodic_wf(
    subject,
    bidsroot,
    fmriprepdir,
    space="func_preproc",
    approach: str = "runwise",
    fwhm: float = 4.0,
    hpf_sec: float = 120.0,
    derivname: str = "melodic",
    nprocs: int = 30,
    melodic_mem: float = 400.0,
    tr: float = 1.5,
    try_last_n_runs: bool or int = False,
):
    """
    Run Melodic on the preprocessed functional images.
    Besides fwhm and hpf for additional preprocessing, user can choose reference space (func_preproc or T1w) and
    approach (runwise or concat). Note that concat requires data to be in T1w space.

    Example:
        import sys
        subject = sys.argv[1]
        bidsroot = abspath(pjoin(pardir, pardir, pardir))
        fmriprepdir = pjoin(bidsroot, 'derivatives', 'fmriprep', f'sub-{subject}')
        run_melodic_wf(
            subject, bidsroot, fmriprepdir, approach='runwise', space='T1w',
        )
    """
    wf_bdir = pjoin(
        bidsroot, pardir, "melodicwf_wdir", f"space-{space}", f"sub-{subject}"
    )
    derivdir = pjoin(bidsroot, "derivatives", derivname, approach)
    for d in [wf_bdir, derivdir]:
        if not os.path.exists(d):
            os.makedirs(d)
    boldfiles, masks, outdirs = grab_data(
        subject, bidsroot, fmriprepdir, derivdir, space=space
    )
    if try_last_n_runs:
        boldfiles = boldfiles[-try_last_n_runs:]
        masks = masks[-try_last_n_runs:]
        outdirs = outdirs[-try_last_n_runs:]

    wf = Workflow(name="melodicwf", base_dir=wf_bdir)
    calcthresh = MapNode(
        Function(
            function=calc_susan_thresh,
            input_names=["boldfile", "maskfile"],
            output_names=["smooth_thresh"],
        ),
        name="calcthresh",
        iterfield=["boldfile", "maskfile"],
    )

    calcthresh.inputs.boldfile = boldfiles
    calcthresh.inputs.maskfile = masks
    susan = MapNode(
        SUSAN(fwhm=fwhm), iterfield=["in_file", "brightness_threshold"], name="susan"
    )
    susan.inputs.in_file = boldfiles
    tfilt = MapNode(
        TemporalFilter(highpass_sigma=float(hpf_sec / tr)),
        iterfield=["in_file"],
        name="tfilt",
    )
    calcthresh.inputs.maskfile = masks
    if approach == "runwise":
        melodic = MapNode(
            MELODIC(tr_sec=tr, out_all=True, no_bet=True, report=True),
            iterfield=["in_files", "mask", "out_dir"],
            name="melodic_runwise",
        )
        melodic.inputs.mask = masks
        melodic.inputs.out_dir = outdirs
    elif approach == "concat":
        melodic = Node(
            MELODIC(
                tr_sec=tr,
                out_all=True,
                no_bet=True,
                report=True,
                approach="concat",
                args="--debug",
            ),
            name="melodic_concat",
            mem_gb=melodic_mem,
            n_procs=nprocs,
        )
        outdir = abspath(pjoin(derivdir, f"space-{space}", f"sub-{subject}", "concat"))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        umask_img = intersect_masks(masks, threshold=0)
        umask_img.to_filename(pjoin(wf_bdir, "umask.nii.gz"))
        melodic.inputs.out_dir = outdir
        melodic.terminal_output = "stream"
        melodic.inputs.mask = pjoin(wf_bdir, "umask.nii.gz")
    else:
        raise ValueError(f'"approach" must be in ["runwise", "concat"]')
    wf.connect(
        [
            (calcthresh, susan, [("smooth_thresh", "brightness_threshold")]),
            (susan, tfilt, [("smoothed_file", "in_file")]),
            (tfilt, melodic, [("out_file", "in_files")]),
        ]
    )
    wf.run(plugin="MultiProc", plugin_args=dict(n_procs=nprocs))


class EvalMelodic:
    """
    Evaluate Melodic results in terms of ...
        - correlation with motion/physio/design parameters
        - High frequency content
        - 'Edge Fraction'
    Estimated features are saved to a tsv file and visual reports are generated for manual classification.
    Since plotting and saving visual reports for all ~24,000 components would take very long (estimate ~18 hours),
    one may specify to generate reports at random with a certain probability.

    Examples:
        # run with defaults
        evalmelodic = EvalMelodic()
        evalmelodic.runall()

        # Make a second set for rater number 2, that overlaps with 50% to the first set
        evalmelodic = EvalMelodic(
        exclude_comp_ids=get_set1_firsthalf_compids(),
        random_report_prob=.045,
        out_deriv_name='melodic_features/selection/selection2')
        evalmelodic.runall()

        # Make no reports, only save IC features
        evalmelodic = EvalMelodic(
        random_report_prob=.0,
        out_deriv_name='melodic_features/')
        evalmelodic.runall()
    """

    def __init__(
        self,
        bidsroot,
        out_deriv_name: str = "melodic_features",
        space: str = "T1w",
        edgefrac_thickness: int = 2,
        report_threshold: float = 0.9,
        try_first_n: int = 0,  # 0 means run for all
        report_dpi: int = 200,
        random_report_prob: float = 0.08,  # 1. means generate report always, .5 means only 50% of the time, etc.
        exclude_comp_ids: np.ndarray = None,
        # no new reports will be generated for these comp IDs (e.g. '2_things01_things_7_8)
        stc_reftime: float = 0.701625,
    ):
        self.bidsroot = bidsroot
        self.space = space
        self.out_deriv_name = out_deriv_name
        self.out_basedir = pjoin(bidsroot, "derivatives", out_deriv_name)
        if not os.path.exists(self.out_basedir):
            os.makedirs(self.out_basedir)
        spaces_naming = {"T1w": "_space-T1w", "func_preproc": ""}
        self.space_str = spaces_naming[space]
        self.edgefrac_thickness = edgefrac_thickness
        self.melodic_basedir = pjoin(
            self.bidsroot, "derivatives", "melodic", "runwise", f"space-{self.space}"
        )
        self.physioregs_basedir = pjoin(
            self.bidsroot, "derivatives", "physio_regressors"
        )
        self.ds = ThingsMRIdataset(self.bidsroot)
        self.bidsobs = self.ds.layout.get(
            suffix="bold", extension="nii.gz"
        )  # all runs as pybids objects
        # shuffle_list(self.bidsobs)
        self.try_first_n = try_first_n
        if self.try_first_n:
            self.bidsobs = self.bidsobs[: self.try_first_n]
        self.render3 = get_render3_cmap()
        self.lr = LinearRegression(fit_intercept=True, normalize=True, n_jobs=20)
        self.dpi = report_dpi
        self.random_report_prob = random_report_prob
        self.report_threshold = report_threshold
        self.exclude_comp_ids = exclude_comp_ids
        self.stc_reftime = stc_reftime

        warnings.filterwarnings("ignore", category=UserWarning)

    def get_designmat(self, bo, runinfo):
        events_tsv = bo.path.replace("_bold.nii.gz", "_events.tsv")
        events_df = pd.read_csv(events_tsv, sep="\t")[
            ["trial_type", "onset", "duration"]
        ]
        events_df["trial_type"] = "all"
        designmat = make_first_level_design_matrix(
            frame_times=np.arange(0, runinfo["ntrs"] * runinfo["tr"], runinfo["tr"])
            + self.stc_reftime,
            events=events_df,
            hrf_model="spm",
            drift_model=None,
            high_pass=None,
            drift_order=None,
        )["all"]
        tmp_mean = np.mean(designmat[25:270])
        tmp_sd = np.std(designmat[25:270])
        designmat_rescaled = (designmat.to_numpy() - tmp_mean) / tmp_sd
        return designmat, designmat_rescaled

    def get_physio(self, bo, runinfo):
        physio_tsv = pjoin(
            self.physioregs_basedir,
            f'sub-{runinfo["subject"]}',
            f"ses-{runinfo['session']}",
            "func",
            os.path.basename(bo.path).replace("bold.nii.gz", "physio_regressors.tsv"),
        )
        physio_df = pd.read_csv(physio_tsv, sep="\t")
        physio_df.index = (
            physio_df.index + 2
        )  # physio regs have two missing trs in the beginning
        return physio_df

    def get_motion(self, runinfo):
        confounds_tsv = pjoin(
            self.bidsroot,
            "derivatives",
            "fmriprep",
            f"sub-{runinfo['subject']}",
            f"ses-{runinfo['session']}",
            "func",
            f"sub-{runinfo['subject']}_ses-{runinfo['session']}_task-{runinfo['task']}_run-{runinfo['run']}_desc-confounds_timeseries.tsv",
        )
        confounds_df = pd.read_csv(confounds_tsv, sep="\t")
        return confounds_df[
            [c for c in confounds_df.columns if "trans" in c or "rot" in c]
        ]

    def get_comps(self, runinfo):
        melodic_outdir_run = pjoin(
            self.melodic_basedir,
            f"sub-{runinfo['subject']}",
            f"ses-{runinfo['session']}",
            f"sub-{runinfo['subject']}_ses-{runinfo['session']}_task-{runinfo['task']}_run-{runinfo['run']}_melodic",
        )
        mixmat = np.loadtxt(pjoin(melodic_outdir_run, "melodic_mix"))
        ica_nii_f = pjoin(melodic_outdir_run, "melodic_IC.nii.gz")
        comps_arr = load_img(ica_nii_f).get_fdata()
        return mixmat, comps_arr

    def get_edge_mask(self, runinfo):
        brainmask_f = pjoin(
            self.bidsroot,
            "derivatives",
            "fmriprep",
            f"sub-{runinfo['subject']}",
            f"ses-{runinfo['session']}",
            "func",
            f"sub-{runinfo['subject']}_ses-{runinfo['session']}_task-{runinfo['task']}_run-{runinfo['run']}{self.space_str}_desc-brain_mask.nii.gz",
        )
        csf_anat_f = pjoin(
            self.bidsroot,
            "derivatives",
            "fmriprep",
            f"sub-{runinfo['subject']}",
            "anat",
            f"sub-{runinfo['subject']}_acq-prescannormalized_rec-pydeface_label-CSF_probseg.nii.gz",
        )
        csf_func = threshold_img(
            resample_to_img(csf_anat_f, brainmask_f, interpolation="linear"),
            threshold=1.0,
        )
        brainmask = load_img(brainmask_f).get_fdata()
        mask_img = math_img("img1 - img2", img1=brainmask_f, img2=csf_func)
        mask_arr = mask_img.get_fdata()
        # worked okayish with erosion iterations=2
        ero_mask = binary_erosion(mask_arr, iterations=self.edgefrac_thickness).astype(
            int
        )
        edgemask = mask_arr - ero_mask
        return edgemask.astype(bool), brainmask.astype(bool)

    def calc_fits(self, comp_physio_df, maxphy_ind, comp_motion_df, maxmo_ind, comp_ts):
        """Fit physio and motion regressors to component timeseries. Returns in two predicted time series."""
        maxphy_ts = np.nan_to_num(
            comp_physio_df.to_numpy()[:, maxphy_ind + 1].reshape(-1, 1)
        )
        self.lr.fit(maxphy_ts, comp_ts)
        phy_fit = self.lr.predict(maxphy_ts)
        maxmo_ts = np.nan_to_num(
            comp_motion_df.to_numpy()[:, maxmo_ind + 1].reshape(-1, 1)
        )
        self.lr.fit(maxmo_ts, comp_ts)
        mo_fit = self.lr.predict(maxmo_ts)
        return phy_fit, mo_fit

    def calc_edgefrac(self, comp_arr, edgemask, brainmask):
        """Calculate the edge fraction, i.e. the tendency of the IC to occur at brain edges"""
        return (
            np.absolute(comp_arr[edgemask]).sum()
            / np.absolute(comp_arr[brainmask]).sum()
        )

    def generate_report(
        self,
        runinfo,
        results_dict,
        bo,
        comp_ts,
        comp_arr,
        designmat_rescaled,
        mo_fit,
        phy_fit,
        report_filename,
    ):
        """
        Create a Plot that summarizes the characteristics of a given IC:
        spatial map, edge fraction, frequency spectrum, high frequency content, fit to design/physio/motion
        """
        clim_ = 7
        fd = {"fontsize": 11}
        freqs, power = periodogram(comp_ts, fs=1.0 / runinfo["tr"])
        seconds = np.arange(0, runinfo["tr"] * runinfo["ntrs"], runinfo["tr"])
        comp_arr = np.rot90(np.copy(comp_arr), axes=(0, 2))
        comp_arr[
            np.logical_and(
                comp_arr < self.report_threshold, comp_arr > -self.report_threshold
            )
        ] = np.nan
        func_f = pjoin(
            self.bidsroot,
            "derivatives",
            "fmriprep",
            f'sub-{runinfo["subject"]}',
            f"ses-{runinfo['session']}",
            "func",
            f"sub-{runinfo['subject']}_ses-{runinfo['session']}_task-{runinfo['task']}_run-{runinfo['run']}{self.space_str}_desc-preproc_bold.nii.gz",
        )
        meanbold = np.rot90(load_img(func_f).get_fdata().mean(axis=-1), axes=(0, 2))
        hor_is = np.linspace(
            5, comp_arr.shape[-1] - 15, num=10, dtype=int, endpoint=False
        )
        hor_img = np.flip(
            np.concatenate([comp_arr[slice_i, :, :] for slice_i in hor_is], axis=-1)
        )
        hor_bg = np.flip(
            np.concatenate([meanbold[slice_i, :, :] for slice_i in hor_is], axis=-1)
        )
        cor_is = np.linspace(5, comp_arr.shape[1], num=5, dtype=int, endpoint=False)
        cor_img = np.concatenate(
            [comp_arr[:, slice_i, :] for slice_i in cor_is], axis=-1
        )
        cor_bg = np.concatenate(
            [meanbold[:, slice_i, :] for slice_i in cor_is], axis=-1
        )
        sag_is = np.linspace(10, comp_arr.shape[2], num=5, dtype=int, endpoint=False)
        sag_img = np.concatenate(
            [comp_arr[:, :, slice_i] for slice_i in sag_is], axis=-1
        )
        sag_bg = np.concatenate(
            [meanbold[:, :, slice_i] for slice_i in sag_is], axis=-1
        )

        fig = plt.figure(
            figsize=(16.54, 9.45), facecolor="white"
        )  # constrained_layout=True
        gs = GridSpec(4, 2, figure=fig, hspace=0.2, wspace=0.05)
        # component maps from different views
        hor_ax = fig.add_subplot(gs[0, :])
        hor_ax.imshow(hor_bg, cmap="gray")
        hor_ax.imshow(hor_img, cmap=self.render3, clim=(-clim_, clim_))
        hor_ax.set_title(
            f"Edge Fraction = {results_dict['edgefrac']:.2f}",
            x=0.07,
            y=0.84,
            fontdict=fd,
            color="white",
        )
        cor_ax = fig.add_subplot(gs[1, 0])
        cor_ax.imshow(cor_bg, cmap="gray")
        cor_ax.imshow(cor_img, cmap=self.render3, clim=(-clim_, clim_))
        sag_ax = fig.add_subplot(gs[1, 1])
        sag_ax.imshow(sag_bg, cmap="gray")
        sag_ax.imshow(sag_img, cmap=self.render3, clim=(-clim_, clim_))
        # frequency spectrum
        freq_ax = fig.add_subplot(gs[2, 0])
        freq_ax.plot(freqs, power, color="black", alpha=0.7)
        freq_ax.set_title(
            f'Frequency (HFC = {results_dict["hfc"]:.3f})', y=0.85, fontdict=fd
        )
        # design, physio and motion fit
        design_ax = fig.add_subplot(gs[2, 1])
        design_ax.plot(seconds, comp_ts, color="black", alpha=0.7)
        design_ax.plot(
            seconds, designmat_rescaled, alpha=0.7, color=plt.cm.tab10.colors[2]
        )
        design_ax.set_ylim([-3, None])
        design_ax.set_title(
            f'Design (r = {results_dict["designcorr"]:.3f})',
            color=plt.cm.tab10.colors[2],
            y=0.85,
            fontdict=fd,
        )
        phy_ax = fig.add_subplot(gs[3, 0])
        phy_ax.plot(seconds, zscore(comp_ts), color="black", alpha=0.8)
        phy_ax.plot(seconds, zscore(phy_fit), alpha=0.8, color=plt.cm.tab10.colors[0])
        phy_ax.set_title(
            f'Physio (r = {results_dict["maxphysiocorr"]:.3f})',
            color=plt.cm.tab10.colors[0],
            y=0.85,
            fontdict=fd,
        )
        mo_ax = fig.add_subplot(gs[3, 1])
        mo_ax.plot(seconds, zscore(comp_ts), color="black", alpha=0.8)
        mo_ax.plot(seconds, zscore(mo_fit), alpha=0.8, color=plt.cm.tab10.colors[1])
        mo_ax.set_title(
            f'Motion (r = {results_dict["maxmotioncorr"]:.3f})',
            color=plt.cm.tab10.colors[1],
            y=0.85,
            fontdict=fd,
        )
        for imax in [hor_ax, cor_ax, sag_ax]:
            imax.set(xticks=[], yticks=[])
        plt.suptitle(
            f"{bo.filename} (Component #{results_dict['comp_i']})".replace(
                "_bold.nii.gz", ""
            ),
            y=0.92,
        )
        # plt.show()
        fig.savefig(report_filename, dpi=self.dpi)
        plt.close(fig=fig)

    def runall(self):
        results_dicts = []
        report_dicts = (
            []
        )  # make an extra csv file containing only ICs for which we've generated reports
        for bo in tqdm(self.bidsobs, desc="iterating over runs"):
            runinfo = bo.get_entities()
            # TODO: missing event files for prf task and some memory runs (skipped for now)
            if (runinfo["task"] == "pRF") or (
                runinfo["task"] == "memory" and runinfo["run"] > 10
            ):
                continue
            runinfo["tr"] = bo.get_metadata()["RepetitionTime"]
            run_img = load_img(bo.path)
            runinfo["ntrs"] = run_img.shape[-1]
            designmat, designmat_rescaled = self.get_designmat(bo, runinfo)
            physio_df = self.get_physio(bo, runinfo)
            motion_df = self.get_motion(runinfo)
            mixmat, comps_arr = self.get_comps(runinfo)
            edgemask, brainmask = self.get_edge_mask(runinfo)
            outdir = pjoin(
                self.out_basedir,
                f'sub-{runinfo["subject"]}',
                f'ses-{runinfo["session"]}',
            )
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for comp_i in range(mixmat.shape[-1]):
                results_dict = {
                    "subject": runinfo["subject"],
                    "session": runinfo["session"],
                    "task": runinfo["task"],
                    "run": runinfo["run"],
                    "comp_i": comp_i,
                }
                comp_ts = mixmat[:, comp_i]
                comp_arr = comps_arr[:, :, :, comp_i]
                compdf = pd.DataFrame({"comp_ts": comp_ts})
                # design correlation
                results_dict["designcorr"] = np.absolute(
                    np.corrcoef(comp_ts, designmat)[0, -1]
                )
                # motion correlation
                comp_motion_df = pd.concat([compdf, motion_df], axis=1)
                motioncorrs = comp_motion_df.corr().to_numpy()[1:, 0]
                maxmo_ind = np.argmax(np.absolute(motioncorrs))
                results_dict["maxmotioncorr"] = motioncorrs[maxmo_ind]
                # physio correlation
                comp_physio_df = pd.concat([compdf, physio_df], axis=1)
                physiocorrs = comp_physio_df.corr().to_numpy()[1:, 0]
                maxphy_ind = np.argmax(np.absolute(physiocorrs))
                results_dict["maxphysiocorr"] = physiocorrs[maxphy_ind]
                # edge fraction
                results_dict["edgefrac"] = self.calc_edgefrac(
                    comp_arr, edgemask, brainmask
                )
                # high frequency content
                results_dict["hfc"] = calc_hfc(comp_ts)
                results_dicts.append(results_dict)
                if self.exclude_comp_ids:
                    # skip if in self.exclude_comp_ids
                    comp_id = f"{int(runinfo['subject'])}_{runinfo['session']}_{runinfo['task']}_{int(runinfo['run'])}_{comp_i}"
                    if comp_id in self.exclude_comp_ids:
                        continue
                # Decide whether to generate report, given probability
                if not rndchoice(
                    2, p=[1.0 - self.random_report_prob, self.random_report_prob]
                ):
                    continue
                report_filename = pjoin(
                    outdir,
                    f"{bo.filename}_comp-{comp_i}.png".replace("_bold.nii.gz", ""),
                )
                phy_fit, mo_fit = self.calc_fits(
                    comp_physio_df, maxphy_ind, comp_motion_df, maxmo_ind, comp_ts
                )
                self.generate_report(
                    runinfo,
                    results_dict,
                    bo,
                    comp_ts,
                    comp_arr,
                    designmat_rescaled,
                    mo_fit,
                    phy_fit,
                    report_filename,
                )
                report_dicts.append(
                    {
                        "subject": runinfo["subject"],
                        "session": runinfo["session"],
                        "task": runinfo["task"],
                        "run": runinfo["run"],
                        "comp_i": comp_i,
                        "ManualRating": "",
                    }
                )
        results_df = pd.DataFrame(results_dicts)
        results_df.to_csv(
            pjoin(self.out_basedir, f"melodic_correlations_space-{self.space}.tsv"),
            sep="\t",
        )
        reports_df = pd.DataFrame(report_dicts)
        reports_df.to_csv(
            pjoin(
                self.out_basedir, f"melodic_correlations_space-{self.space}_reports.tsv"
            ),
            sep="\t",
        )


def get_set1_firsthalf_compids(bidsroot, nexclude=804):
    """
    Returns a flat array of component ids (e.g. '2_things01_things_7_8') which represent the first half of selection1.
    These can be used in EvalMelodic in order to make a new set of reports, which does not overlap.
    This way, selection 1 and 2 will overlap to 50 percent.
    """
    exclude_tsv = pjoin(
        bidsroot,
        "derivatives",
        "melodic_features",
        "selection",
        "selection1",
        "melodic_correlations_space-T1w_reports.tsv",
    )
    idcols = ["subject", "session", "task", "run", "comp_i"]
    exclude_df = pd.read_csv(exclude_tsv, sep="\t")[:nexclude]
    for intcol in ["subject", "run", "comp_i"]:
        exclude_df[intcol] = exclude_df[intcol].astype(str)
    exclude_df["comp_id"] = exclude_df[idcols].agg("_".join, axis=1)
    return exclude_df["comp_id"].to_numpy()


def write_noise_tsvs(
    outdir,
    melodic_features_tsv,
    bidsroot,
    thresholds=dict(edgefrac=0.225, hfc=0.4),
    combine_thresholds="any",
):
    # load component features
    feature_df = pd.read_csv(melodic_features_tsv, sep="\t")
    # mask by thresholds
    masks = np.vstack(
        [
            np.array(feature_df[feature] > threshold)
            for feature, threshold in thresholds.items()
        ]
    ).T
    if combine_thresholds == "any":
        combined_mask = np.any(masks, axis=1)
    elif combine_thresholds == "all":
        combined_mask = np.all(masks, axis=1)
    else:
        raise ValueError('combine_thresholds must be "all" or "any"')
    feature_df["noise"] = 0
    feature_df.loc[combined_mask, "noise"] = 1
    # only take rows for noise components
    noise_df = feature_df.loc[feature_df["noise"] == 1,]
    # Get time series for each noise component and make out file names

    def _get_comp_ts(row, bidsroot=bidsroot):
        melodic_basedir = pjoin(
            bidsroot,
            "derivatives",
            "melodic",
            "runwise",
            "space-T1w",
            f"sub-{row['subject']:02d}",
        )
        melodic_outdir = pjoin(
            melodic_basedir,
            f"ses-{row['session']}",
            f"sub-{row['subject']:02d}_ses-{row['session']}_task-{row['task']}_run-{row['run']}_melodic",
        )
        mixmat = np.loadtxt(pjoin(melodic_outdir, "melodic_mix"))
        return mixmat[:, row["comp_i"]]

    def _get_out_txt(row, outdir=outdir):
        out_txt_basename = f"sub-{row['subject']:02d}/ses-{row['session']}/sub-{row['subject']:02d}_ses-{row['session']}_task-{row['task']}_run-{row['run']:02d}.txt"
        return pjoin(outdir, out_txt_basename)

    timeseries = noise_df.apply(_get_comp_ts, axis=1)
    noise_df["comp_ts"] = timeseries
    out_txts = noise_df.apply(_get_out_txt, axis=1)
    noise_df["out_txt"] = out_txts
    # save to file
    for out_txt in tqdm(noise_df.out_txt.unique(), desc="saving to file"):
        outdir = pjoin(*out_txt.split("/")[:-1])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        run_df = noise_df[noise_df["out_txt"] == out_txt]
        print(f"found {len(run_df)} noise components for this run")
        noise_arr = np.vstack(run_df.comp_ts.to_list()).T
        np.savetxt(out_txt, noise_arr)
    return None
