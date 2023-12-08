"""
Code for localizing category selective regions of interests. The nipype implementation can exploit running on multiple CPU,
the number of which can be specified with <n_processors>

Usage:
python localizerGLM_FSL.py <subject_ID> <bids_path> <n_processors>

Example:
python betas.py 01 /home/user/thingsmri 8
"""

import os
from os.path import join as pjoin
from os.path import pardir

import pandas as pd
from niflow.nipype1.workflows.fmri.fsl import (
    create_modelfit_workflow,
    create_fixed_effects_flow,
)
from nilearn.image import resample_img, index_img
from nilearn.masking import intersect_masks
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.base import Bunch
from nipype.interfaces.fsl.model import SmoothEstimate, Cluster
from nipype.interfaces.fsl.preprocess import SUSAN
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Node, Workflow, MapNode

from thingsmri.dataset import ThingsMRIdataset


def grabdata(subject: str, bidsroot: str, sesname: str, nruns: int = 6):
    print("grabbing data")
    thingsmri = ThingsMRIdataset(bidsroot)
    bold_files = [
        pjoin(
            bidsroot,
            "derivatives",
            "fmriprep",
            f"sub-{subject}",
            f"ses-{sesname}",
            "func",
            f"sub-{subject}_ses-{sesname}_task-6cat_run-{run_i + 1}_space-T1w_desc-preproc_bold.nii.gz",
        )
        for run_i in range(nruns)
    ]
    nuisance_tsvs = [
        pjoin(
            bidsroot,
            "derivatives",
            "fmriprep",
            f"sub-{subject}",
            f"ses-{sesname}",
            "func",
            f"sub-{subject}_ses-{sesname}_task-6cat_run-{run_i + 1}_desc-confounds_timeseries.tsv",
        )
        for run_i in range(nruns)
    ]
    masks = [
        pjoin(
            bidsroot,
            "derivatives",
            "fmriprep",
            f"sub-{subject}",
            f"ses-{sesname}",
            "func",
            f"sub-{subject}_ses-{sesname}_task-6cat_run-{run_i + 1}_space-T1w_desc-brain_mask.nii.gz",
        )
        for run_i in range(nruns)
    ]
    anat_mask = pjoin(
        bidsroot,
        "derivatives",
        "fmriprep",
        f"sub-{subject}",
        "anat",
        f"sub-{subject}_acq-prescannormalized_rec-pydeface_desc-brain_mask.nii.gz",
    )
    events_tsvs = thingsmri.layout.get(
        subject=subject,
        task="6cat",
        extension="tsv",
        session=sesname,
        return_type="filename",
    )
    return bold_files, nuisance_tsvs, masks, events_tsvs, anat_mask


def cat_contrasts():
    """return list of category selective contrasts for use in Level1Design"""
    condnames = ["bodyparts", "faces", "objects", "scenes", "words", "scrambled"]
    contrasts = [
        ["all", "T", condnames, [1] * len(condnames)],
        ["FFA_scr", "T", condnames, [0, 1, 0, 0, 0, -1]],
        ["FFA_obj", "T", condnames, [0, 2, -1, 0, 0, -1]],
        ["FFA_obj2", "T", condnames, [0, 1, -1, 0, 0, 0]],
        ["FFA_alt", "T", condnames, [0, 3, -1, -1, 0, -1]],
        ["FFA_alt2", "T", condnames, [0, 2, -1, -1, 0, 0]],
        ["FFA_all", "T", condnames, [-1, 5, -1, -1, -1, -1]],
        ["PPA_scr", "T", condnames, [0, 0, 0, 1, 0, -1]],
        ["PPA_alt", "T", condnames, [0, -1, -1, 3, 0, -1]],
        ["PPA_alt2", "T", condnames, [0, -1, -1, 2, 0, 0]],
        ["PPA_obj", "T", condnames, [0, 0, -1, 2, 0, -1]],
        ["PPA_obj2", "T", condnames, [0, 0, -1, 1, 0, 0]],
        ["PPA_all", "T", condnames, [-1, -1, -1, 5, -1, -1]],
        ["EBA_scr", "T", condnames, [1, 0, 0, 0, 0, -1]],
        ["EBA_all", "T", condnames, [5, -1, -1, -1, -1, -1]],
        ["EBA_obj", "T", condnames, [1, 0, -1, 0, 0, 0]],
        ["EBA_obj2", "T", condnames, [2, 0, -1, 0, 0, -1]],
        ["LOC", "T", condnames, [1, 1, 1, 1, 1, -5]],
        ["LOC_alt", "T", condnames, [0, 0, 1, 0, 0, -1]],
        ["VIS", "T", condnames, [-1, -1, -1, -1, -1, 5]],
        ["VIS_alt", "T", condnames, [0, 0, -1, 0, 0, 1]],
        ["VWF_all", "T", condnames, [-1, -1, -1, -1, 5, -1]],
        ["VWF_scr", "T", condnames, [0, 0, 0, 0, 1, -1]],
        ["VWF_obj", "T", condnames, [0, 0, -1, 0, 1, 0]],
        ["VWF_obj2", "T", condnames, [0, 0, -1, 0, 2, -1]],
    ]
    contrast_names = [con[0] for con in contrasts]
    return contrasts, contrast_names


def make_runinfo(
    events_tsv: str,
    nuisance_tsv: str,
    noiseregs: list,
    stc_reftime: float,
) -> Bunch:
    """Create subjectinfo (bunch) for one run."""
    events_df = pd.read_csv(events_tsv, sep="\t")
    events_df["onset"] = (
        events_df["onset"] - stc_reftime
    )  # take slice timing reference into account
    # get conditions, onsets, durations as lists
    conditions = []
    onsets = []
    durations = []
    for group in events_df.groupby("trial_type"):
        conditions.append(group[0])
        onsets.append(group[1].onset.tolist())
        durations.append(group[1].duration.tolist())
    # add noise regressors if there are any
    if not noiseregs:
        return Bunch(conditions=conditions, onsets=onsets, durations=durations)
    nuisance_df = pd.read_csv(nuisance_tsv, sep="\t")
    nuisance_df = nuisance_df[noiseregs]
    if "framewise_displacement" in noiseregs:
        nuisance_df["framewise_displacement"] = nuisance_df[
            "framewise_displacement"
        ].fillna(0)
    noiseregs_names = nuisance_df.columns.tolist()
    noiseregs_regressors = [
        nuisance_df[noisereg].tolist() for noisereg in noiseregs_names
    ]
    return Bunch(
        conditions=conditions,
        onsets=onsets,
        durations=durations,
        regressor_names=noiseregs_names,
        regressors=noiseregs_regressors,
    )


def sort_copes(files):
    """
    reshape copes (or varcopes) returned by create_fixed_effects_flow()
    for use in create_fixed_effects_flow()
    """
    numelements = len(files[0])
    outfiles = []
    for i in range(numelements):
        outfiles.insert(i, [])
        for j, elements in enumerate(files):
            outfiles[i].append(elements[i])
    return outfiles


def resample_to_file(in_file: str, target_file: str, wdir: str):
    """Resample in_file to the resolution of target_file"""
    target_img = index_img(target_file, 0)
    resampled = resample_img(
        in_file, target_affine=target_img.affine, target_shape=target_img.shape
    )
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    outfile = pjoin(wdir, "resampled_file.nii.gz")
    resampled.to_filename(outfile)
    return outfile


def union_masks_to_filename(masks: list, wdir: str):
    """Create union of list of brain masks, save to file in working directory and return the file path"""
    union_img = intersect_masks(masks, threshold=0)
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    outfile = pjoin(wdir, "unionmask.nii.gz")
    union_img.to_filename(outfile)
    return outfile


def make_localizerGLM_wf(
    subject: str,
    bidsroot: str,
    whichsession: dict = {
        "01": "localizer2",
        "02": "localizer1",
        "03": "localizer1",
    },
    hrf: dict = {"dgamma": {"derivs": False}},
    cluster_thr: float = 3.7,
    cluster_pthr: float = 0.0001,
    hpf: int = 60,
    fwhm: int = 5,
    ar: bool = False,
    tr: float = 1.5,
    stc_reftime: float = 0.701625,
    smoothing_brightness_threshold: float = 2000,
    noiseregs: list = [],  # ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement'],
):
    wdir = pjoin(bidsroot, pardir, "localizerGLM_wdir", f"sub-{subject}")
    contrasts, contrast_names = cat_contrasts()
    locses = whichsession[subject]
    wf = Workflow(name="wf", base_dir=wdir)
    bold_files, nuisance_tsvs, masks, events_tsvs, anat_mask = grabdata(
        subject, bidsroot, locses
    )
    union_mask = union_masks_to_filename(masks, pjoin(wdir, "unionmask"))
    smooth = MapNode(SUSAN(), name="smooth", iterfield=["in_file"])
    smooth.inputs.in_file = bold_files
    smooth.inputs.fwhm = fwhm
    smooth.inputs.brightness_threshold = smoothing_brightness_threshold
    runinfos = [
        make_runinfo(
            events_tsv, nuisance_tsv, noiseregs=noiseregs, stc_reftime=stc_reftime
        )
        for events_tsv, nuisance_tsv in zip(events_tsvs, nuisance_tsvs)
    ]
    modelspec = Node(
        SpecifyModel(
            subject_info=runinfos,
            high_pass_filter_cutoff=hpf,
            input_units="secs",
            time_repetition=tr,
        ),
        name="modelspec",
    )
    modelfit = create_modelfit_workflow()
    modelfit.inputs.inputspec.interscan_interval = tr
    modelfit.inputs.inputspec.contrasts = contrasts
    modelfit.inputs.inputspec.bases = hrf
    modelfit.inputs.inputspec.model_serial_correlations = ar
    if not ar:
        filmgls = modelfit.get_node("modelestimate")
        filmgls.inputs.autocorr_noestimate = True
    ffx = create_fixed_effects_flow()
    l2model = ffx.get_node("l2model")
    l2model.inputs.num_copes = len(bold_files)
    flameo = ffx.get_node("flameo")
    flameo.inputs.mask_file = union_mask
    sortcopes = Node(
        Function(function=sort_copes, input_names=["files"], output_names=["outfiles"]),
        name="sortcopes",
    )
    sortvarcopes = Node(
        Function(function=sort_copes, input_names=["files"], output_names=["outfiles"]),
        name="sortvarcopes",
    )
    smoothest = MapNode(
        SmoothEstimate(mask_file=union_mask), name="smoothest", iterfield=["zstat_file"]
    )
    cluster = MapNode(
        Cluster(
            threshold=cluster_thr,
            pthreshold=cluster_pthr,
            out_threshold_file=True,
            out_pval_file=True,
            out_index_file=True,
            out_localmax_txt_file=True,
            out_localmax_vol_file=True,
            out_max_file=True,
            out_mean_file=True,
            out_size_file=True,
        ),
        name="cluster",
        iterfield=["in_file", "dlh", "volume"],
    )
    sink = Node(
        DataSink(
            infields=["contasts.@con"],
        ),
        name="sink",
    )
    sink.inputs.base_directory = pjoin(
        bidsroot, "derivatives", "localizer", f"sub-{subject}"
    )
    sink.inputs.substitutions = [
        (f"/{outputtype}/_cluster{i}/", f"/contrast-{contname}/")
        for outputtype in [
            "threshold_file",
            "pval_file",
            "index_file",
            "localmax_txt_file",
            "localmax_vol_file",
            "max_file",
            "mean_file",
            "size_file",
        ]
        for i, contname in enumerate(contrast_names)
    ]
    wf.connect(
        [
            (smooth, modelspec, [("smoothed_file", "functional_runs")]),
            (smooth, modelfit, [("smoothed_file", "inputspec.functional_data")]),
            (modelspec, modelfit, [("session_info", "inputspec.session_info")]),
            (modelfit, sortcopes, [("outputspec.copes", "files")]),
            (modelfit, sortvarcopes, [("outputspec.varcopes", "files")]),
            (sortcopes, ffx, [("outfiles", "inputspec.copes")]),
            (sortvarcopes, ffx, [("outfiles", "inputspec.varcopes")]),
            (modelfit, ffx, [("outputspec.dof_file", "inputspec.dof_files")]),
            (ffx, smoothest, [("outputspec.zstats", "zstat_file")]),
            (smoothest, cluster, [("dlh", "dlh"), ("volume", "volume")]),
            (ffx, cluster, [("outputspec.zstats", "in_file")]),
            (
                cluster,
                sink,
                [
                    ("threshold_file", "threshold_file"),
                    ("pval_file", "pval_file"),
                    ("index_file", "index_file"),
                    ("localmax_txt_file", "localmax_txt_file"),
                    ("localmax_vol_file", "localmax_vol_file"),
                    ("max_file", "max_file"),
                    ("mean_file", "mean_file"),
                    ("size_file", "size_file"),
                ],
            ),
        ]
    )
    return wf


if __name__ == "__main__":
    import sys

    subject, bidsroot, nprocs = sys.argv[1], sys.argv[2], sys.argv[3]
    wf = make_localizerGLM_wf(subject, bidsroot)
    wf.run(plugin="MultiProc", plugin_args=dict(n_procs=nprocs))
