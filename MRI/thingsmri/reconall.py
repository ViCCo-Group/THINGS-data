"""
Run Freesurfer's recon-all on the THINGS-fMRI dataset. Mostly intended for documentation.
"""

import stat
from distutils.dir_util import copy_tree
from os import pardir, chmod
from os.path import join as pjoin

from nipype.interfaces.freesurfer.preprocess import ReconAll
from nipype.interfaces.utility import Function
from nipype.pipeline.engine import Node, Workflow


def grabanat(bidsroot, subject):
    """Grab anatomical data for given subject"""
    import sys
    from os.path import join as pjoin
    from dataset import ThingsMRIdataset

    sys.path.insert(0, pjoin(bidsroot, "code", "things", "mri"))
    thingsmri = ThingsMRIdataset(bidsroot)
    t1files = thingsmri.layout.get(
        subject=subject,
        return_type="file",
        extension=".nii.gz",
        suffix="T1w",
        reconstruction="pydeface",
        acquisition="prescannormalized",
    )
    t2file = thingsmri.layout.get(
        subject=subject,
        return_type="file",
        extension=".nii.gz",
        suffix="T2w",
        reconstruction="pydeface",
        acquisition="prescannormalized",
    )[0]
    return t1files, t2file


def make_reconall_wf(subject, wdir, bidsroot, directive="all", nprocs=12) -> Workflow:
    """return very simple workflow running reconall for one subject"""
    wf = Workflow(name="reconall_wf", base_dir=wdir)
    datagrabber = Node(
        Function(
            function=grabanat,
            input_names=["bidsroot", "subject"],
            output_names=["t1files", "t2file"],
        ),
        name="datagrabber",
    )
    datagrabber.inputs.subject = subject
    datagrabber.inputs.bidsroot = bidsroot
    reconall = Node(
        ReconAll(use_T2=True, directive=directive, openmp=nprocs), name="reconall"
    )
    wf.connect(
        [(datagrabber, reconall, [("t1files", "T1_files"), ("t2file", "T2_file")])]
    )
    return wf


def main(
    subject,
    nprocs,
    bidsroot,  # path within container "preproc"
    free_permissions=True,  # free permissions for workdir and output (bc docker has different user than host)
) -> None:
    """Run Reconall for one subject and copy the output to the bids derivatives"""
    wdir = pjoin(bidsroot, pardir, "reconall_workdir", f"sub-{subject}")
    derivdir = pjoin(bidsroot, "derivatives", "reconall", f"sub-{subject}")
    # make and run workflow
    wf = make_reconall_wf(subject=subject, bidsroot=bidsroot, wdir=wdir, nprocs=nprocs)
    wf.write_graph(graph2use="colored", simple_form=True)
    wf.run()
    # copy output to derivatives manually to preserve typical reconall output structure
    ra_nodedir = pjoin(wdir, "reconall_wf", "reconall", "recon_all")
    copy_tree(ra_nodedir, derivdir, preserve_mode=False)
    if free_permissions:
        for d in [wdir, derivdir]:
            chmod(d, stat.S_IRWXO)
    return None
