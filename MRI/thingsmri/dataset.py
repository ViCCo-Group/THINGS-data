#! /usr/env/python

import warnings
from os.path import exists as pexists
from os.path import join as pjoin

from bids import BIDSLayout


class ThingsMRIdataset:
    """
    Data loader for the THINGS-fMRI dataset.
    """

    def __init__(self, root_path: str, validate: bool = True):
        # path attributes
        assert isinstance(root_path, str)
        self.root_path = root_path
        self.rawdata_path = pjoin(self.root_path, "rawdata")
        self.sourcedata_path = pjoin(self.root_path, "sourcedata")
        self.derivs_path = pjoin(self.root_path, "derivatives")
        # pybids layout
        self.layout = BIDSLayout(self.rawdata_path, validate=validate)
        self.subjects = self.layout.get(return_type="id", target="subject")
        self.sessions = self.layout.get(return_type="id", target="session")
        self.things_sessions = [
            ses
            for ses in self.layout.get(return_type="id", target="session")
            if "things" in ses
        ]
        runids = self.layout.get(return_type="id", target="run")
        self.maxnruns = int(runids[-1])  # maximum number of runs per session

    def update_layout(self, validate: bool = True):
        self.layout = BIDSLayout(self.rawdata_path, validate=validate)
        self.layout.add_derivatives(self.derivs_path)
        return None

    def include_derivs(self):
        """Note that this only captures folders in the derivs_path which have a dataset_description.json."""
        if pexists(self.derivs_path):
            self.layout.add_derivatives(self.derivs_path)
        else:
            warnings.warn(
                "Could not find derivatives directory\n{}".format(self.derivs_path)
            )
        return None

    def get_reconall_anat(self, subject: str) -> dict:
        """Collect paths to relevant outputs of reconall"""
        return dict(
            wmseg=pjoin(
                self.derivs_path, "reconall", f"sub-{subject}", "mri", "wm.seg.mgz"
            ),
            t1=pjoin(self.derivs_path, "reconall", f"sub-{subject}", "mri", "nu.mgz"),
            t1_brain=pjoin(
                self.derivs_path, "reconall", f"sub-{subject}", "mri", "norm.mgz"
            ),
        )

    def get_fieldmap_files(self, subject: str) -> dict:
        phasediff_files = self.layout.get(
            subject=subject, return_type="file", extension=".nii.gz", suffix="phasediff"
        )
        mag1_files = self.layout.get(
            subject=subject,
            return_type="file",
            extension=".nii.gz",
            suffix="magnitude1",
        )
        mag2_files = self.layout.get(
            subject=subject,
            return_type="file",
            extension=".nii.gz",
            suffix="magnitude2",
        )
        return dict(
            phasediff_files=phasediff_files,
            mag1_files=mag1_files,
            mag2_files=mag2_files,
        )

    def get_fmriprep_t1w(self, subject):
        return pjoin(
            self.root_path,
            "derivatives",
            "fmriprep",
            f"sub-{subject}",
            "anat",
            f"sub-{subject}_acq-prescannormalized_rec-pydeface_desc-preproc_T1w.nii.gz",
        )
