"""
Estimate single-trial responses. Output will be saved in the "derivatives" subfolder of the bids dataset.
NOTE: This analysis is resource-intensive in terms of computer memory and cpu usage. It should really only be run on a computing cluster and not on home computers or laptops.

Usage:
python betas.py <subject_ID> <bids_path>

Example:
python betas.py 01 /home/user/thingsmri
"""

import glob
import os
import random
import sys
import time
from itertools import combinations
from os.path import join as pjoin
from shutil import copyfile

import numpy as np
import pandas as pd
from fracridge import FracRidgeRegressor
from joblib import Parallel, delayed
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import load_img
from nilearn.masking import unmask, apply_mask
from numpy.random import normal
from scipy.linalg import block_diag
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from thingsmri.glm import THINGSGLM, df_to_boxcar_design, get_nuisance_df
from thingsmri.utils import (
    pearsonr_nd,
    get_hrflib,
    spearman_brown,
    match_scale,
    apply_mask_smoothed,
    regress_out,
    r2_ndarr,
)


class SingleTrialBetas(THINGSGLM):
    """
    Calculate single trial response estimates for the THINGS-fMRI dataset.
    """

    def __init__(
        self,
        bidsroot: str,
        subject: str,
        out_deriv_name: str = "betas_vol",
        usecache: bool = True,
        tuning_procedure: str = "stepwise",
        standardize_noiseregs: bool = True,
        standardize_trialegs: bool = True,
        on_residuals_data: bool = True,
        on_residuals_design: bool = True,
        cv_scheme: str = "loo",
        perf_metric: str = "l2",
        assume_hrf: int or False = False,
        match_scale_runwise: bool = False,
        use_spearman_brown: bool = False,
        unregularized_targets: bool = True,
        hrflib_url: str = "https://raw.githubusercontent.com/kendrickkay/GLMdenoise/master/utilities"
        "/getcanonicalhrflibrary.tsv",
        rescale_hrflib_amplitude: bool = True,
        hrflib_resolution: float = 0.1,
        overfit_hrf_model: str = "onoff",
        fracs: np.ndarray = np.hstack(
            [np.arange(0.1, 0.91, 0.05), np.arange(0.91, 1.01, 0.01)]
        ),
        fmriprep_noiseregs: list = [],
        fmriprep_compcors: bool or int = 0,
        aroma_regressors: bool = False,
        manual_ica_regressors: bool = True,
        drift_model: str = "polynomial",
        poly_order: int = 4,
        high_pass: float = None,
        rescale_runwise_data: str = "z",
        zscore_data_sessionwise: bool = False,
        stims_per_ses: int = 920,
        out_dtype=np.double,
        n_parallel_hrf: int = 50,
        n_parallel_repeated_betas: int = 6,
        n_parallel_splithalf: int = 15,
        mcnc_nsig: int = 1000,
        mcnc_nmes: int = 1,
        mcnc_njobs: int = 50,
        mcnc_ddof: int = 0,
    ):
        super().__init__(
            sigscale_nilearn=False,
            noise_model="ols",
            hrf_model=None,
            bidsroot=bidsroot,
            subject=subject,
            out_deriv_name=out_deriv_name,
            noiseregs=fmriprep_noiseregs,
            acompcors=fmriprep_compcors,
            include_all_aroma=aroma_regressors,
            high_pass=high_pass,
            drift_model=drift_model,
            drift_order=poly_order,
        )
        self.manual_ica_regressors = manual_ica_regressors
        self.usecache = usecache
        self.on_residuals_data = on_residuals_data
        self.on_residuals_design = on_residuals_design
        self.zscore_data_sessionwise = zscore_data_sessionwise
        assert tuning_procedure in ["stepwise", "combined"]
        self.tuning_procedure = tuning_procedure
        assert cv_scheme in ["splithalf", "mcnc", "loo", "unregularized"]
        self.cv_scheme = cv_scheme
        assert perf_metric in ["correlation", "l1", "l2"]
        self.perf_metric = perf_metric
        self.match_scale_runwise = match_scale_runwise
        self.use_spearman_brown = use_spearman_brown
        self.standardize_noiseregs = standardize_noiseregs
        self.standardize_trialegs = standardize_trialegs
        self.n_parallel_repeated_betas = n_parallel_repeated_betas
        self.n_parallel_splithalf = n_parallel_splithalf
        self.kf = KFold(n_splits=self.n_sessions)
        self.stims_per_ses = stims_per_ses
        self.hrflib_url = hrflib_url
        self.hrflib = get_hrflib(self.hrflib_url)
        self.nsamples_hrf, self.nhrfs = self.hrflib.shape
        assert fracs[-1] == 1.0  # make sure we include OLS
        self.fracs = fracs
        self.unregularized_targets = unregularized_targets
        self.nfracs = len(self.fracs)
        assert rescale_runwise_data in [
            "z",
            "psc",
            "center",
        ]  # don't allow uncentered data
        self.rescale_runwise_data = rescale_runwise_data
        assert overfit_hrf_model in ["onoff", "single-trial"]
        self.overfit_hrf_model = overfit_hrf_model
        self.assume_hrf = assume_hrf  # picked 10 as canonical hrf
        self.n_parallel_hrf = n_parallel_hrf
        self.hrflib_resolution = hrflib_resolution
        self.rescale_hrflib_amplitude = rescale_hrflib_amplitude
        self.microtime_factor = int(
            self.tr / self.hrflib_resolution
        )  # should be 15 in our case
        self.frame_times_microtime = (
            np.arange(0, self.ntrs * self.tr, self.hrflib_resolution) + self.stc_reftime
        )
        self.frf = FracRidgeRegressor(fracs=1.0, fit_intercept=False, normalize=False)
        # mcnc settings
        self.mcnc_n_sig, self.mcnc_n_mes = mcnc_nsig, mcnc_nmes
        self.mcnc_njobs = mcnc_njobs
        self.mcnc_ddof = mcnc_ddof
        # directories and files
        self.outdir = pjoin(
            self.bidsroot, "derivatives", self.out_deriv_name, f"sub-{self.subject}"
        )
        self.best_hrf_nii = pjoin(self.outdir, "best_hrf_inds.nii.gz")
        self.best_frac_inds_nii = pjoin(self.outdir, "best_frac_inds.nii.gz")
        self.max_performance_nii = pjoin(self.outdir, "max_performance.nii.gz")
        self.best_fracs_nii = pjoin(self.outdir, "best_fracs.nii.gz")
        self.out_dtype = out_dtype
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def _make_design_df(self, event_file):
        """
        Make data frame containing specifying each trial as a separate condition.
        Also returns the list of condition names, and condition names for the repeated trials.
        """
        design_df = pd.read_csv(event_file, sep="\t")[
            ["duration", "onset", "file_path", "trial_type"]
        ]
        rep_condnames = design_df.loc[
            design_df.trial_type == "test", "file_path"
        ].to_numpy()
        design_df = design_df.drop(columns="trial_type")
        design_df = design_df.rename(columns={"file_path": "trial_type"})
        design_df = design_df.sort_values(by="onset", ignore_index=True)
        return design_df, rep_condnames

    def _onoff_df_from_design_df(self, design_df):
        """Take a single trial design data frame and turn it into an onoff design data frame"""
        onoff_df = design_df.copy(deep=True)
        onoff_df["trial_type"] = "onoff"
        return onoff_df

    def make_design_dfs(self, event_files):
        """
        Given our event files, give us the design data frames and condition names per run
        as well as the list of unique names (across all runs) of repeated conditions.
        """
        designs_zipped = [
            self._make_design_df(ef)
            for ef in tqdm(event_files, desc="reading event files")
        ]
        design_dfs, rep_condnames_runs = list(zip(*designs_zipped))
        rep_condnames = np.unique(np.hstack(rep_condnames_runs))
        onoff_dfs = [self._onoff_df_from_design_df(ddf) for ddf in design_dfs]
        return design_dfs, onoff_dfs, rep_condnames

    def convolve_designmat(self, designmat, rescale_hrflib_amplitude=True):
        """
        Convolve a boxcar design matrix with each hrf in self.hrflib.
        Returns array of shape (nhrfs, ntrs, ntrials)
        """
        convolved_ups = np.zeros(
            shape=(self.nhrfs, self.ntrs * self.microtime_factor, designmat.shape[1])
        )
        for hrf_i in range(self.nhrfs):
            conv_thishrf_ups = np.apply_along_axis(
                lambda m: np.convolve(m, self.hrflib[:, hrf_i], mode="full"),
                arr=designmat,
                axis=0,
            )[: self.ntrs * self.microtime_factor, :]
            if rescale_hrflib_amplitude:
                conv_thishrf_ups = np.nan_to_num(
                    conv_thishrf_ups / conv_thishrf_ups.max(axis=0)
                )
            convolved_ups[hrf_i] = conv_thishrf_ups
        convolved_designmat = convolved_ups[:, :: self.microtime_factor, :]
        return convolved_designmat

    def make_designs(self, event_files, normalize_convolved_regressors: bool = False):
        """
        Make convolved design matrices for each event file, also return condition names per run
        and list of unique names of repeated stimuli (across all runs)
        """
        design_dfs, onoff_dfs, rep_condnames = self.make_design_dfs(event_files)
        with Parallel(n_jobs=-1) as parallel:
            designmats = parallel(
                delayed(df_to_boxcar_design)(df, self.frame_times_microtime)
                for df in tqdm(design_dfs, "making single-trail design matrices")
            )
        with Parallel(n_jobs=-1) as parallel:
            onoffmats = parallel(
                delayed(df_to_boxcar_design)(df, self.frame_times_microtime)
                for df in tqdm(onoff_dfs, "making onoff design matrices")
            )
        # nilearn sorts conditions alphabetically, hence get them from designmat instead of dataframe
        condnames_runs = [dm.columns for dm in designmats]
        convolved_designs = [
            self.convolve_designmat(
                designmat, rescale_hrflib_amplitude=self.rescale_hrflib_amplitude
            )
            for designmat in tqdm(
                designmats,
                desc="Convolving single-trial designs for each run with HRF library",
                position=0,
                leave=True,
            )
        ]
        convolved_onoff = [
            self.convolve_designmat(
                onoffmat, rescale_hrflib_amplitude=self.rescale_hrflib_amplitude
            )
            for onoffmat in tqdm(
                onoffmats,
                desc="Convolving onoff designs for each run with HRF library",
                position=0,
                leave=True,
            )
        ]
        if normalize_convolved_regressors:
            convolved_designs = [zscore(arr, axis=1) for arr in convolved_designs]
            convolved_onoff = [zscore(arr, axis=1) for arr in convolved_onoff]
        return convolved_designs, convolved_onoff, condnames_runs, rep_condnames

    def make_noise_mat(self, nuisance_tsv, ica_txt=None, add_constant=False):
        """
        Make design matrix for noise regressors obtained from fmripreps nuisance tsv files
        and/or our manually classified ICs.
        """
        nuisance_df = get_nuisance_df(
            self.noiseregs, nuisance_tsv, include_all_aroma=self.include_all_aroma
        )
        if ica_txt:
            ica_arr = np.loadtxt(ica_txt)
            nuisance_df = pd.DataFrame(
                np.hstack([nuisance_df.to_numpy(), ica_arr]),
                columns=[
                    f"noisereg-{i}"
                    for i in range(nuisance_df.shape[1] + ica_arr.shape[1])
                ],
            )
        dropcols = [] if add_constant else ["constant"]
        return make_first_level_design_matrix(
            frame_times=self.frame_times_tr,
            add_regs=nuisance_df,
            hrf_model=None,
            drift_model=self.drift_model,
            drift_order=self.drift_order,
            high_pass=self.high_pass,
            events=None,
        ).drop(columns=dropcols)

    def regress_out_noise_runwise(self, noise_mats, data, zscore_residuals=True):
        """
        Regress the noise matrices out of our data separately for each session (to save memory).
        Original data is overwritten.
        """
        # fit intercept only if data was not rescaled runwise
        fitint = True if self.rescale_runwise_data in ["off", "psc"] else False
        for run_i in tqdm(range(self.nruns_total), desc="Regressing out noise runwise"):
            start_sample, stop_sample = run_i * self.ntrs, run_i * self.ntrs + self.ntrs
            data_run = data[start_sample:stop_sample]
            data_filtered = regress_out(
                noise_mats[run_i],
                data_run,
                lr_kws=dict(
                    copy_X=False, fit_intercept=fitint, normalize=False, n_jobs=-1
                ),
            )
            # overwrite raw data
            if zscore_residuals:
                data[start_sample:stop_sample] = np.nan_to_num(
                    zscore(data_filtered, axis=0)
                )
            else:
                data[start_sample:stop_sample] = np.nan_to_num(data_filtered)
        return data

    def orthogonalize_designmats(self, convolved_designs, convolved_onoff, noise_mats):
        """
        Predict the design regressors with the noise regressors and only keep the residuals.
        """
        convolved_designs = [
            np.stack(
                [regress_out(noisemat, designmat[hrf_i]) for hrf_i in range(self.nhrfs)]
            )
            for noisemat, designmat in zip(noise_mats, convolved_designs)
        ]
        convolved_onoff = [
            np.stack(
                [regress_out(noisemat, designmat[hrf_i]) for hrf_i in range(self.nhrfs)]
            )
            for noisemat, designmat in zip(noise_mats, convolved_onoff)
        ]
        return convolved_designs, convolved_onoff

    def overfit_hrf(self, data, convolved_designs, noise_mats=[], chunksize_runs=1):
        """
        Find best HRF per voxel measured by best within-sample r-squared with a single-trial design
        and no regularization. Noise regressors are optional and can be left out if data was cleaned beforehand.
        """
        n_chunks = int(self.nruns_total / chunksize_runs)
        start = time.time()
        # chunk inputs for parallelization
        print(f"HRF overfitting with {self.n_parallel_hrf} chunks in parallel")
        start_is = [chunk_i * chunksize_runs for chunk_i in range(n_chunks)]
        stop_is = [start_i + chunksize_runs for start_i in start_is]
        datachunks = [
            data[start_i * self.ntrs : stop_i * self.ntrs]
            for start_i, stop_i in zip(start_is, stop_is)
        ]
        designchunks = [
            convolved_designs[start_i:stop_i]
            for start_i, stop_i in zip(start_is, stop_is)
        ]
        if not noise_mats:
            noisechunks = [[]] * n_chunks
        else:
            noisechunks = [
                noise_mats[start_i:stop_i] for start_i, stop_i in zip(start_is, stop_is)
            ]
        # Fit to each chunk
        with Parallel(n_jobs=self.n_parallel_hrf) as parallel:
            chunk_scores = parallel(
                delayed(overfit_hrf_to_chunk)(
                    data_, designs_, noise_mats_, self.nvox_masked, self.nhrfs
                )
                for data_, designs_, noise_mats_ in zip(
                    datachunks, designchunks, noisechunks
                )
            )
        # aggregate
        mean_r2s = np.nanmean(np.stack(chunk_scores), axis=0)
        best_hrf_inds = np.argmax(mean_r2s, axis=0)
        print(f"HRF overfitting completed in {(time.time() - start) / 60:.1f} minutes")
        return best_hrf_inds, mean_r2s

    def get_repeated_betas(
        self,
        data,
        convolved_designs,
        noise_mats,
        condnames_runs,
        rep_condnames,
        hrf_inds,
    ):
        """
        Fit single trial GLM to each session and return the beta estimates of the repeatedly presented stimuli.
        """
        betas_per_session = []
        for ses_i in tqdm(
            range(self.n_sessions), desc="Getting repeated stimuli runwise", leave=True
        ):
            startrun, stoprun = (
                ses_i * self.nruns_perses,
                ses_i * self.nruns_perses + self.nruns_perses,
            )
            startsample, stopsample = startrun * self.ntrs, stoprun * self.ntrs
            data_ses = data[startsample:stopsample]
            designs_ses = convolved_designs[startrun:stoprun]
            noisemats_ses = noise_mats[startrun:stoprun] if noise_mats else []
            condnames_ses = np.hstack(condnames_runs[startrun:stoprun])
            repis_ses = np.hstack(
                [np.where(condnames_ses == repcond) for repcond in rep_condnames]
            ).squeeze()
            # _ = get_betas_run(0, 0, data_ses, designs_ses, noisemats_ses, hrf_inds, self.fracs, self.nvox_masked)
            with Parallel(n_jobs=-1) as parallel:
                betas_per_run = parallel(
                    delayed(
                        get_betas_run
                    )(  # TODO: could be parallelized more elegantly
                        ses_i,
                        run_i,
                        data_ses,
                        designs_ses,
                        noisemats_ses,
                        hrf_inds,
                        self.fracs,
                        self.nvox_masked,
                    )
                    for run_i in range(self.nruns_perses)
                )
            betas_per_session.append(np.concatenate(betas_per_run, axis=0)[repis_ses])
        return betas_per_session

    def splithalf(self, betas_per_session, n_splits: int = 0):
        """
        Calculate mean split-half correlation across sessions in order to evaluate best alpha fraction.
        Within each split, beta estimates of respective stimuli will be averaged.
        """
        combs = combinations(np.arange(self.n_sessions), int(self.n_sessions / 2))
        if n_splits:
            combs = list(combs)
            random.shuffle(combs)
            combs = combs[:n_splits]
        with Parallel(n_jobs=self.n_parallel_splithalf) as parallel:
            split_performances = parallel(
                delayed(eval_split_comb)(
                    comb_i,
                    comb,
                    betas_per_session,
                    self.nfracs,
                    self.use_spearman_brown,
                    self.unregularized_targets,
                    self.perf_metric,
                )
                for comb_i, comb in enumerate(combs)
            )
        performances = np.mean(np.stack(split_performances), axis=0)
        return performances

    def loo(self, betas_per_session):
        """
        Leave-one-out cross validation
        """
        with Parallel(n_jobs=self.n_parallel_splithalf) as parallel:
            performance_folds = parallel(
                delayed(eval_loo)(
                    test_i,
                    betas_per_session,
                    self.nfracs,
                    self.unregularized_targets,
                    self.perf_metric,
                )
                for test_i in range(self.n_sessions)
            )
        performances = np.mean(np.stack(performance_folds), axis=0)
        return performances

    def combined_tuning(
        self, data, convolved_designs, condnames_runs, rep_condnames, noise_mats
    ):
        """
        Find best pair of hyperparameters by searching through all combinations.
        """
        print("running LOO for all HRFs and alpha fractions")
        performances_hrfs = []  # one for each HRF
        for hrf_i in tqdm(
            range(self.nhrfs),
            desc="running LOO for all HRFs and alpha fractions",
            leave=True,
        ):
            hrf_inds = np.full(shape=self.nvox_masked, fill_value=hrf_i, dtype=int)
            print(f"getting repeated betas for HRF {hrf_i}")
            betas_per_session = self.get_repeated_betas(
                data,
                convolved_designs,
                noise_mats,
                condnames_runs,
                rep_condnames,
                hrf_inds,
            )
            print(f"running {self.tuning_procedure} for HRF {hrf_i}")
            if self.cv_scheme == "mcnc":
                performances = self.mcnc(betas_per_session)
            elif self.cv_scheme == "splithalf":
                performances = self.splithalf(betas_per_session)
            elif self.cv_scheme == "loo":
                performances = self.loo(betas_per_session)
            performances_hrfs.append(performances)
        performances_hrfs = np.stack(
            performances_hrfs
        )  # will have shape (hrf, fracs, nvox)???
        performances_combined = performances_hrfs.reshape(
            (self.nhrfs * self.nfracs, self.nvox_masked)
        )
        best_hrf_inds, best_frac_inds = np.unravel_index(
            np.argmax(performances_combined, axis=0), shape=(self.nhrfs, self.nfracs)
        )
        max_performance = np.max(performances_combined, axis=0)
        return best_hrf_inds, best_frac_inds, max_performance

    def mcnc(self, betas_per_session):
        """
        Use Monte Carlo noise ceiling to determine best alpha fraction for each voxel.
        """
        # stack over sessions
        betas_stacked = (
            np.stack(betas_per_session).T.swapaxes(2, 0).swapaxes(2, 1).swapaxes(2, 3)
        )
        del betas_per_session
        ncs_per_frac = []
        for frac_i in tqdm(range(self.nfracs), desc="MCNC for different fracs"):
            betas_frac = betas_stacked[frac_i]
            # estimate signal and noise distribution
            mn = betas_frac.mean(axis=1)
            se = betas_frac.var(axis=1, ddof=self.mcnc_ddof)
            noisevar = se.mean(axis=1)
            noisesd = np.sqrt(noisevar)
            overallmn = mn.mean(axis=1)
            overallvar = np.var(mn, axis=1, ddof=self.mcnc_ddof)
            signalsd = np.sqrt(np.clip(overallvar - noisevar, 0, None))
            # repeat for sampling convenience
            overallmn = np.repeat(overallmn[:, np.newaxis], 100, axis=-1)
            signalsd = np.repeat(signalsd[:, np.newaxis], 100, axis=-1)
            noisesd = np.repeat(noisesd[:, np.newaxis], 100, axis=-1)
            # run simulations in parallel
            r2s = Parallel(n_jobs=self.mcnc_njobs)(
                delayed(_run_job_sim_notnested)(overallmn, signalsd, noisesd)
                for _ in range(self.mcnc_n_sig)
            )
            r2s = np.stack(r2s, axis=-1)
            ncs_per_frac.append(np.median(r2s, axis=-1))
        performances = np.stack(ncs_per_frac)
        return performances

    def final_fit(
        self, data, convolved_designs, best_param_inds, condnames_runs, noise_mats
    ):
        """
        Fit the model with best HRF and alpha fraction per voxel to obtain final single trial beta estimates.
        """
        for ses_i in tqdm(range(self.n_sessions), desc="sessions"):
            sesdir = pjoin(self.outdir, f"ses-things{ses_i + 1:02d}")
            if not os.path.exists(sesdir):
                os.makedirs(sesdir)
            for run_i in tqdm(range(self.nruns_perses), desc="runs"):
                # figure out indices
                flat_i = ses_i * self.nruns_perses + run_i
                startsample, stopsample = (
                    flat_i * self.ntrs,
                    flat_i * self.ntrs + self.ntrs,
                )
                nconds = len(condnames_runs[flat_i])
                # iterate over voxel sets and populate our results array
                betas = np.zeros(shape=(self.nvox_masked, nconds))
                for param_i in tqdm(
                    np.unique(best_param_inds), "parameter combinations"
                ):
                    hrf_i, frac_i = np.unravel_index(
                        param_i, shape=(self.nhrfs, self.nfracs)
                    )
                    voxel_inds = np.where(best_param_inds == param_i)
                    data_sub = data[startsample:stopsample, voxel_inds[0]].squeeze()
                    design = convolved_designs[flat_i][hrf_i]
                    if noise_mats:
                        design = np.hstack([design, noise_mats[flat_i]])
                    if self.match_scale_runwise:
                        self.frf.fracs = [self.fracs[frac_i], 1.0]
                        self.frf.fit(design, data_sub)
                        betas_thisparam = match_scale(
                            self.frf.coef_[:nconds, 0], self.frf.coef_[:nconds, 1]
                        )
                    else:
                        self.frf.fracs = self.fracs[frac_i]
                        self.frf.fit(design, data_sub)
                        betas_thisparam = self.frf.coef_[:nconds]
                    betas[voxel_inds] = betas_thisparam.T
                # save betas and condition names for this run to file
                betas_nii = pjoin(
                    sesdir,
                    f"sub-{self.subject}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz",
                )
                conds_tsv = betas_nii.replace("_betas.nii.gz", "_conditions.tsv")
                betas_img = unmask(betas.T.astype(self.out_dtype), self.union_mask)
                betas_img.to_filename(betas_nii)
                pd.DataFrame(condnames_runs[flat_i]).to_csv(
                    conds_tsv, sep="\t", header=["image_filename"]
                )

    def run(self):
        print("\nLoading design and noise regressors\n")
        event_files, bold_files, nuisance_tsvs, masks = self.get_inputs()
        self.add_union_mask(masks)

        (
            convolved_designs,
            convolved_onoff,
            condnames_runs,
            rep_condnames,
        ) = self.make_designs(event_files)

        if self.manual_ica_regressors:
            ica_tsvs = self._get_ica_txts()
            noise_mats = [
                self.make_noise_mat(nuisance_tsv, ica_tsv)
                for nuisance_tsv, ica_tsv in zip(nuisance_tsvs, ica_tsvs)
            ]
        else:
            noise_mats = [
                self.make_noise_mat(nuisance_tsv) for nuisance_tsv in nuisance_tsvs
            ]

        if self.standardize_noiseregs:
            print("\nStandardizing noise regressors")
            noise_mats = [np.nan_to_num(zscore(m, axis=0)) for m in noise_mats]
        if self.standardize_trialegs:
            print("\nStandardizing trial regressors")
            convolved_designs = [
                np.nan_to_num(zscore(m, axis=1)) for m in convolved_designs
            ]
            convolved_onoff = [
                np.nan_to_num(zscore(m, axis=1)) for m in convolved_onoff
            ]

        if self.on_residuals_design:
            print("\nRegressing out noise from design\n")
            convolved_designs, convolved_onoff = self.orthogonalize_designmats(
                convolved_designs, convolved_onoff, noise_mats
            )

        print("\nLoading data\n")
        data = self.vstack_data_masked(
            bold_files, rescale_runwise=self.rescale_runwise_data
        )

        if self.on_residuals_data:
            print("\nRegressing out noise from data\n")
            data = self.regress_out_noise_runwise(
                noise_mats, data, zscore_residuals=True
            )

        if self.on_residuals_data or self.on_residuals_design:
            print(
                "\nOmitting noise regressors for model fitting since data and/or design was orthogonalized\n"
            )
            noise_mats = []

        if self.tuning_procedure == "combined":
            if (
                self.usecache
                and os.path.exists(self.best_hrf_nii)
                and os.path.exists(self.best_frac_inds_nii)
            ):
                print("\nLoading HRF indices and alpha fractions from pre-stored files")
                best_hrf_inds = apply_mask(
                    self.best_hrf_nii, self.union_mask, dtype=int
                )
                best_frac_inds = apply_mask(
                    self.best_frac_inds_nii, self.union_mask, dtype=int
                )
            else:
                print("\nCombined tuning of HRF and regularization")
                best_hrf_inds, best_frac_inds, max_performance = self.combined_tuning(
                    data, convolved_designs, condnames_runs, rep_condnames, noise_mats
                )
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
                unmask(max_performance, self.union_mask).to_filename(
                    self.max_performance_nii
                )

        elif self.tuning_procedure == "stepwise":
            if self.assume_hrf:
                print(
                    f"\nAssuming HRF with index {self.assume_hrf}, creating temporary file\n"
                )
                assert np.abs(self.assume_hrf) < self.nhrfs
                best_hrf_inds = np.full(
                    fill_value=self.assume_hrf, shape=self.nvox_masked, dtype=int
                )
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)

            if self.usecache and os.path.exists(self.best_hrf_nii):
                print("\nLoading HRF indices from pre-stored file\n")
                best_hrf_inds = apply_mask(
                    self.best_hrf_nii, self.union_mask, dtype=int
                )
            else:
                print("\nOverfitting HRF\n")
                if self.overfit_hrf_model == "onoff":
                    best_hrf_inds, _ = self.overfit_hrf(
                        data, convolved_onoff, noise_mats
                    )
                else:
                    best_hrf_inds, _ = self.overfit_hrf(
                        data, convolved_designs, noise_mats
                    )
                # save best HRF per voxel
                unmask(best_hrf_inds, self.union_mask).to_filename(self.best_hrf_nii)
            # regularization
            if self.usecache and os.path.exists(self.best_frac_inds_nii):
                print("\nLoading best alpha fractions from pre-stored file\n")
                best_frac_inds = apply_mask(
                    self.best_frac_inds_nii, self.union_mask, dtype=int
                )
            else:
                if self.cv_scheme == "unregularized":
                    print(
                        "\nSkipping cross-validation and calculating unregularized betas\n"
                    )
                    best_frac_inds = np.full(
                        fill_value=self.nfracs - 1, shape=self.nvox_masked, dtype=int
                    )
                else:
                    print(
                        "\nEstimating betas of repeated stimuli for each session and each alpha fraction\n"
                    )
                    betas_per_session = self.get_repeated_betas(
                        data,
                        convolved_designs,
                        noise_mats,
                        condnames_runs,
                        rep_condnames,
                        best_hrf_inds,
                    )
                    print(
                        f"\nUsing {self.cv_scheme} to find best alpha fraction per voxel"
                    )
                    if self.cv_scheme == "mcnc":
                        performances = self.mcnc(betas_per_session)
                    elif self.cv_scheme == "splithalf":
                        print(
                            f"Using unregularized targets = {self.unregularized_targets}"
                        )
                        performances = self.splithalf(betas_per_session)
                    elif self.cv_scheme == "loo":
                        performances = self.loo(betas_per_session)
                        # best_frac_inds = np.argmax(performance, axis=0)
                        # max_performance = np.max(performance, axis=0)
                    max_performance = np.max(performances, axis=0)
                    best_frac_inds = np.argmax(performances, axis=0)
                    unmask(max_performance, self.union_mask).to_filename(
                        self.max_performance_nii
                    )

        # save best regularization parameter
        best_fracs = self.fracs[best_frac_inds]
        for arr, fname in zip(
            [best_frac_inds, best_fracs], [self.best_frac_inds_nii, self.best_fracs_nii]
        ):
            unmask(arr, self.union_mask).to_filename(fname)

        print("\nFinal fit\n")
        best_param_inds = np.ravel_multi_index(
            (best_hrf_inds, best_frac_inds), (self.nhrfs, self.nfracs)
        )
        self.final_fit(
            data, convolved_designs, best_param_inds, condnames_runs, noise_mats
        )
        print("\nDone.\n")


def get_betas_run(
    ses_i,
    run_i,
    data_ses,
    designs_ses,
    noisemats_ses,
    hrf_inds,
    fracs,
    nvox_masked,
    nconds=92,
    ntrs=284,
    fr_kws=dict(fit_intercept=False, normalize=False),
):
    fr = FracRidgeRegressor(fracs=fracs, **fr_kws)
    betas = np.zeros(shape=(nconds, len(fracs), nvox_masked), dtype=np.single)
    for hrf_i in tqdm(
        np.unique(hrf_inds),
        desc=f"Getting repeated betas, run {run_i} session {ses_i}",
        leave=False,
    ):
        voxel_inds = np.where(hrf_inds == hrf_i)
        data_vox = data_ses[run_i * ntrs : run_i * ntrs + ntrs, voxel_inds[0]]
        design = designs_ses[run_i][hrf_i]
        if noisemats_ses:
            design = np.hstack([design, noisemats_ses[run_i]])
        fr.fit(design, data_vox)
        if len(voxel_inds[0]) == 1:  # special case if only one voxel has this HRF index
            betas[:, :, voxel_inds[0]] = fr.coef_[:nconds, :, None]
        else:
            betas[:, :, voxel_inds[0]] = fr.coef_[:nconds]
    return betas


def overfit_hrf_to_chunk(
    data,
    convolved_designs,
    noise_mats,
    nvoxmasked,
    nhrfs,
    reghrf_kws=dict(fit_intercept=False, normalize=False, n_jobs=-1),
):
    reghrf = LinearRegression(**reghrf_kws)
    scores = np.zeros(shape=(nhrfs, nvoxmasked))
    for hrf_i in tqdm(
        range(nhrfs), desc="Iterating over sub-chunks for HRF overfitting"
    ):
        design_hrf = block_diag(*[rundesign[hrf_i] for rundesign in convolved_designs])
        if noise_mats:
            design_hrf = np.hstack([design_hrf, block_diag(*noise_mats)])
        reghrf.fit(design_hrf, data)
        scores[hrf_i] = r2_score(
            data, reghrf.predict(design_hrf), multioutput="raw_values"
        )
    return scores


def eval_split_comb(
    comb_i,
    comb,
    betas_per_session,
    nfracs,
    use_spearman_brown=False,
    unregularized_targets=True,
    metric="l2",
):
    start = time.time()
    split_train = np.mean(
        np.stack([b for i, b in enumerate(betas_per_session) if i in comb]), axis=0
    )
    split_test = np.mean(
        np.stack([b for i, b in enumerate(betas_per_session) if i not in comb]), axis=0
    )
    # each element of betas_per_session is (nconds, len(fracs), nvox_masked)
    if metric == "correlation":
        if unregularized_targets:
            performance = np.stack(
                [
                    pearsonr_nd(split_train[:, frac_i, :], split_test[:, -1, :])
                    for frac_i in range(nfracs)
                ]
            )
        else:
            performance = np.stack(
                [
                    pearsonr_nd(split_train[:, frac_i, :], split_test[:, frac_i, :])
                    for frac_i in range(nfracs)
                ]
            )
        if use_spearman_brown:
            performance = spearman_brown(performance)
    elif metric == "l1":
        if unregularized_targets:
            err = np.stack(
                [
                    np.abs(split_train[:, frac_i, :] - split_test[:, -1, :]).sum(axis=0)
                    for frac_i in range(nfracs)
                ]
            )
        else:
            err = np.stack(
                [
                    np.abs(split_train[:, frac_i, :] - split_test[:, frac_i, :]).sum(
                        axis=0
                    )
                    for frac_i in range(nfracs)
                ]
            )
        performance = err * -1
    elif metric == "l2":
        if unregularized_targets:
            err = np.stack(
                [
                    np.square(
                        np.abs(split_train[:, frac_i, :] - split_test[:, -1, :])
                    ).sum(axis=0)
                    for frac_i in range(nfracs)
                ]
            )
        else:
            err = np.stack(
                [
                    np.square(
                        np.abs(split_train[:, frac_i, :] - split_test[:, frac_i, :])
                    ).sum(axis=0)
                    for frac_i in range(nfracs)
                ]
            )
        performance = err * -1
    print(
        f"Finished split combination {comb_i} in {((time.time() - start) / 60.):.2f} minutes"
    )
    return performance


def eval_loo(test_i, betas_per_session, nfracs, unregularized_targets, metric):
    # betas_per_session is list of len 12, each element is shape (nrepeated, nfracs, nvox)
    betas_test = betas_per_session[test_i]  # shape (nrepeated, nfracs, nvox)
    betas_train = np.mean(
        np.stack([b for sesi, b in enumerate(betas_per_session) if sesi != test_i]),
        axis=0,
    )  # shape (nrepeated, nfracs, nvox)
    if metric == "correlation":
        if unregularized_targets:
            performance = np.stack(
                [
                    pearsonr_nd(betas_train[:, frac_i, :], betas_test[:, -1, :])
                    for frac_i in range(nfracs)
                ]
            )
        else:
            performance = np.stack(
                [
                    pearsonr_nd(betas_train[:, frac_i, :], betas_test[:, frac_i, :])
                    for frac_i in range(nfracs)
                ]
            )
    elif metric == "l1":
        if unregularized_targets:
            err = np.stack(
                [
                    np.abs(betas_train[:, frac_i, :] - betas_test[:, -1, :]).sum(axis=0)
                    for frac_i in range(nfracs)
                ]
            )
        else:
            err = np.stack(
                [
                    np.abs(betas_train[:, frac_i, :] - betas_test[:, frac_i, :]).sum(
                        axis=0
                    )
                    for frac_i in range(nfracs)
                ]
            )
        performance = err * -1
    elif metric == "l2":
        if unregularized_targets:
            err = np.stack(
                [
                    np.square(
                        np.abs(
                            betas_train[
                                :,
                                frac_i,
                            ]
                            - betas_test[:, -1, :]
                        )
                    ).sum(axis=0)
                    for frac_i in range(nfracs)
                ]
            )
        else:
            err = np.stack(
                [
                    np.square(
                        np.abs(betas_train[:, frac_i, :] - betas_test[:, frac_i, :])
                    ).sum(axis=0)
                    for frac_i in range(nfracs)
                ]
            )
        performance = err * -1
    return performance.astype(np.single)


def list_stb_outputs_for_mcnc(
    sub: str = "01",
    bidsroot: str = "/LOCAL/ocontier/thingsmri/bids",
    betas_basedir: str = "/LOCAL/ocontier/thingsmri/bids/derivatives/betas",
) -> tuple:
    betas_dir = pjoin(betas_basedir, f"sub-{sub}")
    betas = []
    for ses_i in tqdm(range(12), desc="Loading betas of repeated stimuli"):
        sesdir = pjoin(betas_dir, f"ses-things{ses_i + 1:02d}")
        rawdir = pjoin(
            bidsroot, "rawdata", f"sub-{sub}", f"ses-things{ses_i + 1:02d}", "func"
        )
        event_tsvs = [
            pjoin(
                rawdir,
                f"sub-{sub}_ses-things{ses_i + 1:02d}_task-things_run-{run_i + 1:02d}_events.tsv",
            )
            for run_i in range(10)
        ]
        cond_tsvs = [
            pjoin(
                sesdir,
                f"sub-{sub}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_conditions.tsv",
            )
            for run_i in range(10)
        ]
        conds = np.hstack(
            [
                pd.read_csv(tsv, sep="\t")["image_filename"].to_numpy()
                for tsv in cond_tsvs
            ]
        )
        event_dfs = [pd.read_csv(tsv, sep="\t") for tsv in event_tsvs]
        if ses_i == 0:
            repcondnames = np.unique(
                np.hstack(
                    [
                        df[df["trial_type"] == "test"]["file_path"].to_numpy(dtype=str)
                        for df in event_dfs
                    ]
                )
            )
        run_niis = [
            pjoin(
                sesdir,
                f"sub-{sub}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz",
            )
            for run_i in range(10)
        ]
        ses_betas = np.moveaxis(
            np.concatenate(
                [load_img(ni).get_fdata(dtype=np.single) for ni in run_niis], axis=3
            ),
            -1,
            0,
        )
        repcond_is = np.hstack(
            [np.where(conds == repcond) for repcond in repcondnames]
        ).squeeze()
        rep_betas = ses_betas[repcond_is]
        betas.append(rep_betas)
    betas = np.moveaxis(
        (np.moveaxis(np.stack(betas), 0, -1)), 0, -1
    )  # shape (voxX, voxY, nRepetitions, nStimuli)
    return betas, run_niis[0]


def _run_job_sim_notnested(overallmn, signalsd, noisesd):
    simsig = normal(overallmn, signalsd)
    mes = simsig + normal(0, noisesd)
    return r2_ndarr(simsig, mes)


def match_lsq(reg, unreg, nnls: bool = False):
    lr = LinearRegression(fit_intercept=True, n_jobs=-1, positive=nnls)
    lr.fit(reg.reshape(-1, 1), unreg)
    pred = lr.predict(reg.reshape(-1, 1))
    return pred


def match_nnslope(reg, unreg):
    reg_c, unreg_c = reg - reg.mean(), unreg - unreg.mean()
    lr = LinearRegression(fit_intercept=False, n_jobs=-1, positive=True)
    lr.fit(reg_c.reshape(-1, 1), unreg_c)
    pred_c = lr.predict(reg_c.reshape(-1, 1))
    pred = pred_c + unreg.mean()
    return pred


def load_betas(
    bidsroot: str,
    sub: str,
    mask: str,
    betas_derivname: str = "betas_loo/on_residuals/scalematched",
    smoothing=0.0,
    dtype=np.single,
) -> np.ndarray:
    if not smoothing:
        smoothing = None
    betasdir = pjoin(bidsroot, "derivatives", betas_derivname, f"sub-{sub}")
    betafiles = [
        pjoin(
            betasdir,
            f"ses-things{ses_i:02d}",
            f"sub-{sub}_ses-things{ses_i:02d}_run-{run_i:02d}_betas.nii.gz",
        )
        for ses_i in range(1, 13)
        for run_i in range(1, 11)
    ]
    for b in betafiles:
        assert os.path.exists(b)
    with Parallel(n_jobs=-1) as parallel:
        betas_l = parallel(
            delayed(apply_mask_smoothed)(bf, mask, smoothing, dtype)
            for bf in betafiles
            )
        
    betas = np.vstack(betas_l)
    return betas  # shape (ntrials, nvoxel)


def load_betas_img(
    sub,
    bidsroot,
    betas_derivname="betas_loo/on_residuals/scalematched",
    dtype=np.single,
):
    betafiles = [
        pjoin(
            bidsroot,
            "derivatives",
            betas_derivname,
            f"ses-things{ses_i:02d}",
            f"sub-{sub}_ses-things{ses_i:02d}_run-{run_i:02d}_betas.nii.gz",
        )
        for ses_i in range(1, 13)
        for run_i in range(1, 11)
    ]
    return load_img(betafiles)


def load_filenames(sub, bidsroot, betas_derivname):
    # load condition names
    betas_basedir = pjoin(bidsroot, "derivatives", betas_derivname)
    tsv_files = [
        pjoin(
            betas_basedir,
            f"sub-{sub}",
            f"ses-things{ses_i:02d}",
            f"sub-{sub}_ses-things{ses_i:02d}_run-{run_i:02d}_conditions.tsv",
        )
        for ses_i in range(1, 13)
        for run_i in range(1, 11)
    ]
    filenames = np.hstack(
        [pd.read_csv(tsv, sep="\t").image_filename.to_numpy() for tsv in tsv_files]
    ).astype(str)
    return filenames


def filter_catch_trials(betas, filenames):
    noncatch_is = np.array([False if "catch" in f else True for f in filenames])
    betas_noncatch, filenames_noncatch = betas[noncatch_is], filenames[noncatch_is]
    return betas_noncatch, filenames_noncatch, noncatch_is


def average_betas_per_concept(betas, filenames):
    """
    catch trials should have been excluded. Does not distinguish between test and normal trials.
    """
    trial_concepts = np.array([fn.split("/")[1][:-8] for fn in filenames])
    concepts, counts = np.unique(trial_concepts, return_counts=True)
    assert np.sum(counts % 12) == 0
    betas_concepts = np.zeros((len(concepts), betas.shape[1]))
    for i, c_ in tqdm(enumerate(concepts), desc="averaging betas per concept"):
        mask = trial_concepts == c_
        assert np.sum(mask) % 12 == 0
        betas_c_ = betas[mask].mean(axis=0)
        betas_concepts[i] = betas_c_
    return np.stack(betas_concepts), concepts


def posthoc_scaling(
    sub="01",
    bidsroot="/LOCAL/ocontier/thingsmri/bids",
    reg_derivname="betas/regularized",
    unreg_derivname="betas/unregularized",
    out_derivname="betas/scalematched",
    njobs=50,
    nconds=92,
    method: str = "ols",
):
    """
    Use OLS to find (intercept) and scalar to match the regularized betas to the scale of the unregularized betas.
    """
    assert method in ["ols", "nnls", "nnslope"]
    outdir = pjoin(bidsroot, "derivatives", out_derivname, f"sub-{sub}")
    # get union mask
    thingsglm = THINGSGLM(bidsroot, sub, out_deriv_name="tmp")
    _, _, _, masks = thingsglm.get_inputs()
    thingsglm.add_union_mask(masks)
    # load regularized betas
    reg_betas = load_betas(sub, thingsglm.union_mask, bidsroot, reg_derivname)
    unreg_betas = load_betas(sub, thingsglm.union_mask, bidsroot, unreg_derivname)
    # run regressions in parallel
    nvox = reg_betas.shape[1]
    print("\nrescaling\n")
    if method in ["ols", "nnls"]:
        with Parallel(n_jobs=njobs) as parallel:
            matched_l = parallel(
                delayed(match_lsq)(
                    reg=reg_betas[:, vox_i],
                    unreg=unreg_betas[:, vox_i],
                    nnls=True if method == "nnls" else False,
                )
                for vox_i in tqdm(range(nvox), "voxels")
            )
    elif method == "nnslope":
        with Parallel(n_jobs=njobs) as parallel:
            matched_l = parallel(
                delayed(match_nnslope)(
                    reg=reg_betas[:, vox_i],
                    unreg=unreg_betas[:, vox_i],
                )
                for vox_i in tqdm(range(nvox), "voxels")
            )
    matched = np.stack(matched_l, axis=1)
    # save output
    for ses_i in tqdm(range(12), desc="saving output for each session"):
        sesdir = pjoin(outdir, f"ses-things{ses_i + 1:02d}")
        if not os.path.exists(sesdir):
            os.makedirs(sesdir)
        for run_i in tqdm(range(10), desc="runs"):
            flat_i = ses_i * 10 + run_i
            starti, stopi = flat_i * nconds, flat_i * nconds + nconds
            matched_img = unmask(matched[starti:stopi], thingsglm.union_mask)
            nii = pjoin(
                sesdir,
                f"sub-{sub}_ses-things{ses_i + 1:02d}_run-{run_i + 1:02d}_betas.nii.gz",
            )
            matched_img.to_filename(nii)
    # copy condition files
    orig_tsvs = glob.glob(
        pjoin(bidsroot, "derivatives", reg_derivname, f"sub-{sub}", "ses-*", "*.tsv")
    )
    matched_tsvs = [t.replace(reg_derivname, out_derivname) for t in orig_tsvs]
    for src, trgt in tqdm(zip(orig_tsvs, matched_tsvs), "copying tsv files"):
        copyfile(src, trgt)
    return


if __name__ == "__main__":

    sub, bidsroot = sys.argv[1], sys.argv[2]
    # get an unregularized estimate of responses
    stb_unreg = SingleTrialBetas(
        bidsroot, sub, cv_scheme="unregularized", out_deriv_name="betas_unregularized"
    )
    stb_unreg.run()
    # use cross-validation and l2 regularization
    stb_reg = SingleTrialBetas(bidsroot, sub, out_deriv_name="betas_regularized")
    stb_reg.run()
    # apply post-hoc scaling
    posthoc_scaling(
        sub=sub,
        bidsroot=bidsroot,
        reg_derivname="betas_regularized",
        unreg_derivname="betas_unregularized",
        out_derivname="betas_vol",
    )
