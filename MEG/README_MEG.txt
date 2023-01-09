README MEG 

We collected MEG data along with eyetracking data from four participants in twelve sessions. The BIDS-formatted data can be downloaded from OpenNeuro. 
This repository contains all codes for the MEG analysis and the eyetracking analyses. 
Running the scripts in order creates all the outputs necessary to recreate the plots shown in the paper. 
All codes are command-line executable (tested on Mac and Linux) and are written in Python (find full list of dependencies in environment_THINGS-MEG.yml file) & MATLAB (tested on MATLAB R2022a). 
In addition to standard toolboxes, the MATLAB toolbox CoSMoMVPA (https://www.cosmomvpa.org) and mne-matlab(https://github.com/mne-tools/mne-matlab) are required.

Description of MEG codes:
1)  PREPROCESSING
-   step1_preprocessing.py:                                 preprocess the BIDS-formatted data. [call from command line with inputs -participant and -bids_dir]
2)	DATA QUALITY
-	step2a_data_quality-head_position.py:                   extract the head position measurements from the raw meg data and calculate displacement across runs and sessions. [call from command line with inputs -bids_dir]
-	step2b_data_quality-ERPs.py:                            calculate ERFs for the repeat images in every session to see how consistent the data is across sessions. [call from command line with inputs -bids_dir]
-	step2c_data_quality-noiseceiling.py:                    calculate noise ceilings for the repeat images to assess how good the signal can be expected to be in each sensor group. [call from command line with inputs -bids_dir]
3)	VALIDATION
-	step3a_validation-animacy_size.py:                      run a cross-validated linear regression to predict animacy and size ratings from the MEG sensor activation patterns. [call from command line with inputs -bids_dir]. 
-   step3aa_plot_validation_size_animacy.m:                 plots the results of the a cross-validated linear regression (step 3a) using MATLAB. [call from command line with inputs bids_dir] 
-	step3b_validation-fmri_meg_combo.py:                    run a cross-validated linear regression to predict the univariate response in fMRI ROIs from the MEG sensor activation patterns over time. [call from command line with inputs -bids_dir]
-   step_3bb_plot_validation_fmri_meg_combo.m               plots the results of the a cross-validated linear regression (step 3b) using MATLAB). [call from command line with inputs bids_dir] 
-	Step3c_validation_pairwise_decoding_mne_to_cosmo.m:     helper script to transform the MNE-preprocessed data into a cosmo struct. [call from command line with inputs bids_dir and toolbox_dir] 
-	Step3d_validation_pairwise_decoding200.m:               function to run the pairwise decoding for the 200 repeat image trials (image-level decoding). [call from command line with inputs bids_dir, toolbox_dir, participant, blocknr, n_blocks. Variable n_blocks is used to parallelize the process. ~20GB memory is needed for each process] 
-	Step3e_validation_pairwise_decoding1854.m:              function to run the pairwise decoding for the 1854 image concepts (concept-level decoding). [call from command line with inputs bids_dir, toolbox_dir, participant, blocknr, n_blocks. Variable n_blocks is used to parallelize the process. ~240GB memory is needed for each process] 
-	Step3f_validation_pairwise_decoding_stack_plot.m:       pairwise decoding in step 3d and 3e was run in parallel chunks. This script combines the chunks together so we can plot, for example, an RDM. [call from command line with inputs bids_dir, toolbox_dir, imagewise_nblocks, sessionwise_nblocks]
-	Step3g_validation_pairwise_decoding_mds.m:              plotting the results of the pairwise decoding analyses alongside snapshot MDS plots that show distinction between high-level categories. [call from command line with inputs bids_dir and toolbox_dir] 
-	Step3h_validation_rsa_behaviour.m:                      correlate the behavioural pairwise similarities with the MEG decoding RDMs and plot. [call from command line with inputs bids_dir and toolbox_dir] 

Descriptions of eyetracking codes (can be found in the eyetracking subfolder)
1)	step1_eyetracking_preprocess.py:                        extract the x-, y-, and pupil-data from the raw MEG data, preprocess, and epoch. [call from command line with inputs -bids_dir]
2)  step1a_eyetracking_plot_preprocessing.py:               plot overview of the preprocessing steps. [call from command line with inputs -bids_dir]
3)	step2_eyetracking_plot.py:                              load epoched eyetracking data and make the plots shown in the supplementary materials. [call from command line with inputs -bids_dir]




If you find any errors or if something does not work, please reach out to lina.teichmann@nih.gov



