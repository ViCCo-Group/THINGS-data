#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -participant
        -bids_dir

    OUTPUTS:
    epoched and cleaned data will be written into the preprocessing directory

    NOTES: 
    This script contains the following preprocessing steps:
    - channel exclusion (one malfunctioning channel ('MRO11-1609'), based on experimenter notes)
    - filtering (0.1 - 40Hz)
    - epoching (-100 - 1300ms) --> based on onsets of the optical sensor
    - baseline correction (zscore)
    - downsampling (200Hz)

    Use preprocessed data ("preprocessed_P{participant}-epo.fif") saved in preprocessing directory for the next steps

"""

import mne, os
import numpy as np
import pandas as pd 
from joblib import Parallel, delayed


#*****************************#
### PARAMETERS ###
#*****************************#

n_sessions                  = 12
trigger_amplitude           = 64
l_freq                      = 0.1
h_freq                      = 40
pre_stim_time               = -0.1
post_stim_time              = 1.3
std_deviations_above_below  = 4
output_resolution           = 200
trigger_channel             = 'UPPT001'


#*****************************#
### HELPER FUNCTIONS ###
#*****************************#
def setup_paths(meg_dir, session):
    run_paths,event_paths = [],[]
    for file in os.listdir(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/'):
        if file.endswith(".ds") and file.startswith("sub"):
            run_paths.append(os.path.join(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/', file))
        if file.endswith("events.tsv") and file.startswith("sub"):
            event_paths.append(os.path.join(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/', file))
    run_paths.sort()
    event_paths.sort()

    return run_paths, event_paths 

def read_raw(curr_path,session,run,participant):
    raw = mne.io.read_raw_ctf(curr_path,preload=True)
    # signal dropout in one run -- replacing values with median
    if participant == '1' and session == 11 and run == 4:  
        n_samples_exclude   = int(0.2/(1/raw.info['sfreq']))
        raw._data[:,np.argmin(np.abs(raw.times-13.4)):np.argmin(np.abs(raw.times-13.4))+n_samples_exclude] = np.repeat(np.median(raw._data,axis=1)[np.newaxis,...], n_samples_exclude, axis=0).T
    elif participant == '2' and session == 10 and run == 2: 
        n_samples_exclude = int(0.2/(1/raw.info['sfreq']))
        raw._data[:,np.argmin(np.abs(raw.times-59.8)):np.argmin(np.abs(raw.times-59.8))+n_samples_exclude] = np.repeat(np.median(raw._data,axis=1)[np.newaxis,...], n_samples_exclude, axis=0).T

    raw.drop_channels('MRO11-1609')
        
    return raw

def read_events(event_paths,run,raw):
    # load event file that has the corrected onset times (based on optical sensor and replace in the events variable)
    event_file = pd.read_csv(event_paths[run],sep='\t')
    event_file.value.fillna(999999,inplace=True)
    events = mne.find_events(raw, stim_channel=trigger_channel,initial_event=True)
    events = events[events[:,2]==trigger_amplitude]
    events[:,0] = event_file['sample']
    events[:,2] = event_file['value']
    return events

def concat_epochs(raw, events, epochs):
    if epochs:
        epochs_1 = mne.Epochs(raw, events, tmin = pre_stim_time, tmax = post_stim_time, picks = 'mag',baseline=None)
        epochs_1.info['dev_head_t'] = epochs.info['dev_head_t']
        epochs = mne.concatenate_epochs([epochs,epochs_1])
    else:
        epochs = mne.Epochs(raw, events, tmin = pre_stim_time, tmax = post_stim_time, picks = 'mag',baseline=None)
    return epochs

def baseline_correction(epochs):
    baselined_epochs = mne.baseline.rescale(data=epochs.get_data(),times=epochs.times,baseline=(None,0),mode='zscore',copy=False)
    epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin,event_id=epochs.event_id)
    return epochs

def stack_sessions(sourcedata_dir,preproc_dir,participant,session_epochs,output_resolution):
    for epochs in session_epochs:
        epochs.info['dev_head_t'] = session_epochs[0].info['dev_head_t']
    all_epochs = mne.concatenate_epochs(epochs_list = session_epochs, add_offset=True)
    all_epochs.metadata = pd.read_csv(f'{sourcedata_dir}/sample_attributes_P{str(participant)}.csv')
    all_epochs.decimate(decim=(1200/output_resolution))
    all_epochs.save(f'{preproc_dir}/preprocessed_P{str(participant)}-epo.fif', overwrite=True)
    print(all_epochs.info)


#*****************************#
### FUNCTION TO RUN PREPROCESSING ###
#*****************************#
def run_preprocessing(meg_dir,session,participant):
    epochs = []
    run_paths, event_paths = setup_paths(meg_dir, session)
    for run, curr_path in enumerate(run_paths):
        raw = read_raw(curr_path,session,run, participant)
        events = read_events(event_paths,run,raw)
        raw.filter(l_freq=l_freq,h_freq=h_freq)
        epochs = concat_epochs(raw, events, epochs)
        epochs.drop_bad()
    print(epochs.info)
    epochs = baseline_correction(epochs)
    return epochs


#*****************************#
### COMMAND LINE INPUTS ###
#*****************************#
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-participant",
        required=True,
        help='participant bids ID (e.g., 1)',
    )

    parser.add_argument(
        "-bids_dir",
        required=True,
        help='path to bids root',
    )

    args = parser.parse_args()
    
    bids_dir                    = args.bids_dir
    participant                 = args.participant
    meg_dir                     = f'{bids_dir}/sub-BIGMEG{participant}/'
    sourcedata_dir              = f'{bids_dir}/sourcedata/'
    preproc_dir                 = f'{bids_dir}/derivatives/preprocessed/'
    if not os.path.exists(preproc_dir):
        os.makedirs(preproc_dir)

    ####### Run preprocessing ########
    session_epochs = Parallel(n_jobs=12, backend="multiprocessing")(delayed(run_preprocessing)(meg_dir,session,participant) for session in range(1,n_sessions+1))
    stack_sessions(sourcedata_dir,preproc_dir,participant,session_epochs,output_resolution)
