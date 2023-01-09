#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -bids_dir

    OUTPUTS:
    epoched and cleaned data will be written into the preprocessing directory plus the metadata

    NOTES: 
    Preprocessing is done for every run separately and is based on Kret et al., 2019:
        # - removing invalid samples
        # - removing based on dilation speeds
        # - removing based on deviation from a fitted line
        # - detrending

"""

import pandas as pd 
import numpy as np
import math, mne, os
from scipy import stats
from scipy.signal import butter,filtfilt
from scipy.interpolate import interp1d
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Helvetica"
matplotlib.rcParams.update({'font.size': 12})


#*****************************#
### PARAMETERS ###
#*****************************#
n_participants              = 4
n_sessions                  = 12
n_runs                      = 10
trigger_amplitude           = 64

# MEG info
et_refreshrate              = 1200 # eye-tracker signal is recorded with MEG -- so it's the same resolution
meg_refreshrate             = 1200
trigger_channel             = 'UPPT001'
pd_channel                  = 'UADC016-2104'
eye_channel                 = ['UADC009-2104','UADC010-2104','UADC013-2104'] # x, y, pupil


# settings for eyetracker to volts conversion
minvoltage                  = -5
maxvoltage                  = 5
minrange                    = -0.2
maxrange                    = 1.2
screenleft                  = 0
screenright                 = 1023
screentop                   = 0
screenbottom                = 767

# parameters to transform from pixels to degrees
screenwidth_cm              = 42
screenheight_cm             = 32
screendistance_cm           = 75

screensize_pix              = [1024, 768]
pix_per_deg                 = screensize_pix[0]/(math.degrees(math.atan(screenwidth_cm/2/screendistance_cm)*2))
stim_size_deg               = 10
stim_width                  = stim_size_deg*pix_per_deg
stim_height                 = stim_size_deg*pix_per_deg


#*****************************#
### HELPER FUNCTIONS ###
#*****************************#
def pix_to_deg(full_size_pix,screensize_pix,screenwidth_cm,screendistance_cm):
    pix_per_cm = screensize_pix[0]/screenwidth_cm
    size_cm = full_size_pix/pix_per_cm
    dva = math.atan(size_cm/2/screendistance_cm)*2
    return np.rad2deg(dva)

# volts_to_pixels: converts voltages recorded by the MEG to pixels - (0,0) is the middle of the screen
def volts_to_pixels(x,y,pupil,minvoltage,maxvoltage,minrange,maxrange,screenbottom,screenleft,screenright,screentop,scaling_factor):
    S_x                                             = ((x-minvoltage)/(maxvoltage-minvoltage))*(maxrange-minrange)+minrange
    S_y                                             = ((y-minvoltage)/(maxvoltage-minvoltage))*(maxrange-minrange)+minrange
    Xgaze                                           = S_x*(screenright-screenleft+1)+screenleft
    Ygaze                                           = S_y*(screenbottom-screentop+1)+screentop
    # pupil_n                                         = (pupil-pupil.min())*scaling_factor
    return(Xgaze,Ygaze)

# deviation_calculator: fits a smooth line over the samples and checks how much each sample deviates from it 
def deviation_calculator(tv,dia,is_valid,t_interp,smooth_filt_a,smooth_filt_b):
    dia_valid                                       = dia[[x and y for x,y in zip(is_valid,~np.isnan(dia))]]
    t_valid                                         = tv[[x and y for x,y in zip(is_valid,~np.isnan(dia))]]
    interp_f_lin                                    = interp1d(t_valid,dia_valid,kind='linear',bounds_error=False)
    interp_f_near                                   = interp1d(t_valid,dia_valid,kind='nearest',fill_value='extrapolate')
    extrapolated                                    = interp_f_near(t_interp)
    uniform_baseline                                = interp_f_lin(t_interp)
    uniform_baseline[np.isnan(uniform_baseline)]    = extrapolated[np.isnan(uniform_baseline)]
    smooth_uniform_baseline                         = filtfilt(smooth_filt_b,smooth_filt_a,uniform_baseline)
    interp_f_baseline                               = interp1d(t_interp,smooth_uniform_baseline,kind='linear',bounds_error=False)
    smooth_baseline                                 = interp_f_baseline(tv)
    dev = np.abs(dia-smooth_baseline)

    return(dev,smooth_baseline)

# expand_gap: this pads significanly large gaps (>75ms). Before the gap we padded 100ms, after the gap for 150ms (based on Matthias Nau pipeline in NSD paper)
def expand_gap(tv,is_valid):
    min_gap_width                                   = 75
    max_gap_width                                   = 2000
    pad_back                                        = 100
    pad_forward                                     = 150
    valid_t                                         = tv[is_valid]
    valid_idx                                       = np.where(is_valid)[0]
    gaps                                            = np.diff(valid_t)
    needs_padding                                   = [x and y for x,y in zip(gaps>min_gap_width,gaps<max_gap_width)]
    gap_start_t                                     = valid_t[np.pad(needs_padding,(0,1),constant_values=False)]
    gap_end_t                                       = valid_t[np.pad(needs_padding,(1,0),constant_values=False)]

    remove_idx = []
    for i_start,i_end in zip(gap_start_t,gap_end_t):
        # when the gap is super large, it's most likely a recording artifact (and not an eyeblink), so we should clean around it more
        if i_end -i_start > 500:
            pb                                      = pad_back * 2
            pf                                      = pad_forward * 2
        else:
            pb                                      = pad_back
            pf                                      = pad_forward
        remove_idx.extend(np.where([x and y for x,y in zip(valid_t>(i_start-pb),valid_t<(i_end+pf))])[0])
    remove_idx                                      = np.unique(remove_idx)

    if remove_idx.any():
        is_valid[valid_idx[remove_idx]]             = False
    return is_valid

# remove_loners: see whether there are any chunks of data that are temporally isolated and relatively short. If yes, exclude them.
def remove_loners(is_valid,et_refreshrate):
    lonely_sample_max_length                        = 100 #in ms
    time_separation                                 = 40 #in ms
    valid_idx                                       = np.where(is_valid)[0]
    gap_start                                       = valid_idx[np.where(np.pad(np.diff(valid_idx),(0,1),constant_values=1)>1)]
    gap_end                                         = valid_idx[np.where(np.pad(np.diff(valid_idx),(1,0),constant_values=1)>1)]
    start_valid_idx                                 = [valid_idx[0]]
    end_valid_idx                                   = [valid_idx[-1]]
    valid_data_chunks                               = np.reshape(np.sort(np.concatenate([start_valid_idx,gap_start,gap_end,end_valid_idx])),[-1,2])
    size_valid_data_chunks                          = np.diff(valid_data_chunks,axis=1)
    size_idx                                        = np.where((size_valid_data_chunks/et_refreshrate*1000)<lonely_sample_max_length)[0]
    separation                                      = np.squeeze(np.diff(np.reshape(np.sort(np.concatenate([gap_start,gap_end])),[-1,2]),axis=1))
    sep_idx                                         = np.where(np.pad([i>(time_separation*1/et_refreshrate) for i in separation],(1,0)))[0]
    data_chunks_to_delete                           = valid_data_chunks[np.intersect1d(sep_idx,size_idx)]

    valid_out                                       = is_valid.copy()
    for i in data_chunks_to_delete:
        valid_out[np.arange(i[0],i[1]+1)]           = 0

    print('removed ' + str(is_valid.sum()-valid_out.sum()) + ' samples')
    
    return valid_out.astype(bool)


#*****************************#
### DATA LOADING FUNCTIONS ###
#*****************************#
# This function loads the data, if "iseyes" is true it will return the raw data for the eyetracking channels, otherwise it will return the raw data for the optical sensor to make the epochs
def load_raw_data(rootdir,p,s,r,trigger_channel,pd_channel,eye_channel,iseyes):
    sess_num                                        = str(s+1)
    run_num                                         = str(r+1)
    data_dir                                        = rootdir + '/sub-BIGMEG' + str(p)
    data_ses_dir                                    = data_dir + '/ses-' + sess_num.zfill(2) + '/meg'
    meg_fn                                          = data_ses_dir + '/sub-BIGMEG' + str(p) + '_ses-' + sess_num.zfill(2) + '_task-main_run-' + run_num.zfill(2) + '_meg.ds'
    print('loading participant ' + str(p) + ' session ' + sess_num + ' run ' + run_num + '...')

    raw                                             = mne.io.read_raw_ctf(meg_fn,preload=True)   
    if iseyes:
        raw_eyes                                    = raw.copy().pick_channels([eye_channel[0],eye_channel[1],eye_channel[2]])
        raw_eyes.get_data()
        print(raw_eyes.ch_names)
        return raw_eyes
    
    else:
        raw_triggers                                = raw.copy().pick_channels([pd_channel])
        raw_triggers.get_data()
        return raw_triggers

# The MEG saves the data in volts. Here we convert the volts to pixels for x/y and return a dataframe for easier handling
# I'm also median centering the data in case there was some drift over the session
def raw2df(raw_et,minvoltage,maxvoltage,minrange,maxrange,screenbottom,screenleft,screenright,screentop,screensize_pix):
    raw_et_df                                       = pd.DataFrame(raw_et._data.T,columns=['x_volts','y_volts','pupil'])
    #  not scaling the pupil anymore
    raw_et_df['x'],raw_et_df['y']                   = volts_to_pixels(raw_et_df['x_volts'],raw_et_df['y_volts'],raw_et_df['pupil'],minvoltage,maxvoltage,minrange,maxrange,screenbottom,screenleft,screenright,screentop,scaling_factor=978.982673828819)
    raw_et_df['x']                                  = raw_et_df['x']-screensize_pix[0]/2
    raw_et_df['x']                                  = raw_et_df['x']-np.median(raw_et_df['x'])
    raw_et_df['y']                                  = raw_et_df['y']-screensize_pix[1]/2
    raw_et_df['y']                                  = raw_et_df['y']-np.median(raw_et_df['y'])
    raw_et_df['pupil']                              = raw_et_df['pupil']-np.median(raw_et_df['pupil'])
    return raw_et_df


#*****************************#
### PREPROC FUNCTIONS ###
#*****************************#
# Step 1: We are removing all samples that either go beyond the stimulus height or width (10 degrees), or where the recorded pupil size is outside of the recording range (+/-5 volts)
def remove_invalid_samples(eyes,tv):
    withinwidth                                     = np.abs(eyes['x'])<(stim_width/2)
    withinheight                                    = np.abs(eyes['y'])<(stim_height/2)
    is_valid                                        = np.array([x and y for x,y in zip(withinwidth,withinheight)]).astype(bool)
    if not any(is_valid):
        is_valid                                    = remove_loners(is_valid,et_refreshrate)
        is_valid                                    = expand_gap(tv,is_valid)

    return is_valid.astype(bool)

# Step 2: Checking how much the pupil dliation changes from timepoint to timepoint and exclude timepoints where the dilation change is large
def madspeedfilter(tv,dia,is_valid):
    max_gap                                     = 200
    dilation                                    = dia[is_valid]
    cur_tv                                      = tv[is_valid]
    cur_dia_speed                               = np.diff(dilation)/np.diff(cur_tv)
    cur_dia_speed[np.diff(cur_tv)>max_gap]      = np.nan

    back_dilation                               = np.pad(cur_dia_speed,(1,0),constant_values=np.nan)
    fwd_dilation                                = np.pad(cur_dia_speed,(0,1),constant_values=np.nan)
    back_fwd_dilation                           = np.vstack([back_dilation,fwd_dilation])

    max_dilation_speed                          = np.empty_like(dia)
    max_dilation_speed[is_valid]                = np.nanmax(np.abs(back_fwd_dilation),axis=0)
    max_dilation_speed

    mad                                         = np.nanmedian(np.abs(max_dilation_speed-np.nanmedian(max_dilation_speed)))
    mad_multiplier                              = 16 # as defined in Kret et al., 2019
    if mad == 0: 
        print('mad is 0, using dilation speed plus constant as threshold')
        threshold                               = np.nanmedian(max_dilation_speed)+mad_multiplier
    else:
        threshold                               = np.nanmedian(max_dilation_speed)+mad_multiplier*mad
    print('threshold: ' + str(threshold))
    

    valid_out                                   = is_valid.copy()

    valid_out[max_dilation_speed>=threshold]    = False
    valid_out                                   = remove_loners(valid_out.astype(bool),et_refreshrate)
    valid_out                                   = expand_gap(tv,valid_out)
    valid_out                                   = remove_loners(valid_out.astype(bool),et_refreshrate)

    return valid_out.astype(bool)

# Step 3: Fitting a smooth line and exclude samples that deviate from that fitted line
def mad_deviation(tv,dia,is_valid):
    n_passes                                    = 4
    mad_multiplier                              = 16
    interp_fs                                   = 100
    lowpass_cf                                  = 16
    [smooth_filt_b,smooth_filt_a]               = butter(1,lowpass_cf/(interp_fs/2))
    t_interp                                    = np.arange(tv[0],tv[-1],1000/lowpass_cf)
    dia[~is_valid]                              = np.nan
    is_valid_running                            = is_valid.copy()
    residuals_per_pass                          = np.empty([len(is_valid),n_passes])
    smooth_baseline_per_pass                    = np.empty([len(is_valid),n_passes])

    is_done                                     = False
    for pass_id in range(n_passes):
        if is_done: 
            break
        is_valid_start                          = is_valid_running.copy()

        residuals_per_pass[:,pass_id], smooth_baseline_per_pass[:,pass_id] = deviation_calculator(tv,dia,[x and y for x, y in zip(is_valid_running, is_valid)],t_interp,smooth_filt_a,smooth_filt_b)

        mad                                     = np.nanmedian(np.abs(residuals_per_pass[:,pass_id]-np.nanmedian(residuals_per_pass[:,pass_id])))
        threshold                               = np.nanmedian(residuals_per_pass[:,pass_id])+mad_multiplier*mad

        is_valid_running                        = [x and y for x,y in zip((residuals_per_pass[:,pass_id] <= threshold), is_valid)]
        is_valid_running                        = remove_loners(np.array(is_valid_running).astype(bool),et_refreshrate)
        is_valid_running                        = expand_gap(tv,np.array(is_valid_running).astype(bool))
        
        if (pass_id>0 and np.all(is_valid_start==is_valid_running)):
            is_done                             = True
    valid_out                                   = is_valid_running
    return valid_out.astype(bool)


# This is the last step of the preocessing, all invalid samples are removed and the data is detrended
def remove_invalid_detrend(eyes_in,is_valid,isdetrend):
    all_tp                                          = np.arange(len(eyes_in))
    eyes_in[~is_valid]                              = np.nan
    if isdetrend:
        m, b, _, _, _                                   = stats.linregress(all_tp[is_valid],eyes_in[is_valid])
        eyes_in                                         = eyes_in - (m*all_tp + b)
    return eyes_in


#*****************************#
### COMMAND LINE INPUTS ###
#*****************************#
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-bids_dir",
        required=True,
        help='path to bids root',
    )

    args = parser.parse_args()
    
    bids_dir                    = args.bids_dir
    preprocdir                  = f'{bids_dir}/derivatives/preprocessed/'
    figdir                      = f'{bids_dir}/derivatives/figures/'
    resdir                      = f'{bids_dir}/derivatives/output/'

    if not os.path.exists(preprocdir):
        os.makedirs(preprocdir)
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    ####### Run preprocessing ########
    for p in range(1,n_participants+1):
        sa = pd.read_csv(f'{bids_dir}/sourcedata/sample_attributes_P{str(p)}.csv')
        sessions = [0]*n_sessions

        for s in range(n_sessions):
            for r in range(n_runs):
                # load raw eye-tracking data from the MEG
                raw_eyes = load_raw_data(rootdir=bids_dir,p=p,s=s,r=r,trigger_channel=trigger_channel,pd_channel=pd_channel,eye_channel=eye_channel,iseyes=True)

                # now we are getting the onsets from the photodiode channel
                raw_photodiode = load_raw_data(bids_dir,p,s,r,trigger_channel,pd_channel,eye_channel,False)
                photo_d = np.where(np.diff([0]+raw_photodiode._data[0])>1.5)
                pd_new= photo_d[0][np.where(np.diff([0]+photo_d[0])>1000)]

                # cut raw-eyes so that you don't keep all the data after the end of the run
                raw_eyes_cut = raw_eyes.copy()
                start = pd_new[0]*1/meg_refreshrate-0.2
                end = pd_new[-1]*1/meg_refreshrate+2
                raw_eyes_cut.crop(tmin=start, tmax=end, include_tmax=True)
                pd_new= pd_new-raw_eyes_cut.first_samp

                # transform MNE-struct to pandas and change from volts to degrees (x,y) and area (pupil)
                eyes = raw2df(raw_eyes_cut,minvoltage,maxvoltage,minrange,maxrange,screenbottom,screenleft,screenright,screentop,screensize_pix)

                # Define parameters
                tv=(eyes.index.to_numpy()*1/meg_refreshrate)*1000
                dia = eyes['pupil'].copy().to_numpy()

                # PREPROCESSING
                # Step 1: remove out of bounds
                isvalid1 = remove_invalid_samples(eyes,tv)

                # Step 2: speed dilation exclusion
                isvalid2 = madspeedfilter(tv,dia,isvalid1)

                # Step 3: deviation from smooth line
                isvalid3 = mad_deviation(tv,dia,isvalid2)

                # remove invalid and detrend
                eyes_preproc_meg = eyes.copy()
                eyes_preproc_meg['x'] = remove_invalid_detrend(eyes_preproc_meg['x'].to_numpy(),isvalid3,True)
                eyes_preproc_meg['x'] = [pix_to_deg(i,screensize_pix,screenwidth_cm,screendistance_cm) for i in eyes_preproc_meg['x']]

                eyes_preproc_meg['y'] = remove_invalid_detrend(eyes_preproc_meg['y'].to_numpy(),isvalid3,True)
                eyes_preproc_meg['y'] = [pix_to_deg(i,screensize_pix,screenwidth_cm,screendistance_cm) for i in eyes_preproc_meg['y']]

                eyes_preproc_meg['pupil'] = remove_invalid_detrend(eyes_preproc_meg['pupil'].to_numpy(),isvalid3,True)

                # Replace data with preprocessed data
                preprocessed_eyes = raw_eyes.copy()
                preprocessed_eyes._data = eyes_preproc_meg.loc[:,['x','y','pupil']].to_numpy().T


                # make epochs based on photodiode
                event_dict = {'onset_pd':4}
                ev_pd = np.empty(shape=(len(pd_new),3),dtype=int)
                for i,ev in enumerate(pd_new):
                    ev_pd[i]=([int(ev),0,4])

                if r == 0:
                    epochs = mne.Epochs(preprocessed_eyes,ev_pd,event_id = event_dict, tmin = -0.1, tmax = 1.3, baseline=None,preload=False)
                if r > 0:
                    epochs_1 = mne.Epochs(preprocessed_eyes,ev_pd,event_id = event_dict, tmin = -0.1, tmax = 1.3, baseline=None,preload=False)
                    epochs_1.info['dev_head_t'] = epochs.info['dev_head_t']
                    epochs = mne.concatenate_epochs([epochs,epochs_1])


            # add metadata and get rid of catch trials
            epochs.metadata = sa.loc[sa.session_nr==s+1,:]
            epochs = epochs[(epochs.metadata['trial_type']!='catch')]
            # save as dataframe
            tmp = pd.DataFrame(np.repeat(epochs.metadata.values,len(epochs.times), axis=0))
            tmp.columns = epochs.metadata.columns
            tosave = pd.concat([epochs.to_data_frame(),tmp],axis=1)
            tosave.to_csv(preprocdir + '/eyes_epoched_cleaned_P' + str(p) + '_S' + str(s+1) + '.csv')
