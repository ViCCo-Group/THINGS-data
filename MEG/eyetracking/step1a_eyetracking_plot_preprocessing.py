#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -bids_dir

    OUTPUTS:
    makes an overview plot of the eyetracking preprocessing steps 

"""
# step1_eyetracking_preprocess.py>
from step1_eyetracking_preprocess import *
from matplotlib import gridspec
import pandas as pd 
import numpy as np 
import mne 

def make_overview_plot(bids_dir, figdir,p,s,r):
    sa = pd.read_csv(f'{bids_dir}/sourcedata/sample_attributes_P{str(p)}.csv')
    raw_eyes = load_raw_data(rootdir=bids_dir,p=p,s=s,r=r,trigger_channel=trigger_channel,pd_channel=pd_channel,eye_channel=eye_channel,iseyes=True)

    # now we are getting the onsets from the photodiode channel
    raw_photodiode = load_raw_data(bids_dir,p,s,r,trigger_channel,pd_channel,eye_channel,False)
    photo_d = np.where(np.diff([0]+raw_photodiode._data[0])>1.5)
    pd_new= photo_d[0][np.where(np.diff([0]+photo_d[0])>1000)]

    # cut raw-eyes so that you don't keep all the data after the end of the run
    raw_eyes_cut = raw_eyes.copy()
    start = pd_new[0]*1/1200-0.2
    end = pd_new[-1]*1/1200+2
    raw_eyes_cut.crop(tmin=start, tmax=end, include_tmax=True)
    pd_new= pd_new-raw_eyes_cut.first_samp

    # transform MNE-struct to pandas and change from volts to degrees (x,y) and area (pupil)
    eyes = raw2df(raw_eyes_cut,minvoltage,maxvoltage,minrange,maxrange,screenbottom,screenleft,screenright,screentop,screensize_pix)

    # Define parameters
    tv=(eyes.index.to_numpy()*1/1200)*1000
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

    epochs = mne.Epochs(preprocessed_eyes,ev_pd,event_id = event_dict, tmin = -0.1, tmax = 1.3, baseline=None,preload=False)

    epochs.metadata = sa.loc[(sa.session_nr==s+1)&(sa.run_nr==r+1),:]
    epochs = epochs[(epochs.metadata['trial_type']!='catch')]

    # save as dataframe
    tmp = pd.DataFrame(np.repeat(epochs.metadata.values,len(epochs.times), axis=0))
    tmp.columns = epochs.metadata.columns
    tosave = pd.concat([epochs.to_data_frame(),tmp],axis=1)

    ### FIGURE 1 #####
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    spec = gridspec.GridSpec(ncols=2, nrows=5,width_ratios=[3, 1], wspace=0.1,hspace=0.7)

    def plot_run(toplot,ax,ylabel,xlabel,title,n_samples,is_preprocessed):
        print(is_preprocessed)
        if is_preprocessed==0:
            toplot = [pix_to_deg(i,screensize_pix,screenwidth_cm,screendistance_cm) for i in toplot]

        ax.plot(np.take(toplot,np.arange(n_samples)),'grey')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title)
        ax.set_ylim([-3,3])


    selection = np.repeat([None, isvalid1, isvalid2, isvalid3],2)
    titles = np.repeat(['raw','step1: invalid samples exclusion', 'step2: dilation speed exclusion', 'step3: deviation exclusion'],2)
    for i,v in enumerate(range(len(titles))):
        print(i,v)
        ax = fig.add_subplot(spec[v])
        tmp = eyes.copy()
        if (selection[i] is not None):
            empt = np.ones(len(eyes))
            empt[selection[i]] = 0
            tmp.loc[empt.astype(bool),:] = np.nan
        
        if i % 2 == 0:
            plot_run(tmp.y,ax,'y (\N{DEGREE SIGN})','',titles[i],len(tmp),False)
        else: 
            plot_run(tmp.y,ax,'y (\N{DEGREE SIGN})','',titles[i],20000,False)
            ax.axes.yaxis.set_visible(False)
        ax.axes.xaxis.set_visible(False)


    ax = fig.add_subplot(spec[8])
    plot_run(eyes_preproc_meg['y'],ax,'y (\N{DEGREE SIGN})','samples (1200 Hz)','step4: linear detrending',len(eyes_preproc_meg),True)

    ax = fig.add_subplot(spec[9])
    plot_run(eyes_preproc_meg['y'],ax,'y (\N{DEGREE SIGN})','samples (1200 Hz)','step4: linear detrending',20000,True)
    ax.axes.yaxis.set_visible(False)

    fig.savefig(f'{figdir}/ET_preprocess_overview.png',dpi=600)




    ### FIGURE 2 #####
    # only plot pre-processing and post-processsing example 
    eyes = raw2df(raw_eyes_cut,minvoltage,maxvoltage,minrange,maxrange,screenbottom,screenleft,screenright,screentop,screensize_pix)

    fig = plt.figure()
    fig.set_figheight(2)
    fig.set_figwidth(5)
    spec = gridspec.GridSpec(ncols=2, nrows=2,width_ratios=[1, 1], wspace=0.1,hspace=0.25,bottom=0.22,left=0.14)

    samples  = 30000

    def plot_run(toplot,ax,ylabel,xlabel,title,n_samples,is_preprocessed):
        y = np.take(toplot,np.arange(n_samples))
        x = np.arange(len(y))*(1/1200)
        if is_preprocessed==1:
            ax.plot(x,y,'darkgrey',lw=1)
        else:
            y= [pix_to_deg(i,screensize_pix,screenwidth_cm,screendistance_cm) for i in y]
            ax.plot(x,y,'lightgrey',lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title)
    
    ax = fig.add_subplot(spec[0])
    tmp = eyes.copy()
    plot_run(tmp.y,ax,'','','',len(tmp),False)
    ax.axes.xaxis.set_visible(False)
    ax.set_ylim([-10,10])

    ax = fig.add_subplot(spec[1])
    plot_run(tmp.y,ax,'y (\N{DEGREE SIGN})','','',samples,False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.set_ylim([-10,10])

    ax = fig.add_subplot(spec[2])
    plot_run(eyes_preproc_meg['y'],ax,'','time (s)','',len(eyes_preproc_meg),True)
    ax.set_ylim([-1,1])

    ax = fig.add_subplot(spec[3])
    plot_run(eyes_preproc_meg['y'],ax,'y (\N{DEGREE SIGN})','time (s)','',samples,True)
    ax.axes.yaxis.set_visible(False)
    ax.set_ylim([-1,1])

    fig.supylabel('      y (\N{DEGREE SIGN})')

    fig.savefig(f'{figdir}/supplementary_ET_preprocess.png',dpi=600)




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
    figdir                      = f'{bids_dir}/derivatives/figures/'

    if not os.path.exists(figdir):
        os.makedirs(figdir)


    make_overview_plot(bids_dir,figdir,p=1,s=0,r=0)
