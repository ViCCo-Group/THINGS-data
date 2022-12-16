#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -bids_dir

    OUTPUTS:
    Plots the ERFs of the repeat trials.

    NOTES:
    If it doesn't exist, the script makes a figures folder in the BIDS derivatives folder
  
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne, os
import pandas as pd 
import seaborn as sns


#*****************************#
### PARAMETERS ###
#*****************************#
n_participants              = 4
n_sessions                  = 12
n_images                    = 200
channel_picks               = ['O','T','P']
title_names                 = ['Occipital','Temporal','Parietal']
colors                      = ['mediumseagreen','steelblue','goldenrod','indianred','grey']
plt.rcParams['font.size']   = '16'
plt.rcParams['font.family'] = 'Helvetica'

#*****************************#
### HELPER FUNCTIONS ###
#*****************************#
def load_epochs(preproc_dir,all_epochs = []):
    for p in range(1,n_participants+1):
        epochs = mne.read_epochs(f'{preproc_dir}/preprocessed_P{str(p)}-epo.fif', preload=False)
        all_epochs.append(epochs)
    return all_epochs

# helper function 
def plot_erfs(epochs,n_sessions,name,color,ax,ax2,lab):
    ctf_layout = mne.find_layout(epochs.info)
    picks_epochs = [epochs.ch_names[i] for i in np.where([s[2]==name for s in epochs.ch_names])[0]]
    picks = np.where([i[2]==name for i in ctf_layout.names])[0]

    # get evoked data
    for s in range(n_sessions):    
        evoked = epochs[(epochs.metadata['trial_type']=='test') & (epochs.metadata['session_nr']==s+1)].average()
        evoked.pick_channels(ch_names=picks_epochs)
        ax.plot(epochs.times*1000,np.mean(evoked.data.T,axis=1),color=color,lw=0.5,alpha=0.4)
    evoked = epochs[(epochs.metadata['trial_type']=='test')].average()
    evoked.pick_channels(ch_names=picks_epochs)

    # plot ERFs for selected sensor group
    ax.plot(epochs.times*1000,np.mean(evoked.data.T,axis=1),color=color,lw=1,label=lab)
    ax.set_xlim([epochs.times[0]*1000,epochs.times[len(epochs.times)-1]*1000])
    ax.set_ylim([-0.6,0.6])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #  plot sensor locations
    ax2.plot(ctf_layout.pos[:,0],ctf_layout.pos[:,1],color='gainsboro',marker='.',linestyle='',markersize=5)
    ax2.plot(ctf_layout.pos[picks,0],ctf_layout.pos[picks,1],color='grey',marker='.',linestyle='',markersize=5)
    ax2.set_aspect('equal')
    plt.axis('off')

#  Make the ERF plot
def make_figure(all_epochs,fig_dir):
    fig = plt.figure(num=1,tight_layout=True,figsize = (11,6))
    gs = GridSpec(3, 5, figure=fig)
    for i,ch in enumerate(channel_picks):
        for p in range(n_participants):
            ax = fig.add_subplot(gs[i, p])
            if i == 0:
                ax.set_title('M' + str(p+1))
            if i == 2:
                ax.set_xlabel('time (ms)')
            else: 
                plt.setp(ax.get_xticklabels(), visible=False)
            if p == 0:
                ax.set_ylabel('fT')
            else: 
                plt.setp(ax.get_yticklabels(), visible=False)

            ax2=fig.add_subplot(gs[i, -1])
        
            plot_erfs(all_epochs[p],12,ch,colors[p],ax,ax2,'Sub' + str(p+1))
        ax2.set_title(title_names[i])
    plt.savefig(f'{fig_dir}/data_quality-ERFs.pdf',dpi=1000)


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
    preproc_dir                 = f'{bids_dir}/derivatives/preprocessed/'
    sourcedata_dir              = f'{bids_dir}/sourcedata/'
    fig_dir                      = f'{bids_dir}/derivatives/figures/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    ####### Run ########
    all_epochs = load_epochs(preproc_dir)
    make_figure(all_epochs,fig_dir)
