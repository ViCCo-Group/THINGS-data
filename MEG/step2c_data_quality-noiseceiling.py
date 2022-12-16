#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -bids_dir

    OUTPUTS:
    Calculates and plots noise ceilings based on the 200 repeat images for all sensors and each sensor group

    NOTES:
    If it doesn't exist, the script makes a figures folder in the BIDS derivatives folder
  
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne,os
from scipy.stats import zscore

#*****************************#
### PARAMETERS ###
#*****************************#
n_participants              = 4
n_sessions                  = 12
n_images                    = 200
names                       = ['O','T','P','F','C']
labs                        = ['Occipital','Temporal','Parietal','Frontal','Central']
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

def kknc(data: np.ndarray, n: int or None = None):
    """
    Calculate the noise ceiling reported in the NSD paper (Allen et al., 2021)
    Arguments:
        data: np.ndarray
            Should be shape (ntargets, nrepetitions, nobservations)
        n: int or None
            Number of trials averaged to calculate the noise ceiling. If None, n will be the number of repetitions.
    returns:
        nc: np.ndarray of shape (ntargets)
            Noise ceiling without considering trial averaging.
        ncav: np.ndarray of shape (ntargets)
            Noise ceiling considering all trials were averaged.
    """
    if not n:
        n = data.shape[-2]
    normalized = zscore(data, axis=-1)
    noisesd = np.sqrt(np.mean(np.var(normalized, axis=-2, ddof=1), axis=-1))
    sigsd = np.sqrt(np.clip(1 - noisesd ** 2, 0., None))
    ncsnr = sigsd / noisesd
    nc = 100 * ((ncsnr ** 2) / ((ncsnr ** 2) + (1 / n)))
    return nc

def calculate_noise_ceiling(all_epochs,all_nc = []):
    n_time = len(all_epochs[0].times)
    for p in range(n_participants):
        n_channels = len(all_epochs[p].ch_names)

        #  load data
        epochs = all_epochs[p]
        # select repetition trials only and load epoched data into memory
        epochs_rep = epochs[(epochs.metadata['trial_type']=='test')]
        epochs_rep.load_data()

        # select session data and sort based on category number
        res_mat=np.empty([n_channels,n_sessions,n_images,n_time])
        for sess in range(n_sessions):
            epochs_curr = epochs_rep[epochs_rep.metadata['session_nr']==sess+1]
            sort_order = np.argsort(epochs_curr.metadata['things_category_nr'])
            epochs_curr=epochs_curr[sort_order]
            epochs_curr = np.transpose(epochs_curr._data, (1,0,2))

            res_mat[:,sess,:,:] = epochs_curr

        # run noise ceiling
        nc = np.empty([n_channels,n_time])
        for t in range(n_time):
            dat = res_mat[:,:,:,t]
            nc[:,t] = kknc(data=dat,n=n_sessions)
        all_nc.append(nc)
    return all_nc

def make_supplementary_plot(all_epochs,fig_dir):
    plt.close('all')
    fig = plt.figure(num=2,figsize = (12,8))
    gs1 = gridspec.GridSpec(n_participants+1, len(names))
    gs1.update(wspace=0.2, hspace=0.2)
    ctf_layout = mne.find_layout(all_epochs[1].info)
    counter = 0
    for i,n in enumerate(names):
        for p in range(n_participants):
            ax = fig.add_subplot(gs1[counter])
            counter+=1
            ax.clear()
            picks_epochs = np.where([s[2]==n for s in all_epochs[p].ch_names])[0]
            picks = np.where([i[2]==n for i in ctf_layout.names])[0]
            [ax.plot(all_epochs[p].times*1000,ii,color=colors[p],label=labs[i],lw=0.1,alpha=0.2) for ii in all_nc[p][picks_epochs,:]]
            ax.plot(all_epochs[p].times*1000,np.mean(all_nc[p][picks_epochs,:],axis=0),color=colors[p],label=labs[i],lw=1.5)
            ax.set_ylim([0,100])

            if i ==0:
                ax.set_title('M' + str(p+1))
                
            if i < len(names)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel('time (ms)')

            if p == 0:
                plt.setp(ax.get_yticklabels(), visible=True)
                ax.set_ylabel(labs[i])
            else: 
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if i == 2 and p == 0:
                ax.set_ylabel('Explained Variance (%)\n' + labs[i])

        #  plot sensor locations
        ax2 = fig.add_subplot(gs1[counter])
        counter+=1
        ax2.plot(ctf_layout.pos[:,0],ctf_layout.pos[:,1],color='gainsboro',marker='.',linestyle='',markersize=3)
        ax2.plot(ctf_layout.pos[picks,0],ctf_layout.pos[picks,1],color='grey',marker='.',linestyle='',markersize=3)
        ax2.axis('equal')
        ax2.axis('off')
    fig.savefig(f'{fig_dir}/data_quality-noiseceiling_all.pdf')

def make_main_plot(all_epochs,all_nc):
    plt.close('all')
    fig = plt.figure(num=1,figsize = (12,3))
    gs1 = gridspec.GridSpec(1,len(names),wspace=0.1,)
    ctf_layout = mne.find_layout(all_epochs[1].info)

    for i,n in enumerate(names):
        ax = fig.add_subplot(gs1[i])

        # plot niose ceilings
        picks_epochs = [np.where([s[2]==n for s in all_epochs[p].ch_names])[0] for p in range(n_participants)]
        picks = np.where([i[2]==n for i in ctf_layout.names])[0]
        [ax.plot(all_epochs[p].times*1000,np.mean(all_nc[p][picks_epochs[p],:],axis=0),color=colors[p],label='M'+str(p+1),lw=2) for p in range(n_participants)]

        ax.set_ylim([0,90])
        ax.set_xlim([all_epochs[1].times[0]*1000,all_epochs[1].times[len(all_epochs[1].times)-1]*1000])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i == len(names)-1:
            plt.legend(frameon=False, bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('time (ms)')
        if  i ==0: 
            ax.set_ylabel('Explained variance (%)')
        else: 
            plt.setp(ax.get_yticklabels(), visible=False)

        #  plot sensor locations
        ax2 = ax.inset_axes([0.55, 0.55, 0.5, 0.5])
        ax2.plot(ctf_layout.pos[:,0],ctf_layout.pos[:,1],color='darkgrey',marker='.',linestyle='',markersize=2)
        ax2.plot(ctf_layout.pos[picks,0],ctf_layout.pos[picks,1],color='k',marker='.',linestyle='',markersize=2)
        ax2.axis('equal')
        ax2.axis('off')
        ax2.set_title(labs[i],y=0.8,fontsize=14)

    fig.savefig(f'{fig_dir}/data_quality-noiseceiling_avgd.pdf')



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
    all_nc = calculate_noise_ceiling(all_epochs)
    make_supplementary_plot(all_epochs,fig_dir)
    make_main_plot(all_epochs,all_nc)

