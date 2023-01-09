#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -bids_dir

    OUTPUTS:
    Plots for head motion data.
    If it doesn't exist, the script makes a figures folder in the BIDS derivatives folder
    

    NOTES: 
    This script is using the pyctf toolbox to extract electrode positions from the three sensors (nasion, left, right) that were mounted to the participants head.
    (the pyctf toolbox can be downloaded from the nih-megcore github: https://github.com/nih-megcore/pyctf)
    
    Once the electrode positions are extracted over time, we calculated the distance between the measured head positions.  We then report the average distance within the session and across the session. 
    Note that extra care should be taken when the head coil measurements are being used for source localization: it seems that for participant #4 there are a few sessions where the position recording mal-functioned or where the coil wasn't attached to the same position.  


"""

from pyctf import dsopen
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import itertools, os, sys
import seaborn as sns

#*****************************#
### PARAMETERS ###
#*****************************#
n_participants              = 4
n_runs                      = 10
n_sessions                  = 12
colors                      = ['mediumseagreen','steelblue','goldenrod','indianred','grey']
electrodes                  = ['nas','lpa','rpa']
electrode_labels            = ['Nasion','LPA','RPA']
ppt_labels                  = ['M1','M2','M3','M4']
plt.rcParams['font.size']   = '16'
plt.rcParams['font.family'] = 'Helvetica'


#*****************************#
### HELPER FUNCTIONS ###
#*****************************#

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def load_sensor_positions(rootdir, recording_dir = ['x','y','z'], n_participants = 4, n_session = 12, n_runs = 10):
    # un-usable recordings of head coil position based on notes by experimenter (each row is a participant, each tuple is (session, run))
    invalid_measures = [[],
        [(8,3)],
        [],
        [(4,4),(5,2),(7,7),(7,8),(7,9),(12,5),(12,10)]]

    # initialize data frame
    df                  = pd.DataFrame(columns=['participant','session','run']+[i + '_' + ii for i in electrodes for ii in recording_dir])
    df.participant      = np.repeat(np.arange(1,n_participants+1),n_sessions*n_runs)
    df.session          = np.tile(np.repeat(np.arange(1,n_sessions+1),n_runs),n_participants)
    df.run              = np.tile(np.arange(1,n_runs+1),n_sessions*n_participants)

    for p in range(1,n_participants+1):
        for s in range(1,n_sessions+1):
            for r in range(1,n_runs+1):
                meg_fn = f'{rootdir}/sub-BIGMEG{str(p)}/ses-{str(s).zfill(2)}/meg/sub-BIGMEG{str(p)}_ses-{str(s).zfill(2)}_task-main_run-{str(r).zfill(2)}_meg.ds'
                for i,v in enumerate(electrodes):
                    filter_col = [col for col in df if col.startswith(v)]
                    df.loc[(df.participant==p)&(df.session==s)&(df.run==r),filter_col]  = dsopen(meg_fn).dewar[i]

        # deleting all entries from invalid measurements
        for ii in invalid_measures[p-1]:
            for i,v in enumerate(electrodes):
                filter_col = [col for col in df if col.startswith(v)]
                df.loc[(df['participant']==p) & (df['session']==ii[0]) & (df['run']==ii[1]),filter_col] = np.nan

    return df


def plot_rdm(res,ax,cbar, fig):
    im                  = ax.imshow(res,interpolation='none', cmap='flare',aspect='equal',vmin=0,vmax=5)
    major_ticks         = np.arange(-.5, len(res)-1, 10)
    minor_ticks         = np.arange(-.5, len(res)-1)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 0)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor = True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor = True)

    ax.set_xticklabels(['S'+ str(i) for i in range(1,n_sessions+1)],rotation=45)
    ax.set_yticklabels(['S'+ str(i) for i in range(1,n_sessions+1)])
    ax.grid(which = 'major', alpha = 0.9, color='w')
    ax.grid(which = 'minor', alpha = 0.2, color='w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if cbar: 
        cbar_ax         = fig.add_axes([0.92, 0.4, 0.01, 0.2])
        cbar            = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Distance (mm)', fontsize=16)


def make_supplementary_plot(df):
    fig, ax             = plt.subplots(n_participants,len(electrodes),figsize=(10,20))
    for p in range(1,n_participants+1):
        tmp             = df.loc[df.participant==p,:].copy()
        tmp['id']       = ['S' + str(tmp.session.to_list()[i]) + '_' + 'R' + str(tmp.run.to_list()[i]) for i in range(len(tmp))]
        # get all possible pairwise comparisons
        combs           = list(itertools.combinations(tmp['id'].to_list(), 2))
        res             = np.zeros((len(tmp),len(tmp)))
        res[res == 0.0] = np.nan
        res             = pd.DataFrame(res,columns = tmp['id'].to_numpy(),index = tmp['id'].to_numpy())
        
        # loop over electrodes and calculate distances and plot
        for i,v in enumerate(electrodes):
            filter_col  = [col for col in df if col.startswith(v)]
            res1        = res.copy()
            for vv in combs:
                res1.loc[vv[1],vv[0]]=np.sqrt(((tmp.loc[tmp.id==vv[0],filter_col].to_numpy()-tmp.loc[tmp.id==vv[1],filter_col].to_numpy())**2).sum())

            if (p==4) & (i==2):
                plot_rdm(res1,ax[p-1][i],True,fig) # plot with colorbar
            else:
                plot_rdm(res1,ax[p-1][i],False,fig) # plot without colorbar

    # label the rows and columns
    for a, col in zip(ax[0], electrode_labels):
        a.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for a, row in zip(ax[:,0], ppt_labels):
        a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0),
                    xycoords=a.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.subplots_adjust(left=0.15, top=0.95,right=0.9,hspace=0)

    # save
    fig.savefig(figdir + '/supplementary_motion.pdf')

    return res, res1

def make_boxplot(df,res, res1):
    # initialize
    cross_all                       = np.zeros((n_sessions*n_runs*n_sessions*n_runs,n_participants))
    within_all                      = np.zeros((n_sessions*n_runs*n_sessions*n_runs,n_participants))

    # make masks to filter the distance matrix to extract within and cross-session differences

    within_mask                     = ~res.copy().isna()
    cross_mask                      = ~res.copy().isna()
    for s in range(1,n_sessions+1):
        filter_row                  = [col for col in res1 if col.startswith('S'+str(s)+'_')]
        within_mask.loc[filter_row,filter_row] = True

        filter_col                  = [col for col in res1 if not col.startswith('S'+str(s)+'_')]
        cross_mask.loc[filter_row,filter_col] = True
        cross_mask.loc[filter_col,filter_row] = True

    # loop over participants and calculate the distances for cross- and within-session comparisons
    for p in range(1,n_participants+1):
        tmp                         = df.loc[df.participant==p,:].copy()
        # average three sensors to find midpoint
        for i,v in enumerate(['x','y','z']):
            filter_col              = [col for col in tmp if col.endswith(v)]
            tmp[v]                  = tmp[filter_col].mean(axis=1)
        # label the sessions/runs
        tmp['id']                   = ['S' + str(tmp.session.to_list()[i]) + '_' + 'R' + str(tmp.run.to_list()[i]) for i in range(len(tmp))]
        # find all combinations between all measurement pairs and make a matrix that has all pairwise distances
        combs                       = list(itertools.combinations(tmp['id'].to_list(), 2))
        res                         = np.zeros((len(tmp),len(tmp)))
        res[res == 0.0]             = np.nan
        res                         = pd.DataFrame(res,columns = tmp['id'].to_numpy(),index = tmp['id'].to_numpy())
        res1                        = res.copy()
        for vv in combs:
            res1.loc[vv[1],vv[0]]   = np.sqrt(((tmp.loc[tmp.id==vv[0],['x','y','z']].to_numpy()-tmp.loc[tmp.id==vv[1],['x','y','z']].to_numpy())**2).sum())

        # use the mask to extract the cross-session and within-session distances
        cross_all[:,p-1]            = (res1[cross_mask].to_numpy()*10).ravel()  
        within_all[:,p-1]           = (res1[within_mask].to_numpy()*10).ravel()

    # make the boxplot with cross- and within-session distances
    fig, ax                 = plt.subplots(1,1)
    for p in range(n_participants):
            x               = within_all[:,p]
            boxplot         = ax.boxplot(x[~np.isnan(x)],sym='',whis=(0,90),notch=True,patch_artist=True,widths=0.25,positions = [p-0.15],
                                    boxprops=dict(facecolor=(colors[p]), color='k'),
                                    medianprops=dict(color='k',lw=1))

            x               = cross_all[:,p]
            boxplot         = ax.boxplot(x[~np.isnan(x)],sym='',whis=(0,90),notch=True,patch_artist=True,widths=0.25,positions = [p+0.15],
                                    boxprops=dict(facecolor=lighten_color(colors[p],amount=0.3), color='k'),
                                    medianprops=dict(color='k',lw=1))

    # make plot look pretty
    ax.set_xticks(np.arange(n_participants))
    ax.set_xticklabels(['M' + str(p+1) for p in np.arange(n_participants)])

    caps                    = boxplot['caps']
    med                     = boxplot['medians'][0]
    xpos                    = med.get_xdata()
    xoff                    = 0.10 * (xpos[1] - xpos[0])
    xlabel                  = xpos[1] + xoff
    capbottom               = caps[0].get_ydata()[0]
    captop                  = caps[1].get_ydata()[0]

    ax.text(xlabel, capbottom,
            '5th percentile', va='center')
    ax.text(xlabel, captop,
            '95th percentile', va='center')

    ax.set_ylabel('Head coil movement (mm)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    within_patch            = Patch(facecolor=[0.5,0.5,0.5])
    cross_patch             = Patch(facecolor=lighten_color([0.5,0.5,0.5],amount=0.3))

    ax.legend([within_patch,cross_patch],['within-session ','cross-session'],frameon=False,loc='upper left')
    ax.set_ylim([-0.1,10])

    # save
    fig.subplots_adjust(right=0.8)
    fig.savefig(figdir+'/data_quality-motion-box.pdf')


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
    rootdir                     = args.bids_dir
    sourcedata_dir              = f'{rootdir}/sourcedata/'

    figdir                      = f'{rootdir}/derivatives/figures/'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    ####### Run ########
    df = load_sensor_positions(rootdir, recording_dir = ['x','y','z'], n_participants = 4, n_session = 12, n_runs = 10)
    res, res1 = make_supplementary_plot(df)
    make_boxplot(df,res, res1)
