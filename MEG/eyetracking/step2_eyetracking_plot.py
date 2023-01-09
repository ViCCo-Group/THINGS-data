#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -bids_dir

    OUTPUTS:
    - plot showing percentage of data we lost after preprocessing
    - plot showing the time-resolved euclidean x/y coordinates from the central fixation
    - plot showing the time-resolved pupil size across all sessions
    - plot showing the gaze position along with a threshold inlet

"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

n_participants              = 4
n_sessions                  = 12
colors                      = ['mediumseagreen','steelblue','goldenrod','indianred']
plt.rcParams['font.size']   = '14'
plt.rcParams['font.family'] = 'Helvetica'

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def load_data(preprocdir):
    invalid_samples = np.zeros([n_participants,n_sessions])
    data = [[[] for i in range(n_sessions)] for j in range(n_participants)]
    for p in range(n_participants):
        print('participant ' + str(p+1))
        for s in range(n_sessions):
            print('loading session ' + str(s+1))
            sess_dat = pd.read_csv(preprocdir + '/eyes_epoched_cleaned_P' + str(p+1) + '_S' + str(s+1) + '.csv')
            sess_dat.rename(columns={"UADC009-2104": "x", "UADC010-2104": "y","UADC013-2104":"pupil"},inplace=True)

            invalid_samples[p,s] = (np.mean([np.isnan(sess_dat.loc[sess_dat.run_nr==i,'x']).sum()/len(sess_dat.loc[sess_dat.run_nr==i,'x']) for i in range(1,11)]))

            tv = np.linspace(-100,1300,len(np.where(sess_dat.image_nr==sess_dat.image_nr.iloc[0])[0]))
            sess_dat['time_samples'] =np.concatenate([np.arange(len(tv))]*sum(sess_dat.time==-100))

            data[p][s]=sess_dat
    
    return data,invalid_samples

        
def make_plots(data,invalid_samples,figdir):
    # plot invalid data
    fig,axs = plt.subplots(ncols=1, nrows=1,num=10,tight_layout=True,figsize = (4,3))

    [plt.plot(np.arange(n_sessions),np.sort(invalid_samples[i])*100,color=colors[i],marker='o',lw=1,label='M'+ str(i+1)) for i in range(n_participants)]
    plt.ylabel('proportion removed (%)')
    plt.legend(loc='upper left',ncol=2,frameon=False)

    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([])
    ax.set_xlabel('sessions (sorted)')

    ax.set_ylim([0,80])
    fig.set_size_inches(5,3)

    fig.savefig(figdir + '/supplementary_ET_invalid.pdf',dpi=600)


    # plot time-resolved euclidean distance of the x/y position from the central fixation
    euc_dist_all,pupil_size_all = [],[]
    for p in range(n_participants):
        curr_dat = pd.concat(data[p],axis=0)
        euc_dist = pd.DataFrame(columns =  np.unique(curr_dat.time_samples))
        pupil_size = pd.DataFrame(columns =  np.unique(curr_dat.time_samples))

        for t in np.unique(curr_dat.time_samples):
            print(t)
            x = curr_dat.loc[curr_dat.time_samples==t].x.to_numpy()
            y = curr_dat.loc[curr_dat.time_samples==t].y.to_numpy()
            euc_dist[t] = np.sqrt((x-0)**2+(y-0)**2)
            euc_dist.reset_index(drop=True,inplace=True)
            pupil_size[t] = curr_dat.loc[curr_dat.time_samples==t].pupil.to_numpy()

        euc_dist_all.append(euc_dist)
        pupil_size_all.append(pupil_size)

    tv = ((np.unique(data[0][0].time_samples.to_numpy())*1/1200)-0.1)*1000

    plt.close('all')
    fig,ax = plt.subplots(ncols=1, nrows=1,num=1,tight_layout=True,figsize = (6,4),sharex=True,sharey=True)

    for p in range(n_participants):
        toplot = euc_dist_all[p]
        baseline = toplot[np.where(tv<=0)[0]].mean(axis=1)
        toplot = toplot.sub(baseline,axis=0)
        ax.plot(tv,np.mean(toplot,axis=0),color=colors[p],label='M'+str(p+1)) 
        ax.fill_between(tv, np.mean(toplot,axis=0)-np.std(toplot,axis=0)/np.sqrt(len(toplot)), np.mean(toplot,axis=0)+np.std(toplot,axis=0)/np.sqrt(len(toplot)),color=colors[p],alpha=0.2,lw=0)
        ax.hlines(0,tv[0],tv[-1],'grey',linestyles='--')

        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Euclidean distance (\N{DEGREE SIGN})\nbaseline corrected')
        ax.legend(frameon=False,ncol=1,loc='upper left',borderpad=0.1,labelspacing=0.1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    fig.set_size_inches(5,3)
    fig.savefig(figdir + '/supplementary_ET_timeresolved.pdf',dpi=500)

    # plot time-resolved pupil size
    tv = ((np.unique(data[0][0].time_samples.to_numpy())*1/1200)-0.1)*1000

    plt.close('all')
    fig,ax = plt.subplots(ncols=1, nrows=1,num=1,tight_layout=True,figsize = (6,4),sharex=True,sharey=True)

    for p in range(n_participants):
        toplot = pupil_size_all[p]
        ax.plot(tv,np.mean(toplot,axis=0),color=colors[p],label='M'+str(p+1)) 
        ax.fill_between(tv, np.mean(toplot,axis=0)-np.std(toplot,axis=0)/np.sqrt(len(toplot)), np.mean(toplot,axis=0)+np.std(toplot,axis=0)/np.sqrt(len(toplot)),color=colors[p],alpha=0.2,lw=0)

        ax.set_xlabel('time (ms)')
        ax.set_ylabel('Pupil size (a.u.)')
        ax.legend(frameon=False,ncol=1,loc='lower left',borderpad=0.1,labelspacing=0.1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig.set_size_inches(5,3)
    fig.savefig(figdir + '/supplementary_ET_pupil_timeresolved.pdf',dpi=500)

    #plot gaze-position with threshold inlets
    # (note that data is downsampled to 100 Hz here because otherwise the plotting takes too long)
    plt.close('all')
    print('making gaze position KDE-plots')
    fig,axs = plt.subplots(ncols=4, nrows=1,num=4,tight_layout=True,figsize = (10,3),sharex=True,sharey=True)
    downsample = 1
    thresholds = np.linspace(0,5,50)
    for p,ax in enumerate(axs.flatten()):
        allxs,allys = [],[]
        for s in range(n_sessions):
            if downsample:
                allxs.extend(data[p][s].x[0::12])
                allys.extend(data[p][s].y[0::12])
            else: 
                allxs.extend(data[p][s].x)
                allys.extend(data[p][s].y)
    
        sns.histplot(x=allxs,y=allys,ax=ax,color=colors[p])
        if downsample:
            sns.kdeplot(x=allxs,y=allys,ax=ax,cmap='Greys',levels=[0.25,0.5,0.75],linewidths=0.3)
        else:
            circle1 = plt.Circle((0, 0), 1, edgecolor='white',fill=False,linestyle='--')
            ax.add_patch(circle1)

        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        ax.set_xlabel('x-gaze (\N{DEGREE SIGN})')
        ax.set_ylabel('y-gaze (\N{DEGREE SIGN})')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal','box')

        # add threshold
        xx = np.ma.array(allxs,mask=np.isnan(allxs))
        yy = np.ma.array(allys,mask=np.isnan(allys))
        outx = [np.mean(xx<=thresh)*100 for thresh in thresholds]
        outy = [np.mean(yy<=thresh)*100 for thresh in thresholds]

        
        # inlet showing % of data below thresholds
        ins = ax.inset_axes([0.85,0.85,0.45,0.45])
        ins.plot(thresholds,outy,color=lighten_color(colors[p],1),label='y',lw=1)
        ins.plot(thresholds,outx,color=lighten_color(colors[p],0.5),label='x',lw=1)
        ins.vlines(1,0,100,'grey',linestyles='--',lw=1)
        ins.hlines(outy[np.argmin(np.abs(1-thresholds))],0,5,color=lighten_color(colors[p],1),linestyles='--',lw=1)
        ins.hlines(outx[np.argmin(np.abs(1-thresholds))],0,5,color=lighten_color(colors[p],0.5),linestyles='--',lw=1)
        ins.set_xlabel('degrees',fontsize=9)
        ins.set_ylabel('Prop. (%)',fontsize=9)
        ins.spines['right'].set_visible(False)
        ins.spines['top'].set_visible(False)
        ins.set_yticks([0,50,100])
        ins.set_yticklabels([0,50,100],fontsize=9)
        ins.set_xticks(np.arange(5))
        ins.set_xticklabels(np.arange(5),fontsize=9)

    fig.set_size_inches(9,5)
    fig.savefig(figdir + '/supplementary_ET_gazepos_withrings.png',dpi=500)



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
    preprocdir                  = f'{bids_dir}/derivatives/preprocessed/'

    if not os.path.exists(figdir):
        os.makedirs(figdir)


    data,invalid_samples = load_data(preprocdir)
    make_plots(data,invalid_samples,figdir)
    
