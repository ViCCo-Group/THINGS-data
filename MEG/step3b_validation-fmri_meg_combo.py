#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -bids_dir

    OUTPUTS:
    Using the MEG data to predict univariate fMRI activation in two ROIs 

    NOTES:
    The plot was made in matlab so it looks the same as the decoding plots (see Step3bb)
    If the output directory does not exist, this script makes an output folder in BIDS/derivatives
  
"""

import numpy as np
import mne,itertools,os, sys
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed



# parameters
n_participants              = 4
n_participants_fmri         = 3
n_sessions                  = 12

#*****************************#
### HELPER FUNCTIONS ###
#*****************************#
def load_epochs(preproc_dir,all_epochs = []):
    for p in range(1,n_participants+1):
        epochs = mne.read_epochs(f'{preproc_dir}/preprocessed_P{str(p)}-epo.fif', preload=False)
        all_epochs.append(epochs)
    return all_epochs


## Helper functions


class load_data: 
    def __init__(self,dat,beta_dir,trial_type='exp'):
        self.dat = dat
        self.trial_type = trial_type
        self.beta_dir = beta_dir
    
    def load_meg(self):
        trialinfo = pd.read_csv(self.beta_dir + '/sub-01/sub-01_trialinfo.csv')
        trialinfo = trialinfo[trialinfo.trial_type=='exp']
        fmri_concepts = np.sort(trialinfo.things_category_nr.unique())

        #  select exp trails & sort the trials based on things_category_nr & image name & select fMRI concepts only
        epochs_exp = self.dat[(self.dat.metadata['trial_type']=='exp')]   
        sort_order = np.argsort(epochs_exp.metadata['image_path'])
        dat_sorted=epochs_exp[sort_order]

        meg_fmriconcepts = dat_sorted[dat_sorted.metadata.things_category_nr.isin(fmri_concepts)]
        meg_fmriconcepts.metadata['cv_split'] = np.tile(np.arange(1,n_sessions+1),720)
        sort_order = np.lexsort((meg_fmriconcepts.metadata['image_path'],meg_fmriconcepts.metadata['cv_split']))
        meg_fmriconcepts = meg_fmriconcepts[sort_order]

        # getting data from each session and load it
        self.n_categories = len(meg_fmriconcepts.metadata.things_category_nr.unique())
        self.n_sessions = len(meg_fmriconcepts.metadata.session_nr.unique())
        self.n_channels = len(meg_fmriconcepts.ch_names)
        self.n_time = len(meg_fmriconcepts.times)
        sess_data = np.empty([self.n_categories,self.n_channels,self.n_time,self.n_sessions])
        for split in range(self.n_sessions):
            print('loading data for cv-split ' + str(split+1))
            curr_data = meg_fmriconcepts[meg_fmriconcepts.metadata['cv_split']==split+1]
            curr_data = curr_data.load_data()
            sess_data[:,:,:,split]= curr_data._data
        return sess_data

    def load_roi_betas(self):
        roi_ffa,roi_v1 = [],[]
        for ppt in range(1,n_participants_fmri+1):
            for roi_name, roi_array in zip(['ffa','v1'],[roi_ffa,roi_v1]):
                trialinfo = pd.read_csv(f'{self.beta_dir}/sub-{str(ppt).zfill(2)}/sub-{str(ppt).zfill(2)}_trialinfo.csv')
                try:
                    roi = np.load(f'{self.beta_dir}/sub-{str(ppt).zfill(2)}/{roi_name}.npy')
                    roi = roi.mean(axis=1)
                except:
                    raise ValueError('This ROI file does not exist.')

                # take only experimental trials
                idx_exp = trialinfo.trial_type=='exp'
                roi_exp = roi[idx_exp]
                trialinfo_exp = trialinfo[idx_exp]
                trialinfo_exp.reset_index(drop=True,inplace=True)

                # sort based on things category
                trialinfo_exp_sorted = trialinfo_exp.sort_values('filename')
                trialinfo_exp_sorted['cv_split'] = np.tile(np.arange(1,n_sessions+1),720)
                trialinfo_exp_sorted=trialinfo_exp_sorted.sort_values(['cv_split','filename'])
                sort_index = trialinfo_exp_sorted.index.to_numpy()
                roi_exp_sorted = roi_exp[sort_index]
                roi_array.append(roi_exp_sorted.reshape([-1,n_sessions],order='F'))

        print('ROI shape: ' + str(np.array(roi_ffa).shape))

        # average across people
        roi_ffa = np.array(roi_ffa).mean(axis=0)
        roi_v1 = np.array(roi_v1).mean(axis=0)

        return roi_ffa,roi_v1


# Cross-validated linear regression
class linear_regression:
    def __init__(self,dat,label):
        self.dat = dat
        self.label = label
        self.n_categories = dat.shape[0]
        self.n_channels = dat.shape[1]
        self.n_time = dat.shape[2]
        self.n_sessions = dat.shape[3]

    def train_test_splits(self):
        self.train_splits,self.test_splits = [],[]
        for comb in itertools.combinations(np.arange(self.n_sessions), self.n_sessions-1):
            self.train_splits.append(comb)
            self.test_splits.append(list(set(np.arange(self.n_sessions)) - set(comb)))
        return self.train_splits,self.test_splits

    def run(self):
        sess_dat = self.dat
        train_splits,test_splits = self.train_test_splits()

        pipe = Pipeline([('scaler', StandardScaler()),
            ('regression', LinearRegression())])

        corr_coef = np.empty([self.n_time,self.n_sessions])

        def fit_predict(pipe,train_x,train_y,test_x,test_y):
            pipe.fit(train_x,train_y)
            y_pred = pipe.predict(test_x)
            return np.corrcoef(y_pred,test_y)[0,1]


        for split in range(self.n_sessions):
            print('cv-split ' + str(split))
            
            training_x = np.take(sess_dat,train_splits[split],axis=3)
            training_x = np.concatenate(tuple(training_x[:,:,:,i] for i in range(training_x.shape[3])),axis=0)

            training_y = np.take(self.label,train_splits[split],axis=1)
            training_y = np.concatenate(tuple(training_y[:,i] for i in range(training_y.shape[1])),axis=0)

            testing_x=np.take(sess_dat,test_splits[split][0],axis=3)
            testing_y = np.take(self.label,test_splits[split][0],axis=1)

            corr_coef_time = Parallel(n_jobs=24)(delayed(fit_predict)(pipe,training_x[:,:,t],training_y,testing_x[:,:,t],testing_y) for t in range(self.n_time))
            corr_coef[:,split] = corr_coef_time
        
        return corr_coef

def run(p,betas_dir,res_dir):
    all_epochs = load_epochs(preproc_dir)
    data = load_data(all_epochs[p-1],betas_dir,'exp').load_meg()
    Y_ffa,Y_v1 = load_data(all_epochs[p-1],betas_dir,trial_type='exp').load_roi_betas()

    corr_coeff_ffa = linear_regression(data,Y_ffa).run()
    corr_coeff_v1 = linear_regression(data,Y_v1).run()

    pd.DataFrame(corr_coeff_ffa).to_csv(f'{res_dir}/validation_fMRI-MEG-regression_ffa_P{str(p)}.csv')
    pd.DataFrame(corr_coeff_v1).to_csv(f'{res_dir}/validation_fMRI-MEG-regression_v1_P{str(p)}.csv')



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
    res_dir                     = f'{bids_dir}/derivatives/output/'
    betas_dir                   = f'{sourcedata_dir}/betas_roi/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    ####### Run analysis ########
    for p in range(1,n_participants+1):
        run(p,betas_dir,res_dir)

