#!/usr/bin/env python3

"""
@ Lina Teichmann

    INPUTS: 
    call from command line with following inputs: 
        -bids_dir

    OUTPUTS:
    Runs a linear regression, using the MEG data at every timepoint to predict animacy and size ratings for each image. 

    NOTES:
    The plot was made in matlab so it looks the same as the decoding plots (see Step3aa)
    If the output directory does not exist, this script makes an output folder in BIDS/derivatives
  
"""

import numpy as np
import mne,os,itertools,sys
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

#*****************************#
### PARAMETERS ###
#*****************************#
n_participants              = 4
n_sessions                  = 12

#*****************************#
### HELPER FUNCTIONS ###
#*****************************#
class load_data: 
    def __init__(self,dat,sourcedata_dir,trial_type='exp'):
        self.dat = dat
        self.trial_type = trial_type
        self.sourcedata_dir = sourcedata_dir

    def load_animacy_size(self):
        ani_csv = f'{self.sourcedata_dir}/ratings_animacy.csv'
        size_csv = f'{self.sourcedata_dir}/ratings_size.csv'
        # load with pandas
        ani_df = pd.read_csv(ani_csv)[['uniqueID', 'lives_mean']]
        ani_df = ani_df.rename(columns={'lives_mean':'animacy'})
        size_df = pd.read_csv(size_csv, sep=';')[['uniqueID', 'meanSize']]
        size_df = size_df.rename(columns={'meanSize':'size'})
        # ani_df has "_", size_df " " as separator in multi-word concepts
        size_df['uniqueID'] = size_df.uniqueID.str.replace(' ', '_')
        # merge
        anisize_df = pd.merge(left=ani_df, right=size_df, on='uniqueID', how='outer')
        assert anisize_df.shape[0] == ani_df.shape[0] == size_df.shape[0]
        return anisize_df
    
    def load_meg(self):
        #  select exp trails & sort the trials based on things_category_nr
        epochs_exp = self.dat[(self.dat.metadata['trial_type']=='exp')]   
        sort_order = np.argsort(epochs_exp.metadata['things_category_nr'])
        dat_sorted=epochs_exp[sort_order]
        # getting data from each session and load it
        self.n_categories = len(dat_sorted.metadata.things_category_nr.unique())
        self.n_sessions = len(dat_sorted.metadata.session_nr.unique())
        self.n_channels = len(dat_sorted.ch_names)
        self.n_time = len(dat_sorted.times)
        self.sess_data = np.empty([self.n_categories,self.n_channels,self.n_time,self.n_sessions])
        for sess in range(self.n_sessions):
            print('loading data for session ' + str(sess+1))
            curr_data = dat_sorted[dat_sorted.metadata['session_nr']==sess+1]
            curr_data = curr_data.load_data()
            self.sess_data[:,:,:,sess]= curr_data._data
        return self.sess_data

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

            training_y = self.label
            training_y = np.tile(training_y,self.n_sessions-1)

            testing_x=np.take(sess_dat,test_splits[split][0],axis=3)
            testing_y = self.label

            corr_coef_time = Parallel(n_jobs=24)(delayed(fit_predict)(pipe,training_x[:,:,t],training_y,testing_x[:,:,t],testing_y) for t in range(self.n_time))
            corr_coef[:,split] = corr_coef_time
        
        return corr_coef

def run(p,preproc_dir):
    epochs = mne.read_epochs(f'{preproc_dir}/preprocessed_P{str(p)}-epo.fif', preload=False)
    anisize_df = load_data(epochs,sourcedata_dir,'exp').load_animacy_size()
    data = load_data(epochs,sourcedata_dir,'exp').load_meg()
    animacy_corr_coeff = linear_regression(data,anisize_df['animacy'].to_numpy()).run()
    size_corr_coeff = linear_regression(data,anisize_df['size'].to_numpy()).run()

    pd.DataFrame(animacy_corr_coeff).to_csv(f'{res_dir}/validation-animacy-P{str(p)}.csv')
    pd.DataFrame(size_corr_coeff).to_csv(f'{res_dir}/validation-size-P{str(p)}.csv')


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
    res_dir                      = f'{bids_dir}/derivatives/output/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

     ####### Run analysis ########
    for p in range(1,n_participants+1):
        run(p,preproc_dir)
