# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:42:05 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import os
import pandas as pd


class cropscan_io(object):
    '''
    Class that provides file management functionality specifically for cropscan
    data. This class loads, filters, and preps cropscan data for use in a
    supervised regression model.
    '''
    def __init__(self):
        '''
        '''





    def get_idx_grid(dir_results_msi, msi_run_id, idx_min=0):
        '''
        Finds the index of the processing scenario based on files written to disk

        The problem I have, is that after 10 loops, I am running into a
        memoryerror. I am not sure why this is, but one thing I can try is to
        restart the Python instance and begin the script from the beginning after
        every main loop. However, I must determine which processing scenario I
        am currently on based on the files written to disk.

        Parameters:
            dir_results_msi: directory to search
            msi_run_id:
            start: The minimum idx_grid to return (e.g., if start=100, then
                idx_grid will be forced to be at least 100; it will be higher if
                other folders already exist and processing as been performed)
        '''
        # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        folders_all = [f for f in os.listdir(dir_results_msi) if os.path.isdir(os.path.join(dir_results_msi, f))]
        folders_out = []
        idx_list = []
        str_match = 'msi_' + str(msi_run_id) + '_'  # eligible folders must have this in their name
        for f in folders_all:
            if str_match in f:
                idx_grid1 = int(f.replace(str_match, ''))
                if idx_grid1 >= idx_min:
                    idx_list.append(idx_grid1)
                    folders_out.append(f)
        for idx_grid2 in range(idx_min, max(idx_list)+2):
            if idx_grid2 not in idx_list: break
        return idx_grid2


def join_and_calc_dae(df_left, df_dates):
    df_left['date'] = pd.to_datetime(df_left['date'])
    df_dates[['date_plant', 'date_emerge']] = df_dates[['date_plant','date_emerge']].apply(pd.to_datetime, format='%Y-%m-%d')
    df_join = df_left.merge(df_dates, on=['study', 'year'], validate='many_to_one')
    df_join['dap'] = (df_join['date']-df_join['date_plant']).dt.days
    df_join['dae'] = (df_join['date']-df_join['date_emerge']).dt.days
    return df_join

def get_random_seed(seed=None):
    '''
    Assign the random seed
    '''
    if seed is None:
        seed = np.random.randint(0, 1e6)
    else:
        seed = int(seed)
    return seed

def split_train_test(df, test_size=0.4, random_seed=None, stratify=None):
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=stratify)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_test

def get_repeated_stratified_kfold(df, n_splits=3, n_repeats=2, random_state=None):
    X_null = np.zeros(len(df))  # not necessary for StratifiedKFold
    y_train_strat = df['dataset_id'].values
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)
    cv_rep_strat = rskf.split(X_null, y_train_strat)
    return cv_rep_strat

def check_stratified_proportions(df, cv_rep_strat):
    '''
    Checks the proportions of the stratifications in the dataset
    '''
    # cols_meta = ['dataset_id', 'date', 'plot_id', 'dae']
    cols_meta = ['dataset_id', 'date', 'plot_id']
    # X_meta_train = df_train[cols_meta].values
    # X_meta_test = df_test[cols_meta].values
    X_meta = df[cols_meta].values
    print('Number of observations in each cross-validation dataset (key=ID; value=n):')
    train_list = []
    val_list = []
    for train_index, val_index in cv_rep_strat:
        X_meta_train_fold = X_meta[train_index]
        X_meta_val_fold = X_meta[val_index]
        X_train_dataset_id = X_meta_train_fold[:,0]
        train = {}
        val = {}
        for uid in np.unique(X_train_dataset_id):
            n1 = len(np.where(X_meta_train_fold[:,0] == uid)[0])
            n2 = len(np.where(X_meta_val_fold[:,0] == uid)[0])
            train[uid] = n1
            val[uid] = n2
        train_list.append(train)
        val_list.append(val)
    print('Train set:')
    for item in train_list:
        print(item)
    print('Test set:')
    for item in val_list:
        print(item)

def impute_missing_data(X, random_seed, method='iterative'):
    '''
    method should be one of "iterative" (takes more time) or "knn"
    '''
    if np.isnan(X).any() is False:
        return X

    if method == 'iterative':
        imp = IterativeImputer(max_iter=10, random_state=random_seed)
    elif method == 'knn':
        imp = KNNImputer(n_neighbors=2, weights='uniform')
    X_out = imp.fit_transform(X)
    return X_out

def visnir_bands(df, wl_range=[400, 900]):
    '''
    Filters dataframe so it only keeps bands between the wavelength range
    (``wl_range``)
    '''
    bands_to_keep = []
    for c in df.columns:
        if not c.isnumeric():
            bands_to_keep.append(c)
        elif int(c) >= wl_range[0] or int(c) <= wl_range[1]:
            bands_to_keep.append(c)
    df_visnir = df[bands_to_keep].dropna(axis=1)
    return df_visnir

def split_by_cs_band_config(df, tissue='Petiole', measure='NO3_ppm', band='1480'):
    df_full = df[(df['tissue']==tissue) & (df['measure']==measure)].dropna(axis=1)
    df_visnir = visnir_bands(df, wl_range=[400, 900])
    df_swir = df[(df['tissue']==tissue) & (df['measure']==measure) & (pd.notnull(df[band]))].dropna(axis=1, how='all')
    df_re = df[(df['tissue']==tissue) & (df['measure']==measure) & (pd.isnull(df[band]))].dropna(axis=1, how='all')
