# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:54:22 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import numpy as np
import os
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


param_grid_las = {'alpha': list(np.logspace(-4, 0, 5))}
param_grid_svr = None
# param_grid_rf = {'n_estimators': [300], 'min_samples_split': list(np.linspace(2, 10, 5, dtype=int)), 'max_features': [0.05, 0.1, 0.3, 0.9]}
param_grid_rf = None
param_grid_pls = {'n_components': list(np.linspace(2, 10, 9, dtype=int)), 'scale': [True, False]}

def hs_grid_search(settings_dict, dir_out=None,
                   fname_out='batch_cv_settings.csv'):
    '''
    Reads ``settings_dict`` and returns a dataframe with all the necessary
    information to execute each specific processing scenario.

    Folder name will be the index of df_grid for each set of outputs, so
    df_grid must be referenced to know which folder corresponds to which
    scenario.

    Parameters:
        settings_dict (``dict``): A dictionary describing all the
            processing scenarios.
        dir_out (``str``): The folder directory to save the resulting
            DataFrame to (default: ``None``).
        fname_out (``str``): Filename to save the resulting DataFrame as
            (default: 'batch_cv_settings.csv').
    '''
    df_grid = pd.DataFrame(columns=settings_dict.keys())
    keys = settings_dict.keys()
    values = (settings_dict[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    for i in combinations:
        data = []
        for col in df_grid.columns:
            data.append(i[col])
        df_temp = pd.DataFrame(data=[data], columns=df_grid.columns)
        df_grid = df_grid.append(df_temp)
    df_grid = df_grid.reset_index(drop=True)
    # if csv is True:
    if dir_out is not None and os.path.isdir(dir_out):
        df_grid.to_csv(os.path.join(dir_out, fname_out), index=False)
    return df_grid

def split_train_test(df, test_size=0.4, random_seed=None, stratify=None):
    '''
    Splits ``df`` into train and test sets based on proportion indicated by
    ``test_size``
    '''
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=stratify)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_test

def get_repeated_stratified_kfold(df, n_splits=3, n_repeats=2,
                                  random_state=None):
    '''
    Stratifies ``df`` by "dataset_id", and creates a repeated, stratified
    k-fold cross-validation object that can be used for any sk-learn model
    '''
    X_null = np.zeros(len(df))  # not necessary for StratifiedKFold
    y_train_strat = df['dataset_id'].values
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)
    cv_rep_strat = rskf.split(X_null, y_train_strat)
    return cv_rep_strat

def check_stratified_proportions(df, cv_rep_strat):
    '''
    Checks the proportions of the stratifications in the dataset and prints
    the number of observations in each stratified group
    '''
    cols_meta = ['dataset_id', 'study', 'date', 'plot_id', 'trt', 'growth_stage']
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
    Imputes missing data in X - sk-learn models will not work with null data

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

def numeric_df_cols(df):
    '''
    Changes all numeric dataframe column headings to integer if they are
    strings. Caution, float-like strings will also be changed to integers.

    Useful becasue we want to access df columns by band number to make for
    convenient construction of the sk-learn X feature matrix
    '''
    df_out = df.copy()
    for c in df.columns:
        if isinstance(c, str) and c.isnumeric():
            df_out.rename(columns = {c: int(c)}, inplace=True)
    return df_out

def get_X_and_y(df, x_labels, y_label, random_seed=None, key_or_val='keys',
                extra=None):
    '''
    Gets the X and y from df; y is determined by the ``y_label`` column

    Parameters:
        df: The input dataframe to pull from
        x_labels: The column headings from ``df`` to include in the X matrix
        y_label: The column heading from ``df`` to include in the y vector
        random_seed: If data are missing in the X matrix, missing data will be
            imputed, which requires a ``random_seed``.
        key_or_val: If ``x_labels`` is a ``dict``, this denotes whether column
            labels should be either "keys" (e.g., column names are in the keys
            of ``x_labels``) or "values" (column names are in the values of
            ``x_labels``).
        extra: can be a string or a list of strings, but should be column names
            in ``df`` that should be appended to ``x_labels`` (e.g.,
            "pctl_10th").
    '''
    if isinstance(x_labels, dict):
        if key_or_val == 'keys':
            x_labels = sorted(list(x_labels.keys()))
        elif key_or_val == 'values':
            print(x_labels)
            x_labels = sorted(list(x_labels.values()))
    if extra is None:
        extra = [None]
    if extra != [None]:
        if not isinstance(extra, list):
            extra = [extra]
        for col in extra:
            x_labels.append(col)
    X = df[x_labels].values
    y = df[y_label].values
    X = impute_missing_data(X, random_seed, method='iterative')
    return X, y, x_labels

