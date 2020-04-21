# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:42:05 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from research_tools import feature_groups
from research_tools import train_prep
from research_tools import join_tables


class feature_data(object):
    '''
    Class that provides file management functionality specifically for research
    data that are used for the basic training of supervised regression models.
    This class assists in the loading, filtering, and preparation of research
    data for its use in a supervised regression model.
    '''
    __allowed_kwargs = (
        'date_tolerance', 'ground_truth',  'group_feats', 'random_seed',
        'stratify', 'test_size')

    def __init__(self, base_dir_data,
                 fname_petiole='tissue_petiole_NO3_ppm.csv',
                 fname_total_n='tissue_wp_N_pct.csv',
                 fname_cropscan='cropscan.csv', random_seed=None,
                 dir_results=None):
        '''

        Parameters:
            base_dir_data:
            fname_petiole
            fname_total_n
            fname_cropscan
            random_seed
            dir_results (``str``): the directory to save any intermediate
                results to. If ``None``, results are stored in ``feature_data``
                object memory.
        '''
        self.base_dir_data = base_dir_data
        self.fname_petiole = fname_petiole
        self.fname_total_n = fname_total_n
        self.fname_cropscan = fname_cropscan
        self.random_seed = random_seed
        self.dir_results = dir_results

        self.df_pet_no3 = None
        self.df_vine_n_pct = None
        self.df_tuber_n_pct = None
        self.df_cs = None

        self.df_X = None
        self.df_y = None

        # "labels" vars indicate the df columns in the X matrix and y vector
        self.labels_id = None
        self.labels_x = None
        self.labels_y_id = None
        self.label_y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        if self.dir_results is not None:
            os.makedirs(self.dir_results, exist_ok=True)

        self._get_random_seed()
        self._load_tables()
        self.join_info = join_tables(self.base_dir_data)

        # extra kwargs that are set when parameters are passed via user functs
        self.date_tolerance = None
        self.ground_truth = None
        self.group_feats = None
        self.stratify = None
        self.test_size = None

    def _get_labels_x(self, group_feats, cols=None):
        '''
        Parses ``group_feats`` and returns a list of column headings so that
        a df can be subset to build the X matrix
        '''
        labels_x = []
        for key in group_feats:
            if 'cropscan_wl_range' in key:
                wl_range = group_feats[key]

                assert cols is not None, ('``cols`` must be passed.')
                for c in cols:
                    if (c.isnumeric() and int(c) > wl_range[0] and
                        int(c) < wl_range[1]):
                        labels_x.append(c)
            elif 'cropscan_bands' in key:
                labels_x.extend(group_feats[key])
            elif 'rate_ntd' in key:
                labels_x.append(group_feats[key]['col_out'])
            else:
                labels_x.append(group_feats[key])
        self.labels_x = labels_x
        return self.labels_x

    def _filter_df_bands(self, df, bands=None, wl_range=None):
        '''
        Filters dataframe so it only keeps bands designated by ``bands`` or
        between the bands in ``wl_range``.
        '''
        msg1 = ('Only one of ``bands`` or ``wl_range`` must be passed, not '
               'both.')
        msg2 = ('At least one of ``bands`` or ``wl_range`` must be passed.')
        assert not (bands is not None and wl_range is not None), msg1
        assert not (bands is None and wl_range is None), msg2

        bands_to_keep = []
        if wl_range is not None:
            for c in df.columns:
                if not c.isnumeric():
                    bands_to_keep.append(c)
                elif int(c) >= wl_range[0] or int(c) <= wl_range[1]:
                    bands_to_keep.append(c)
        if bands is not None:
            for c in df.columns:
                if not c.isnumeric():
                    bands_to_keep.append(c)
                elif c in bands or int(c) in bands:
                    bands_to_keep.append(c)
        df_filter = df[bands_to_keep].dropna(axis=1)
        return df_filter, bands_to_keep

    def _join_group_feats(self, df, group_feats, date_tolerance):
        '''
        Joins predictors to ``df`` based on the contents of group_feats
        '''
        if 'dae' in group_feats:
            df = self.join_info.dae(df)  # add DAE
        if 'dap' in group_feats:
            df = self.join_info.dap(df)  # add DAE
        if 'rate_ntd' in group_feats:
            value = group_feats['rate_ntd']['col_rate_n']
            unit_str = value.rsplit('_', 1)[1]
            df = self.join_info.rate_ntd(df, col_rate_n=value,
                                         unit_str=unit_str)
        for key in group_feats:
            if 'cropscan' in key:
                df = self.join_info.join_closest_date(  # join cropscan by closest date
                    df, self.df_cs, left_on='date', right_on='date',
                    tolerance=date_tolerance)
                break
        return df

    def _load_tables(self):
        '''
        Loads the appropriate table based on the value passed for ``tissue``,
        then filters observations according to
        '''
        fname_petiole = os.path.join(self.base_dir_data, self.fname_petiole)
        df_pet_no3 = pd.read_csv(fname_petiole)
        self.df_pet_no3 = df_pet_no3[pd.notnull(df_pet_no3['value'])]

        fname_total_n = os.path.join(self.base_dir_data, self.fname_total_n)
        df_total_n = pd.read_csv(fname_total_n)
        df_vine_n_pct = df_total_n[df_total_n['tissue'] == 'Vine']
        self.df_vine_n_pct = df_vine_n_pct[pd.notnull(df_vine_n_pct['value'])]

        df_tuber_n_pct = df_total_n[df_total_n['tissue'] == 'Tuber']
        self.df_tuber_n_pct = df_tuber_n_pct[pd.notnull(df_tuber_n_pct['value'])]

        fname_cropscan = os.path.join(self.base_dir_data, self.fname_cropscan)
        df_cs = pd.read_csv(fname_cropscan)
        self.df_cs = df_cs.groupby(['study', 'year', 'plot_id', 'date']
                                   ).mean().reset_index()
        # TODO: Function to filter cropscan data (e.g., low irradiance, etc.)
        # self.df_cs = df_cs[pd.notnull(df_cs['value'])]


    def _write_to_readme(self, msg, msi_run_id=None, row=None):
        '''
        Writes ``msg`` to the README.txt file
        '''
        # Note if I get here to modify foler_name or use msi_run_id:
        # Try to keep msi-run_id out of this class; instead, make all folder
        # names, etc. be reflected in the self.dir_results variable (?)
        if msi_run_id is not None and row is not None:
            folder_name = 'msi_' + str(msi_run_id) + '_' + str(row.name).zfill(3)
            dir_out = os.path.join(self.dir_results, folder_name)
        # with open(os.path.join(self.dir_results, folder_name + '_README.txt'), 'a') as f:

        if self.dir_results is None:
            print('<dir_results> must be set to create README file.')
            return
        else:
            with open(os.path.join(self.dir_results, 'README.txt'), 'a') as f:
                f.write(str(msg) + '\n')

    def _get_random_seed(self):
        '''
        Assign the random seed
        '''
        if self.random_seed is None:
            self.random_seed = np.random.randint(0, 1e6)
        else:
            self.random_seed = int(self.random_seed)
        self._write_to_readme('Random seed: {0}'.format(self.random_seed))

    def _get_response_df(self, ground_truth='vine_n_pct'):
        '''
        Gets the relevant response dataframe

        Parameters:
            ground_truth (``str``): Must be one of "vine_n_pct", "pet_no3_ppm",
                or "tuber_n_pct"; dictates which table to access to retrieve
                the relevant training data.
        '''
        avail_list = ["vine_n_pct", "pet_no3_ppm", "tuber_n_pct"]
        msg = ('``ground_truth`` must be one of: {0}'.format(avail_list))
        assert ground_truth in avail_list, msg

        if ground_truth == 'vine_n_pct':
            self.labels_y_id = ['tissue', 'measure']
            self.label_y = 'value'
            return self.df_vine_n_pct.copy(), self.labels_y_id, self.label_y
        if ground_truth == 'pet_no3_ppm':
            self.labels_y_id = ['tissue', 'measure']
            self.label_y = 'value'
            return self.df_pet_no3.copy(), self.labels_y_id, self.label_y
        if ground_truth == 'tuber_n_pct':
            self.labels_y_id = ['tissue', 'measure']
            self.label_y = 'value'
            return self.df_tuber_n_pct.copy(), self.labels_y_id, self.label_y

    def _add_stratify_id(self, df):
        '''
        Creates stratification column to be able to properly stratify train
        and test sets
        '''
        if isinstance(self.stratify, list):
            if 'stratify_id' not in df.columns:
                df.insert(0, 'stratify_id', None)
            df['stratify_id'] = df.groupby(self.stratify).ngroup()
            self.stratify = 'stratify_id'
        return df

    def _train_test_split_df(self, df):
        '''
        Splits ``df`` into train and test sets based on proportion indicated by
        ``test_size``

        Parameters:
            df:
            test_size:
            random_seed:
            stratify:
        '''
        # df = self._add_stratify_id(df)
        df_stratify = df[self.stratify]
        df_train, df_test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_seed,
            stratify=df_stratify)
        df_train.insert(0, 'train_test', 'train')
        df_test.insert(0, 'train_test', 'test')
        df = df_train.copy()
        df = df.append(df_test).reset_index(drop=True)
        return df

    def _get_repeated_stratified_kfold(self, df, n_splits=3, n_repeats=2,
                                       random_state=None):
        '''
        Stratifies ``df`` by "dataset_id", and creates a repeated, stratified
        k-fold cross-validation object that can be used for any sk-learn model

        Parameters:
            df:
            n_splits:
            n_repeats:
            random_state:
        '''
        X_null = np.zeros(len(df))  # not necessary for StratifiedKFold
        y_train_strat = df['dataset_id'].values
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=random_state)
        cv_rep_strat = rskf.split(X_null, y_train_strat)
        return cv_rep_strat

    def _check_stratified_proportions(self, df, cv_rep_strat):
        '''
        Checks the proportions of the stratifications in the dataset and prints
        the number of observations in each stratified group

        Parameters:
            df:
            cv_rep_strat:
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

    def _set_params(self, **kwargs):
        '''
        Simply sets any of the passed paramers to self as long as they
        '''
        if kwargs is not None:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_kwargs and v is not None:
                    setattr(self, k, v)

    def _impute_missing_data(self, X, method='iterative'):
        '''
        Imputes missing data in X - sk-learn models will not work with null data

        Parameters:
            method (``str``): should be one of "iterative" (takes more time)
                or "knn" (default: "iterative").
        '''
        if np.isnan(X).any() is False:
            return X

        if method == 'iterative':
            imp = IterativeImputer(max_iter=10, random_state=self.random_seed)
        elif method == 'knn':
            imp = KNNImputer(n_neighbors=2, weights='uniform')
        X_out = imp.fit_transform(X)
        return X_out

    def _get_X_and_y(self, df, impute_method='iterative'):
        '''
        Gets the X and y from df; y is determined by the ``y_label`` column.
        This function depends on the having the following variables already
        set:
            1. self.label_y
            2. self.group_feats

        Parameters:
            df (``pd.DataFrame``): The input dataframe to retrieve data from.
        '''
        msg = ('``impute_method`` must be one of: ["iterative", "knn"]')
        assert impute_method in ['iterative', 'knn'], msg

        df = df[pd.notnull(df[self.label_y])]
        labels_x = self._get_labels_x(self.group_feats, cols=df.columns)

        df_train = df[df['train_test'] == 'train']
        df_test = df[df['train_test'] == 'test']

        X_train = df_train[labels_x].values
        X_test = df_test[labels_x].values
        y_train = df_train[self.label_y].values
        y_test = df_test[self.label_y].values

        X_train = self._impute_missing_data(X_train, method=impute_method)
        X_test = self._impute_missing_data(X_test, method=impute_method)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return X_train, X_test, y_train, y_test

    def _save_df_X_y(self):
        '''
        Saves the joined dataframe to a new folder in ``dir_results`` with another
        README.txt file that provides some basic details about the processing that
        was performed in case the grid_idx gets messed up..

        grid_idx is the index of df_grid (this will change if df_grid changes,
            so that is why ``msi_run_id`` is also included in the folder name)
        '''
        if self.group_feats is None or self.label_y is None:
            print('<group_feats> and <label_y> must be set to save data to '
                  '<dir_results>. Have you ran `get_feat_group_X_y()` yet?')
            return
        dir_out = os.path.join(self.dir_results, self.label_y)
        os.makedirs(dir_out, exist_ok=True)

        fname_out_X = os.path.join(dir_out, 'data_X_' + self.label_y + '.csv')
        fname_out_y = os.path.join(dir_out, 'data_y_' + self.label_y + '.csv')
        self.df_X.to_csv(fname_out_X, index=False)
        self.df_y.to_csv(fname_out_y, index=False)

    def get_feat_group_X_y(
            self, group_feats, ground_truth='vine_n_pct', date_tolerance=3,
            random_seed=None, test_size=0.4, stratify=['study', 'date'],
            impute_method='iterative'):
        '''
        Retrieves all the necessary columns in ``group_feats``, then filters
        the dataframe so that it is left with only the identifying columns
        (i.e., study, year, and plot_id), a column indicating if each
        observation belongs to the train or test set (i.e., train_test), and
        the feature columns indicated by ``group_feats``.

        Parameters:
            group_feats (``list`` or ``dict``): The column headings to include
                in the X matrix. ``group_feats`` must follow the naming
                conventions outlined in featuer_groups.py to ensure that the
                intended features are joined to ``df_feat_group``.
            ground_truth (``str``): Must be one of "vine_n_pct", "pet_no3_ppm",
                or "tuber_n_pct"; dictates which table to access to retrieve
                the relevant training data.
            date_tolerance (``int``): Number of days away to still allow join
                between response data and predictor features (if dates are
                greater than ``date_tolerance``, the join will not occur and
                data will be neglected). Only relevant if predictor features
                were collected on a different day than response features.
            test_size (``float``):
            stratify (``str``):
            impute_method (``str``):

        Example:
            >>> from research_tools import feature_data
            >>> from research_tools import feature_groups

            >>> base_dir_data = 'I:/Shared drives/NSF STTR Phase I – Potato Remote Sensing/Historical Data/Rosen Lab/Small Plot Data/Data'
            >>> feat_data_cs = feature_data(base_dir_data)
            >>> group_feats = feature_groups.cs_test2
            >>> feat_data_cs.get_feat_group_X_y(group_feats)
            >>> print('Shape of training matrix "X": {0}'.format(feat_data_cs.X_train.shape))
            >>> print('Shape of training vector "y": {0}'.format(feat_data_cs.y_train.shape))
            >>> print('Shape of testing matrix "X":  {0}'.format(feat_data_cs.X_test.shape))
            >>> print('Shape of testing vector "y":  {0}'.format(feat_data_cs.y_test.shape))
        '''
        self._set_params(
            group_feats=group_feats, ground_truth=ground_truth,
            date_tolerance=date_tolerance, random_seed=random_seed,
            test_size=test_size, stratify=stratify)
        df, labels_y_id, label_y = self._get_response_df(ground_truth)
        df = self._join_group_feats(df, group_feats, date_tolerance)
        df = self._train_test_split_df(df)

        X_train, X_test, y_train, y_test = self._get_X_and_y(
            df, impute_method=impute_method)

        labels_id = ['study', 'year', 'plot_id', 'train_test']
        self.df_X = df[labels_id + self.labels_x]
        self.df_y = df[labels_id + labels_y_id + [label_y]]
        self.labels_id = labels_id

        if self.dir_results is not None:
            self._save_df_X_y()

    def split_by_cs_band_config(df, tissue='Petiole', measure='NO3_ppm', band='1480'):
        df_full = df[(df['tissue']==tissue) & (df['measure']==measure)].dropna(axis=1)
        df_visnir = self.filter_df_bands(df, wl_range=[400, 900])
        df_swir = df[(df['tissue']==tissue) & (df['measure']==measure) & (pd.notnull(df[band]))].dropna(axis=1, how='all')
        df_re = df[(df['tissue']==tissue) & (df['measure']==measure) & (pd.isnull(df[band]))].dropna(axis=1, how='all')
