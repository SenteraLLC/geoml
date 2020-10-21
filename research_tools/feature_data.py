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
import geopandas as gpd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# from research_tools import feature_groups
from research_tools import Tables


class FeatureData(Tables):
    '''
    Class that provides file management functionality specifically for research
    data that are used for the basic training of supervised regression models.
    This class assists in the loading, filtering, and preparation of research
    data for its use in a supervised regression model.
    '''
    __allowed_params = (
        'fname_obs_tissue', 'fname_cropscan', 'fname_sentinel',
        'random_seed', 'dir_results', 'group_feats',
        'ground_truth_tissue', 'ground_truth_measure',
        'date_tolerance', 'test_size', 'stratify', 'impute_method', 'n_splits',
        'n_repeats', 'train_test', 'print_out_fd')

    def __init__(self, **kwargs):
        super(FeatureData, self).__init__(**kwargs)
        self.load_tables()

        # FeatureData defaults
        self.random_seed = None
        # self.fname_obs_tissue = 'obs_tissue.csv'
        # self.fname_cropscan = 'rs_cropscan.csv'
        # self.fname_sentinel = 'rs_sentinel.csv'
        # self.fname_wx = 'calc_weather.csv'
        self.dir_results = None
        self.group_feats = {'dae': 'dae',
                            'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                                         'col_out': 'rate_ntd_kgha'},
                            'cropscan_wl_range1': [400, 900]}
        # self.ground_truth = 'vine_n_pct'
        self.ground_truth_tissue = 'vine'
        self.ground_truth_measure = 'n_pct'
        self.date_tolerance = 3
        self.test_size = 0.4
        self.stratify = ['study', 'date']
        self.impute_method = 'iterative'
        self.n_splits = 2
        self.n_repeats = 3
        self.train_test = 'train'
        self.print_out_fd = False
        # self.test_f_self(**kwargs)

        self._set_params_from_kwargs_fd(**kwargs)
        self._set_attributes_fd()

        if self.base_dir_data is None:
            raise ValueError('<base_dir_data> must be set to access data '
                             'tables, either with <config_dict> or via '
                             '<**kwargs>.')
        self._load_df_response()
        if self.dir_results is not None:
            os.makedirs(self.dir_results, exist_ok=True)
        self._get_random_seed()
        # self.tables = Tables(base_dir_data=self.base_dir_data)

    # def set_params_from_kwargs(self, **kwargs):
    #     print('_set_params_from_kwargs - entering')
    #     print(kwargs)
    #     print('_set_params_from_kwargs - exiting')

    def _set_params_from_dict_fd(self, config_dict):
        '''
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        '''
        if config_dict is not None and 'FeatureData' in config_dict:
            params_fd = config_dict['FeatureData']
        elif config_dict is not None and 'FeatureData' not in config_dict:
            params_fd = config_dict
        else:  # config_dict is None
            return
        for k, v in params_fd.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, v)

    def _set_params_from_kwargs_fd(self, **kwargs):
        '''
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'config_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``FeatureData`` more explicitly.
        '''
        if 'config_dict' in kwargs:
            self._set_params_from_dict_fd(kwargs.get('config_dict'))
        if kwargs is not None:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)

    def _set_attributes_fd(self):
        '''
        Sets any class attribute to ``None`` that will be created in one of the
        user functions
        '''
        self.df_obs_tissue = None
        self.df_tuber_biomdry_Mgha = None
        self.df_vine_biomdry_Mgha = None
        self.df_wholeplant_biomdry_Mgha = None
        self.df_tuber_biomfresh_Mgha = None
        self.df_canopy_cover_pct = None
        self.df_tuber_n_kgha = None
        self.df_vine_n_kgha = None
        self.df_wholeplant_n_kgha = None
        self.df_tuber_n_pct = None
        self.df_vine_n_pct = None
        self.df_wholeplant_n_pct = None
        self.df_petiole_no3_ppm = None
        self.df_cs = None
        self.df_wx = None

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
        self.stratify_train = None
        self.stratify_test = None

    def _handle_wl_cols(self, c, wl_range, labels_x, prefix='wl_'):
        '''
        Checks for reflectance column validity with a previx present.

        Parameters:
            c (``str``): The column label to evaluate. If it is numeric and
                resembles an integer, it will be added to labels_x.
            labels_x (``list``): The list that holds the x labels to use/keep.
            prefix (``str``): The prefix to disregard when evaluating <c> for its
                resemblance of an integer.

        Note:
            If prefix is set to '' or ``None``, <c> will be appended to <labels_x>.
        '''
        if not isinstance(c, str):
            c = str(c)
        col = c.replace(prefix, '') if prefix in c else c

        if (col.isnumeric() and int(col) >= wl_range[0] and
            int(col) <= wl_range[1]):
            labels_x.append(col)
        return labels_x

    def _get_labels_x(self, group_feats, cols=None):
        '''
        Parses ``group_feats`` and returns a list of column headings so that
        a df can be subset to build the X matrix with only the appropriate
        features indicated by ``group_feats``.
        '''
        labels_x = []
        for key in group_feats:
            if 'wl_range' in key:
                wl_range = group_feats[key]
                assert cols is not None, ('``cols`` must be passed.')
                for c in cols:
                    labels_x = self._handle_wl_cols(c, wl_range, labels_x,
                                                    prefix='wl_')
            elif 'bands' in key or 'wx' in key:
                labels_x.extend(group_feats[key])
            elif 'rate_ntd' in key:
                labels_x.append(group_feats[key]['col_out'])
            else:
                labels_x.append(group_feats[key])
        self.labels_x = labels_x
        return self.labels_x

    def _join_group_feats(self, df, group_feats, date_tolerance):
        '''
        Joins predictors to ``df`` based on the contents of group_feats
        '''
        if 'dae' in group_feats:
            df = self.dae(df)  # add DAE
        if 'dap' in group_feats:
            df = self.dap(df)  # add DAP
        if 'rate_ntd' in group_feats:
            col_rate_n = group_feats['rate_ntd']['col_rate_n']
            col_rate_ntd_out = group_feats['rate_ntd']['col_out']
            # unit_str = value.rsplit('_', 1)[1]
            df = self.rate_ntd(df, col_rate_n=col_rate_n,
                 col_rate_ntd_out=col_rate_ntd_out)
        if 'wx' in group_feats:
            df = self.join_closest_date(  # join wx by closest date
                df, self.df_wx, left_on='date', right_on='date',
                tolerance=0, by=['owner', 'study', 'year'])

        for key in group_feats:  # necessary because 'cropscan_wl_range1' must be differentiated
            if 'cropscan' in key:
                df = self.join_closest_date(  # join cropscan by closest date
                    df, self.rs_cropscan, left_on='date', right_on='date',
                    tolerance=date_tolerance, by=['owner', 'study', 'year', 'plot_id'])
                break
        return df

    def _get_primary_keys(self, df):
        '''
        Checks df columns to see if "research" or "client" primary keys exist.

        Duplicate function in db_handler!

        Returns:
            subset (``list``): A list of the primary keys to group by, etc.
        '''
        if set(['owner', 'farm', 'field_id', 'year']).issubset(df.columns):
            subset = ['owner', 'farm', 'field_id', 'year']
        elif set(['owner', 'study', 'plot_id', 'year']).issubset(df.columns):
            subset = ['owner', 'study', 'plot_id', 'year']
        else:
            print('Neither "research" or "client" primary keys are '
                  'present in <df>.')
            subset = None
        return subset

    # def _read_csv_geojson(self, fname):
    #     '''
    #     Depending on file extension, will read from either pd or gpd
    #     '''
    #     if os.path.splitext(fname)[-1] == '.csv':
    #         df = pd.read_csv(fname)
    #     elif os.path.splitext(fname)[-1] == '.geojson':
    #         df = gpd.read_file(fname)
    #     else:
    #         raise TypeError('<fname_sentinel> must be either a .csv or '
    #                         '.geojson...')
    #     return df

    def _load_df_response(self, tissue_col='tissue', measure_col='measure',
                          value_col='value'):
        '''
        Loads the response DataFrame based on <ground_truth_tissue> and
        <ground_truth_measure>. The observations are retrieved from the
        <obs_tissue> table.
        '''
        tissue = self.ground_truth_tissue
        measure = self.ground_truth_measure
        print('\nLoading response dataframe...\nTissue: {0}\nMeasure: {1}\n'
              ''.format(tissue, measure))
        # fname_obs_tissue = os.path.join(self.base_dir_data, self.fname_obs_tissue)
        # df_obs_tissue = self._read_csv_geojson(fname_obs_tissue)
        self.labels_y_id = [tissue_col, measure_col]
        self.label_y = value_col
        self.obs_tissue = self.obs_tissue[pd.notnull(self.obs_tissue[value_col])]
        self.df_response = self.obs_tissue[(self.obs_tissue[measure_col] == measure) &
                                           (self.obs_tissue[tissue_col] == tissue)]

    # def _load_tables(self, tissue='petiole', measure='no3_ppm',
    #                  tissue_col='tissue', measure_col='measure',
    #                  value_col='value'):
    #     '''
    #     Loads the appropriate table based on the value passed for ``tissue``,
    #     then filters observations according to
    #     '''
        # # get all unique combinations of tissue and measure cols
        # tissue = self.obs_tissue.groupby(by=[measure_col, tissue_col], as_index=False).first()[tissue_col].tolist()
        # measure = self.obs_tissue.groupby(by=[measure_col, tissue_col], as_index=False).first()[measure_col].tolist()
        # for tissue, measure in zip(tissue, measure):
        #     df = self.obs_tissue[(self.obs_tissue[measure_col] == measure) &
        #                          (self.obs_tissue[tissue_col] == tissue)]
        #     if tissue == 'tuber' and measure == 'biomdry_Mgha':
        #         self.df_tuber_biomdry_Mgha = df.copy()
        #     elif tissue == 'vine' and measure == 'biomdry_Mgha':
        #         self.df_vine_biomdry_Mgha = df.copy()
        #     elif tissue == 'wholeplant' and measure == 'biomdry_Mgha':
        #         self.df_wholeplant_biomdry_Mgha = df.copy()
        #     elif tissue == 'tuber' and measure == 'biomfresh_Mgha':
        #         self.df_tuber_biomfresh_Mgha = df.copy()
        #     elif tissue == 'canopy' and measure == 'cover_pct':
        #         self.df_canopy_cover_pct = df.copy()
        #     elif tissue == 'tuber' and measure == 'n_kgha':
        #         self.df_tuber_n_kgha = df.copy()
        #     elif tissue == 'vine' and measure == 'n_kgha':
        #         self.df_vine_n_kgha = df.copy()
        #     elif tissue == 'wholeplant' and measure == 'n_kgha':
        #         self.df_wholeplant_n_kgha = df.copy()
        #     elif tissue == 'tuber' and measure == 'n_pct':
        #         self.df_tuber_n_pct = df.copy()
        #     elif tissue == 'vine' and measure == 'n_pct':
        #         self.df_vine_n_pct = df.copy()
        #     elif tissue == 'wholeplant' and measure == 'n_pct':
        #         self.df_wholeplant_n_pct = df.copy()
        #     elif tissue == 'petiole' and measure == 'no3_ppm':
        #         self.df_petiole_no3_ppm = df.copy()

        # fname_cropscan = os.path.join(self.base_dir_data, self.fname_cropscan)
        # if os.path.isfile(fname_cropscan):
        #     df_cs = self._read_csv_geojson(fname_cropscan)
        #     subset = self._get_primary_keys(df_cs)
        #     self.df_cs = df_cs.groupby(subset + ['date']).mean().reset_index()
        # fname_sentinel = os.path.join(self.base_dir_data, self.fname_sentinel)
        # if os.path.isfile(fname_sentinel):
        #     df_sentinel = self._read_csv_geojson(fname_sentinel)
        #     df_sentinel.rename(columns={'acquisition_time': 'date'}, inplace=True)
        #     subset = self._get_primary_keys(df_sentinel)
        #     self.df_sentinel = df_sentinel.groupby(subset + ['date']
        #                                            ).mean().reset_index()
        # fname_wx = os.path.join(self.base_dir_data, self.fname_wx)
        # if os.path.isfile(fname_wx):
        #     df_wx = self._read_csv_geojson(fname_wx)
        #     subset = self._get_primary_keys(df_sentinel)
        #     subset = [i for i in subset if i not in ['field_id', 'plot_id']]
        #     self.df_wx = df_wx.groupby(subset + ['date']).mean().reset_index()
        # TODO: Function to filter cropscan data (e.g., low irradiance, etc.)

    def _write_to_readme(self, msg, msi_run_id=None, row=None):
        '''
        Writes ``msg`` to the README.txt file
        '''
        # Note if I get here to modify foler_name or use msi_run_id:
        # Try to keep msi-run_id out of this class; instead, make all folder
        # names, etc. be reflected in the self.dir_results variable (?)
        # if msi_run_id is not None and row is not None:
        #     folder_name = 'msi_' + str(msi_run_id) + '_' + str(row.name).zfill(3)
        #     dir_out = os.path.join(self.dir_results, folder_name)
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

    def _get_response_df(self, tissue, measure,
                         tissue_col='tissue', measure_col='measure'):
                         # ground_truth='vine_n_pct'):
        '''
        Gets the relevant response dataframe

        Parameters:
            ground_truth_tissue (``str``): The tissue to use for the response
                variable. Must be in "obs_tissue.csv", and dictates which table
                to access to retrieve the relevant training data.
            ground_truth_measure (``str``): The measure to use for the response
                variable. Must be in "obs_tissue.csv"
            tissue_col (``str``): The column name from "obs_tissue.csv" to look
                for ``tissue``.
            measure_col (``str``): The column name from "obs_tissue.csv" to
                look for ``measure``.
        '''
        tissue_list = self.obs_tissue.groupby(by=[measure_col, tissue_col], as_index=False).first()[tissue_col].tolist()
        measure_list = self.obs_tissue.groupby(by=[measure_col, tissue_col], as_index=False).first()[measure_col].tolist()
        avail_list = ['_'.join(map(str, i)) for i in zip(tissue_list, measure_list)]
        # avail_list = ["vine_n_pct", "pet_no3_ppm", "tuber_n_pct",
        #               "biomass_kgha"]
        msg = ('``tissue``  and ``measure`` must be '
               'one of:\n{0}.\nPlease see "obs_tissue" table to be sure your '
               'intended data are available.'
               ''.format(list(zip(tissue_list, measure_list))))
        assert '_'.join((tissue, measure)) in avail_list, msg

        df = self.obs_tissue[(self.obs_tissue[measure_col] == measure) &
                             (self.obs_tissue[tissue_col] == tissue)]
        return df

    def _stratify_set(self):
        '''
        Creates a 1-D array of the stratification IDs (to be used by k-fold)
        for both the train and test sets: <stratify_train> and <stratify_test>
        '''
        msg1 = ('All <stratify> strings must be columns in <df_y>')
        for c in self.stratify:
            assert c in self.df_y.columns, msg1

        self.stratify_train = self.df_y[
            self.df_y['train_test'] == 'train'].groupby(self.stratify
                                                        ).ngroup().values
        self.stratify_test = self.df_y[
            self.df_y['train_test'] == 'test'].groupby(self.stratify
                                                        ).ngroup().values

    def _train_test_split_df(self, df):
        '''
        Splits ``df`` into train and test sets; all parameters used by
        ``sklearn.train_test_split`` must have been set before invoking this
        function.

        Parameters:
            df:
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
        elif method == None:
            return X
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
            impute_method (``str``): The sk-learn imputation method for missing
                data. If ``None``, then any row with missing data is removed
                from the dataset.
        '''
        msg = ('``impute_method`` must be one of: ["iterative", "knn", None]')
        assert impute_method in ['iterative', 'knn', None], msg

        if impute_method is None:
            df = df.dropna()
        else:
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
        return X_train, X_test, y_train, y_test, df

    def _save_df_X_y(self):
        '''
        Saves both ``FeatureData.df_X`` and ``FeatureData.df_y`` to
        ``FeatureData.dir_results``.
        '''
        # if self.group_feats is None or self.label_y is None:
        #     print('<group_feats> and <label_y> must be set to save data to '
        #           '<dir_results>. Have you ran `get_feat_group_X_y()` yet?')
        #     return
        dir_out = os.path.join(self.dir_results, self.label_y)
        os.makedirs(dir_out, exist_ok=True)

        fname_out_X = os.path.join(dir_out, 'data_X_' + self.label_y + '.csv')
        fname_out_y = os.path.join(dir_out, 'data_y_' + self.label_y + '.csv')
        self.df_X.to_csv(fname_out_X, index=False)
        self.df_y.to_csv(fname_out_y, index=False)

    def _kfold_repeated_stratified_print(self, cv_rep_strat, train_test='train'):
        '''
        Checks the proportions of the stratifications in the dataset and prints
        the number of observations in each stratified group. The keys are based
        on the stratified IDs in <stratify_train> or <stratify_test>
        '''
        df_X = self.df_X[self.df_X['train_test'] == train_test]
        if train_test == 'train':
            stratify_vector = self.stratify_train
        elif train_test == 'test':
            stratify_vector = self.stratify_test
        print('The number of observations in each cross-validation dataset '
              'are listed below.\nThe key represents the <stratify_{0}> ID, '
              'and the value represents the number of observations used from '
              'that stratify ID'.format(train_test))
        print('Total number of observations: {0}'.format(len(stratify_vector)))
        train_list = []
        val_list = []
        for train_index, val_index in cv_rep_strat:
            X_meta_train_fold = stratify_vector[train_index]
            X_meta_val_fold = stratify_vector[val_index]
            X_train_dataset_id = X_meta_train_fold[:]
            train = {}
            val = {}
            for uid in np.unique(X_train_dataset_id):
                n1 = len(np.where(X_meta_train_fold[:] == uid)[0])
                n2 = len(np.where(X_meta_val_fold[:] == uid)[0])
                train[uid] = n1
                val[uid] = n2
            train_list.append(train)
            val_list.append(val)
        print('\nK-fold train set:')
        print('Number of observations: {0}'.format(len(train_index)))
        print(*train_list, sep='\n')
        print('\nK-fold validation set:')
        print('Number of observations: {0}'.format(len(val_index)))
        print(*val_list, sep='\n')

    def get_feat_group_X_y(self, **kwargs):
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
            >>> from research_tools import FeatureData
            >>> from research_tools.tests import config

            >>> feat_data_cs = FeatureData(config_dict=config.config_dict)
            >>> feat_data_cs.get_feat_group_X_y(test_size=0.1)
            >>> print('Shape of training matrix "X": {0}'.format(feat_data_cs.X_train.shape))
            >>> print('Shape of training vector "y": {0}'.format(feat_data_cs.y_train.shape))
            >>> print('Shape of testing matrix "X":  {0}'.format(feat_data_cs.X_test.shape))
            >>> print('Shape of testing vector "y":  {0}'.format(feat_data_cs.y_test.shape))
            Shape of training matrix "X": (579, 14)
            Shape of training vector "y": (579,)
            Shape of testing matrix "X":  (65, 14)
            Shape of testing vector "y":  (65,)
        '''
        print('Getting feature data...')
        self._set_params_from_kwargs_fd(**kwargs)
            # group_feats=group_feats, ground_truth=ground_truth,
            # date_tolerance=date_tolerance, test_size=test_size,
            # stratify=stratify)

        # df, labels_y_id, label_y = self._get_response_df(self.ground_truth)
        df = self._get_response_df(self.ground_truth_tissue,
                                   self.ground_truth_measure)
        df = self._join_group_feats(df, self.group_feats, self.date_tolerance)
        df = self._train_test_split_df(df)

        X_train, X_test, y_train, y_test, df = self._get_X_and_y(
            df, impute_method=self.impute_method)

        labels_id = ['owner', 'study', 'year', 'plot_id', 'date', 'train_test']
        self.df_X = df[labels_id + self.labels_x]
        self.df_y = df[labels_id + self.labels_y_id + [self.label_y]]
        self.labels_id = labels_id
        self._stratify_set()

        if self.dir_results is not None:
            self._save_df_X_y()

    def kfold_repeated_stratified(self, **kwargs):
        '''
        Builds a repeated, stratified k-fold cross-validation ``sklearn``
        object for both the X matrix and y vector based on
        ``FeatureData.df_X`` and ``FeatureData.df_y``. The returned
        cross-validation object can be used for any ``sklearn`` model.

        Parameters:
            n_splits (``int``): Number of folds. Must be at least 2
                (default: 4).
            n_repeats (``int``): Number of times cross-validator needs to be
                repeated (default: 3).
            train_test (``str``): Because ``df_X`` and ``df_y`` have a column
                denoting whether any given observation belongs to the training
                set or the test set. This parameter indicates if observations
                from the training set (i.e., "train") or the test set (i.e.,
                "test") should be stratified (default: "train").
            print_out_fd (``bool``): If ``print_out_fd`` is set to ``True``, the
                number of observations in each k-fold stratification will be
                printed to the console (default: ``False``).

        Returns:
            cv_rep_strat: A repeated, stratified cross-validation object
                suitable to be used with sklearn models.

        Example:
            >>> from research_tools import FeatureData
            >>> from research_tools.tests import config

            >>> feat_data_cs = FeatureData(config_dict=config.config_dict)
            >>> feat_data_cs.get_feat_group_X_y()
            >>> cv_rep_strat = feat_data_cs.kfold_repeated_stratified(print_out_fd=True)
            Number of splits: 2
            Number of repetitions: 3
            The number of observations in each cross-validation dataset are listed below.
            The key represents the <stratify_train> ID, and the value represents the number of observations used from that stratify ID
            Total number of observations: 386

            K-fold train set:
            Number of observations: 193
            {0: 10, 1: 9, 2: 10, 3: 9, 4: 10, 5: 9, 6: 9, 7: 10, 8: 9, 9: 11, 10: 11, 11: 11, 12: 11, 13: 16, 14: 16, 15: 16, 16: 16}
            {0: 9, 1: 10, 2: 9, 3: 9, 4: 9, 5: 10, 6: 10, 7: 9, 8: 10, 9: 11, 10: 11, 11: 11, 12: 11, 13: 16, 14: 16, 15: 16, 16: 16}
            {0: 10, 1: 9, 2: 10, 3: 9, 4: 10, 5: 9, 6: 9, 7: 10, 8: 9, 9: 11, 10: 11, 11: 11, 12: 11, 13: 16, 14: 16, 15: 16, 16: 16}
            {0: 9, 1: 10, 2: 9, 3: 9, 4: 9, 5: 10, 6: 10, 7: 9, 8: 10, 9: 11, 10: 11, 11: 11, 12: 11, 13: 16, 14: 16, 15: 16, 16: 16}
            {0: 10, 1: 9, 2: 10, 3: 9, 4: 10, 5: 9, 6: 9, 7: 10, 8: 9, 9: 11, 10: 11, 11: 11, 12: 11, 13: 16, 14: 16, 15: 16, 16: 16}
            {0: 9, 1: 10, 2: 9, 3: 9, 4: 9, 5: 10, 6: 10, 7: 9, 8: 10, 9: 11, 10: 11, 11: 11, 12: 11, 13: 16, 14: 16, 15: 16, 16: 16}
        '''
        self._set_params_from_kwargs_fd(**kwargs)

        if self.train_test == 'train':
            X = self.X_train
            y = self.y_train
            stratify_vector = self.stratify_train
        elif self.train_test == 'test':
            X = self.X_test
            y = self.y_test
            stratify_vector = self.stratify_test
        msg1 = ('<X> and <y> must have the same length')
        msg2 = ('<stratify_vector> must have the same length as <X> and <y>')
        assert len(X) == len(y), msg1
        assert len(stratify_vector) == len(y), msg2

        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats,
            random_state=self.random_seed)
        if self.print_out_fd is True:
            print('\nNumber of splits: {0}\nNumber of repetitions: {1}'
                  ''.format(self.n_splits, self.n_repeats))
            cv_rep_strat = rskf.split(X, stratify_vector)
            self._kfold_repeated_stratified_print(cv_rep_strat)
        cv_rep_strat = rskf.split(X, stratify_vector)
        return cv_rep_strat
