# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:42:05 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
import inspect
import numpy as np
import os
import pandas as pd
import geopandas as gpd
import warnings

from copy import deepcopy
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import db.utilities as db_utils
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
        'date_tolerance', 'cv_method', 'cv_method_kwargs', 'cv_split_kwargs',
        'impute_method', 'cv_method_tune', 'cv_method_tune_kwargs',
        'cv_split_tune_kwargs',
        # 'kfold_stratify', 'n_splits', 'n_repeats',
        'train_test', 'print_out_fd', 'print_splitter_info')

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
        self.cv_method = train_test_split
        self.cv_method_kwargs = {'arrays': 'df', 'test_size': '0.4', 'stratify': 'df[["owner", "year"]]'}
        self.cv_split_kwargs = None
        self.impute_method = 'iterative'
        # self.kfold_stratify = ['owner', 'year']
        # self.n_splits = 2
        # self.n_repeats = 3
        self.train_test = 'train'
        self.cv_method_tune = RepeatedStratifiedKFold
        self.cv_method_tune_kwargs = {'n_splits': 4, 'n_repeats': 3}
        self.cv_split_tune_kwargs = None
        self.print_out_fd = False
        self.print_splitter_info = False
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
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)

    def _set_attributes_fd(self):
        '''
        Sets any class attribute to ``None`` that will be created in one of the
        user functions
        '''
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

        # self.df_obs_tissue = None
        # self.df_tuber_biomdry_Mgha = None
        # self.df_vine_biomdry_Mgha = None
        # self.df_wholeplant_biomdry_Mgha = None
        # self.df_tuber_biomfresh_Mgha = None
        # self.df_canopy_cover_pct = None
        # self.df_tuber_n_kgha = None
        # self.df_vine_n_kgha = None
        # self.df_wholeplant_n_kgha = None
        # self.df_tuber_n_pct = None
        # self.df_vine_n_pct = None
        # self.df_wholeplant_n_pct = None
        # self.df_petiole_no3_ppm = None
        # self.df_cs = None
        # self.df_wx = None

    def _handle_wl_cols(self, c, wl_range, labels_x, prefix='wl_'):
        '''
        Checks for reflectance column validity with a prefix present.

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
            labels_x.append(c)
        # else:
        #     print('Wavelength column <{0}> is not valid. Please note that '
        #           'wavelength columns should begin with <prefix> and the rest '
        #           'must include the wavelength value that can be interpreted '
        #           'as an integer (be sure columns are not decimals).'
        #           ''.format(c))
        return labels_x

    def _get_labels_x(self, group_feats, cols=None):
        '''
        Parses ``group_feats`` and returns a list of column headings so that
        a df can be subset to build the X matrix with only the appropriate
        features indicated by ``group_feats``.
        '''
        labels_x = []
        for key in group_feats:
            print('Loading <group_feats> key: {0}'.format(key))
            if 'wl_range' in key:
                wl_range = group_feats[key]
                assert cols is not None, ('``cols`` must be passed.')
                for c in cols:
                    labels_x = self._handle_wl_cols(c, wl_range, labels_x,
                                                    prefix='wl_')
            elif 'bands' in key or 'weather_derived' in key or 'weather_derived_res' in key:
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
        if 'weather_derived' in group_feats:
            df = self.join_closest_date(  # join weather by closest date
                df, self.weather_derived, left_on='date', right_on='date',
                tolerance=0)
        if 'weather_derived_res' in group_feats:
            df = self.join_closest_date(  # join weather by closest date
                df, self.weather_derived_res, left_on='date', right_on='date',
                tolerance=0)
        for key in group_feats:  # necessary because 'cropscan_wl_range1' must be differentiated
            if 'cropscan' in key:
                df = self.join_closest_date(  # join cropscan by closest date
                    df, self.rs_cropscan_res, left_on='date', right_on='date',
                    tolerance=date_tolerance)
            if 'sentinel' in key:
                df = self.join_closest_date(  # join sentinel by closest date
                    df, self.rs_sentinel, left_on='date',
                    right_on='acquisition_time', tolerance=date_tolerance)
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
        if self.obs_tissue_res is not None:
            obs_tissue = self.obs_tissue_res.copy()
        elif self.obs_tissue is not None:
            obs_tissue = self.obs_tissue.copy()
        else:
            raise ValueError('Both <obs_tissue> and <obs_tisue_res> are None. '
                             'Please be sure either <obs_tissue> or '
                             '<obs_tissue_res> is in <base_dir_data> or '
                             '<db_schema>.')
        if self.obs_tissue_res is not None and self.obs_tissue is not None:
            raise ValueError('Both <obs_tissue> and <obs_tissue_res> are '
                             'populated, so we are unsure which table to '
                             'load. Please be sure only one of <obs_tissue> '
                             'or <obs_tissue_res> is in <base_dir_data> or '
                             '<db_schema>.')
        obs_tissue = obs_tissue[pd.notnull(obs_tissue[value_col])]
        self.df_response = obs_tissue[(obs_tissue[measure_col] == measure) &
                                           (obs_tissue[tissue_col] == tissue)]

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
        tissue_list = self.df_response.groupby(by=[measure_col, tissue_col], as_index=False).first()[tissue_col].tolist()
        measure_list = self.df_response.groupby(by=[measure_col, tissue_col], as_index=False).first()[measure_col].tolist()
        avail_list = ['_'.join(map(str, i)) for i in zip(tissue_list, measure_list)]
        # avail_list = ["vine_n_pct", "pet_no3_ppm", "tuber_n_pct",
        #               "biomass_kgha"]
        msg = ('``tissue``  and ``measure`` must be '
               'one of:\n{0}.\nPlease see "obs_tissue" table to be sure your '
               'intended data are available.'
               ''.format(list(zip(tissue_list, measure_list))))
        assert '_'.join((tissue, measure)) in avail_list, msg

        df = self.df_response[(self.df_response[measure_col] == measure) &
                             (self.df_response[tissue_col] == tissue)]
        return df

    def _stratify_set(self, stratify_cols=['owner', 'farm', 'year'],
                      train_test=None, df=None):
        '''
        Creates a 1-D array of the stratification IDs (to be used by k-fold)
        for both the train and test sets: <stratify_train> and <stratify_test>

        Returns:
            groups (``numpy.ndarray): Array that asssigns each observation to
                a stratification group.
        '''
        if df is None:
            df = self.df_y.copy()
        msg1 = ('All <stratify> strings must be columns in <df_y>')
        for c in stratify_cols:
            assert c in df.columns, msg1
        if train_test is None:
            groups = df.groupby(stratify_cols).ngroup().values
        else:
            groups = df[df['train_test'] == train_test].groupby(
                stratify_cols).ngroup().values

        unique, counts = np.unique(groups, return_counts=True)
        print('\nStratification groups: {0}'.format(stratify_cols))
        print('Number of stratification groups:  {0}'.format(len(unique)))
        print('Minimum number of splits allowed: {0}'.format(min(counts)))
        return groups

    def _check_sklearn_splitter(self, cv_method, cv_method_kwargs,
                                cv_split_kwargs=None, raise_error=False):
        '''
        Checks <cv_method>, <cv_method_kwargs>, and <cv_split_kwargs> for
        continuity.

        Displays a UserWarning or raises ValueError if an invalid parameter
        keyword is provided.

        Parameters:
            raise_error (``bool``): If ``True``, raises a ``ValueError`` if
                parameters do not appear to be available. Otherwise, simply
                issues a warning, and will try to move forward anyways. This
                exists because <inspect.getfullargspec(self.cv_method)[0]> is
                used to get the arguments, but certain scikit-learn functions/
                methods do not expose their arguments to be screened by
                <inspect.getfullargspec>. Thus, the only way to use certain
                splitter functions is to bypass this check.

        Note:
            Does not check for the validity of the keyword argument(s). Also,
            the warnings does not work as fully intended because when
            <inspect.getfullargspec(self.cv_method)[0]> returns an empty list,
            there is no either a warning or ValueError can be raised.
        '''
        if cv_split_kwargs is None:
            cv_split_kwargs = {}

        # import inspect
        # from sklearn.model_selection import RepeatedStratifiedKFold
        # cv_method = RepeatedStratifiedKFold
        # cv_method_kwargs = {'n_splits': 4, 'n_repeats': 3}
        # cv_split_kwargs = None

        cv_method_args = inspect.getfullargspec(cv_method)[0]
        cv_split_args = inspect.getfullargspec(cv_method.split)[0]
        if 'self' in cv_method_args: cv_method_args.remove('self')
        if 'self' in cv_split_args: cv_split_args.remove('self')
        return cv_split_kwargs

        msg1 = ('Some <cv_method_kwargs> parameters do not appear to be '
                'available with the <{0}> function.\nAllowed parameters: {1}\n'
                'Passed to <cv_method_kwargs>: {2}\n\nPlease adjust '
                '<cv_method> and <cv_method_kwargs> so they follow the '
                'requirements of one of the many scikit-learn "splitter '
                'classes". Documentation available at '
                'https://scikit-learn.org/stable/modules/classes.html#splitter-classes.'
                ''.format(cv_method.__name__,
                          cv_method_args,
                          list(cv_method_kwargs.keys())))
        msg2 = ('Some <cv_split_kwargs> parameters are not available with '
                'the <{0}.split()> method.\nAllowed parameters: {1}\nPassed '
                'to <cv_split_kwargs>: {2}\n\nPlease adjust <cv_method>, '
                '<cv_method_kwargs>, and/or <cv_split_kwargs> so they follow '
                'the requirements of one of the many scikit-learn "splitter '
                'classes". Documentation available at '
                'https://scikit-learn.org/stable/modules/classes.html#splitter-classes.'
                ''.format(cv_method.__name__,
                          cv_split_args,
                          list(cv_split_kwargs.keys())))
        if any([i not in inspect.getfullargspec(cv_method)[0]
                for i in cv_method_kwargs]) == True:
            if raise_error:
                raise ValueError(msg1)
            else:
                warnings.warn(msg1, UserWarning)

        if any([i not in inspect.getfullargspec(cv_method.split)[0]
                for i in cv_split_kwargs]) == True:
            if raise_error:
                raise ValueError(msg2)
            else:
                warnings.warn(msg2, UserWarning)

    def _cv_method_check_random_seed(self, cv_method, cv_method_kwargs):
        '''
        If 'random_state' is a valid parameter in <cv_method>, sets from
        <random_seed>.
        '''
        cv_method_args = inspect.getfullargspec(cv_method)[0]
        if 'random_state' in cv_method_args:  # ensure random_seed is set correctly
            cv_method_kwargs['random_state'] = self.random_seed  # if this will get passed to eval(), should be fine since it gets passed to str() first
        return cv_method_kwargs

    def _splitter_eval(self, cv_split_kwargs, df=None):
        '''
        Preps the CV split keyword arguments (evaluates them to variables).
        '''
        if cv_split_kwargs is None:
            cv_split_kwargs = {}
        if 'X' not in cv_split_kwargs and df is not None:  # sets X to <df>
            cv_split_kwargs['X'] = 'df'
        scope = locals()

        if df is None and 'df' in [
                i for i in [a for a in cv_split_kwargs.values()]]:
            raise ValueError(
                '<df> is None, but is present in <cv_split_kwargs>. Please '
                'pass <df> or ajust <cv_split_kwargs>')
        # evaluate any str; keep anything else as is
        cv_split_kwargs_eval = dict(
            (k, eval(str(cv_split_kwargs[k]), scope))
            if isinstance(cv_split_kwargs[k], str)
            else (k, cv_split_kwargs[k])
            for k in cv_split_kwargs)
        return cv_split_kwargs_eval

    def _train_test_split_df(self, df):
        '''
        Splits <df> into train and test sets.

        This function is designed to handle any of the many scikit-learn
        "splitter classes". Documentation available at
        https://scikit-learn.org/stable/modules/classes.html#splitter-classes.
        All parameters used by the <cv_method> function or the
        <cv_method.split> method should be set via
        <cv_method_kwargs> and <cv_split_kwargs>.

        Parameters:
            df (``pandas.DataFrame``): The df to split between train and test
                sets.
            cv_method (``sklearn.model_selection.SplitterClass``): The
                scikit-learn method to use to split into training and test
                groups. In addition to <SplitterClass>(es), <cv_method> can be
                <sklearn.model_selection.train_test_split>, in which case
                <cv_split_kwargs> is ignored and <cv_method_kwargs> should be
                used to pass <cv_method> parameters that will be evaluated via
                the eval() function.
            cv_method_kwargs (``dict``): Keyword arguments to be passed to
                ``cv_method()``.
            cv_split_kwargs (``dict``): Keyword arguments to be passed to
                ``cv_method.split()``. Note that the <X> kwarg defaults to
                ``df`` if not set.

        Note:
            If <n_splits> is set for any <SplitterClass>, it is generally
            ignored. That is, if there are multiple splitting iterations
            (<n_splits> greater than 1), only the first iteration is used to
            split between train and test sets.
        '''
        cv_method = self.cv_method
        cv_method_kwargs = self.cv_method_kwargs
        cv_split_kwargs = self.cv_split_kwargs
        cv_method_kwargs = self._cv_method_check_random_seed(
            cv_method, cv_method_kwargs)

        if cv_method.__name__ == 'train_test_split':
            # Because train_test_split has **kwargs for options, random_state is not caught, so it should be set explicitly
            cv_method_kwargs['random_state'] = self.random_seed
            if 'arrays' in cv_method_kwargs:  # I think can only be <df>?
                df = eval(cv_method_kwargs.pop('arrays', None))
            scope = locals()  # So it understands what <df> is inside func scope
            cv_method_kwargs_eval = dict(
                (k, eval(str(cv_method_kwargs[k]), scope)
                 ) for k in cv_method_kwargs)
            # return
            df_train, df_test = cv_method(df, **cv_method_kwargs_eval)
        else:
            cv_split_kwargs = self._check_sklearn_splitter(
                cv_method, cv_method_kwargs, cv_split_kwargs,
                raise_error=False)
            cv = cv_method(**cv_method_kwargs)
            for key in ['y', 'groups']:
                if key in cv_split_kwargs:
                    if isinstance(cv_split_kwargs[key], list):
                        # assume these are columns to group by and adjust kwargs
                        cv_split_kwargs[key] = self._stratify_set(
                            stratify_cols=cv_split_kwargs[key],
                            train_test=None, df=df)

            # Now cv_split_kwargs should be ready to be evaluated
            cv_split_kwargs_eval = self._splitter_eval(
                cv_split_kwargs, df=df)

            if 'X' not in cv_split_kwargs_eval:  # sets X
                cv_split_kwargs_eval['X'] = df

            train_idx, test_idx = next(cv.split(**cv_split_kwargs_eval))
            df_train, df_test = df.loc[train_idx], df.loc[test_idx]
        print('\nNumber of observations in the "training" set: {0}'.format(len(df_train)))
        print('Number of observations in the "test" set: {0}\n'.format(len(df_test)))

        df_train.insert(0, 'train_test', 'train')
        df_test.insert(0, 'train_test', 'test')
        df_out = df_train.copy()
        df_out = df_out.append(df_test).reset_index(drop=True)
        return df_out

    def _impute_missing_data(self, X, method='iterative'):
        '''
        Imputes missing data in X - sk-learn models will not work with null data

        Parameters:
            method (``str``): should be one of "iterative" (takes more time)
                or "knn" (default: "iterative").
        '''
        if pd.isnull(X).any() is False:
            return X

        if method == 'iterative':
            imp = IterativeImputer(max_iter=10, random_state=self.random_seed)
        elif method == 'knn':
            imp = KNNImputer(n_neighbors=2, weights='uniform')
        elif method == None:
            return X
        X_out = imp.fit_transform(X)
        return X_out
        # if X.shape == X_out.shape:
        #     return X_out  # does not impute if all nan columns (helps debug)
        # else:
        #     return X

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

        # If number of cols are different, then remove from both and update labels_x
        cols_nan_train = df_train.columns[df_train.isnull().all(0)]  # gets columns with all nan
        cols_nan_test = df_test.columns[df_test.isnull().all(0)]
        if len(cols_nan_train) > 0:
            df.drop(cols_nan_train, axis='columns', inplace=True)
            df_train = df[df['train_test'] == 'train']
            df_test = df[df['train_test'] == 'test']
            labels_x = self._get_labels_x(self.group_feats, cols=df.columns)
        if len(cols_nan_test) > 0:
            df.drop(cols_nan_test, axis='columns', inplace=True)
            df_train = df[df['train_test'] == 'train']
            df_test = df[df['train_test'] == 'test']
            labels_x = self._get_labels_x(self.group_feats, cols=df.columns)

        X_train = df_train[labels_x].values
        X_test = df_test[labels_x].values
        y_train = df_train[self.label_y].values
        y_test = df_test[self.label_y].values


        X_train = self._impute_missing_data(X_train, method=impute_method)
        X_test = self._impute_missing_data(X_test, method=impute_method)

        msg = ('There is a different number of columns in <X_train> than in '
               '<X_test>.')
        assert X_train.shape[1] == X_test.shape[1], msg

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

    def _splitter_print(self, splitter, train_test='train'):
        '''
        Checks the proportions of the stratifications in the dataset and prints
        the number of observations in each stratified group. The keys are based
        on the stratified IDs in <stratify_train> or <stratify_test>
        '''
        df_X = self.df_X[self.df_X['self_test'] == train_test]
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
        for train_index, val_index in splitter:
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
            cv_method (``sklearn.model_selection.SplitterClass``): The
                scikit-learn method to use to split into training and test
                groups.
            cv_method_kwargs (``dict``): Keyword arguments to be passed to
                ``cv_method()``.
            cv_split_kwargs (``dict``): Keyword arguments to be passed to
                ``cv_method.split()``. Note that the <X> kwarg defaults to
                ``df`` if not set.
            stratify (``list``): If not None, data is split in a stratified
                fashion, using this as the class labels. Ignored if
                ``cv_method`` is not "stratified". See
                ``sklearn.model_selection.train_test_split()`` documentation
                for more information. Note that if there are less than two
                unique stratified "groups", an error will be raised.
            impute_method (``str``): How to impute missing feature data. Must
                be one of: ["iterative", "knn", None].

        Note:
            This function is designed to handle any of the many scikit-learn
            "splitter classes". Documentation available at
            https://scikit-learn.org/stable/modules/classes.html#splitter-classes.
            All parameters used by the <cv_method> function or the
            <cv_method.split> method should be set via
            <cv_method_kwargs> and <cv_split_kwargs>.

        Example:
            >>> from research_tools import FeatureData
            >>> from research_tools.tests import config

            >>> fd = FeatureData(config_dict=config.config_dict)
            >>> fd.get_feat_group_X_y(test_size=0.1)
            >>> print('Shape of training matrix "X": {0}'.format(fd.X_train.shape))
            >>> print('Shape of training vector "y": {0}'.format(fd.y_train.shape))
            >>> print('Shape of testing matrix "X":  {0}'.format(fd.X_test.shape))
            >>> print('Shape of testing vector "y":  {0}'.format(fd.y_test.shape))
            Shape of training matrix "X": (579, 14)
            Shape of training vector "y": (579,)
            Shape of testing matrix "X":  (65, 14)
            Shape of testing vector "y":  (65,)
        '''
        print('Getting feature data...')
        self._set_params_from_kwargs_fd(**kwargs)

        df = self._get_response_df(self.ground_truth_tissue,
                                   self.ground_truth_measure)
        df = self._join_group_feats(df, self.group_feats, self.date_tolerance)
        df = self._train_test_split_df(df)

        X_train, X_test, y_train, y_test, df = self._get_X_and_y(
            df, impute_method=self.impute_method)

        subset = db_utils.get_primary_keys(df)
        labels_id = subset + ['date', 'train_test']
        self.df_X = df[labels_id + self.labels_x]
        self.df_y = df[labels_id + self.labels_y_id + [self.label_y]]
        self.labels_id = labels_id

        if self.dir_results is not None:
            self._save_df_X_y()

    def get_tuning_splitter(self, **kwargs):
        self._set_params_from_kwargs_fd(**kwargs)

        cv_method = self.cv_method_tune
        cv_method_kwargs = self.cv_method_tune_kwargs
        cv_split_kwargs = self.cv_split_tune_kwargs
        cv_method_kwargs = self._cv_method_check_random_seed(
            cv_method, cv_method_kwargs)

        if cv_method.__name__ == 'train_test_split':
            # Because train_test_split has **kwargs for options, random_state is not caught, so it should be set explicitly
            cv_method_kwargs['random_state'] = self.random_seed
            if 'arrays' in cv_method_kwargs:  # I think can only be <df>?
                df = eval(cv_method_kwargs.pop('arrays', None))
            scope = locals()  # So it understands what <df> is inside func scope
            cv_method_kwargs_eval = dict(
                (k, eval(str(cv_method_kwargs[k]), scope)
                 ) for k in cv_method_kwargs)
            return cv_method(df, **cv_method_kwargs_eval)
        else:
            cv_split_kwargs = self._check_sklearn_splitter(
                cv_method, cv_method_kwargs, cv_split_kwargs,
                raise_error=False)
            self.cv_split_tune_kwargs = cv_split_kwargs
            cv = cv_method(**cv_method_kwargs)
            for key in ['y', 'groups']:
                if key in cv_split_kwargs:
                    if isinstance(cv_split_kwargs[key], list):
                        # assume these are columns to group by and adjust kwargs
                        cv_split_kwargs[key] = self._stratify_set(
                            stratify_cols=cv_split_kwargs[key],
                            train_test='train')

            # Now cv_split_kwargs should be ready to be evaluated
            df_X_train = self.df_X[self.df_X['train_test'] == 'train']
            cv_split_kwargs_eval = self._splitter_eval(
                cv_split_kwargs, df=df_X_train)

            if 'X' not in cv_split_kwargs_eval:  # sets X
                cv_split_kwargs_eval['X'] = df_X_train

        if self.print_splitter_info == True:
            n_train = []
            n_val = []
            for idx_train, idx_val in cv.split(**cv_split_kwargs_eval):
                n_train.append(len(idx_train))
                n_val.append(len(idx_val))
            print('Tuning splitter: number of cross-validation splits: {0}'.format(cv.get_n_splits(**cv_split_kwargs_eval)))
            print('Number of observations in the (tuning) train set (avg): {0:.1f}'.format(np.mean(n_train)))
            print('Number of observations in the (tuning) validation set (avg): {0:.1f}\n'.format(np.mean(n_val)))

        return cv.split(**cv_split_kwargs_eval)
