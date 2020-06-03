# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:43:09 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import numpy as np
import os
import pandas as pd


class JoinTables(object):
    '''
    Class for joining tables that contain training data. In addition to the
    join, there are functions available to add new columns to the table to
    act as unique features to explain the response variable being predicted.
    '''
    __allowed_params = (
        'base_dir_data')
    def __init__(self, **kwargs):
        '''
        For now, we can just access a base directory that contains all the
        tables (as .csv files), but as the tables become larger and/or more
        complex, it may be a good idea to connect via a database.

        Parameters:
            base_dir_data (``str``): The base directory containing all the tables
                available to be joined.
        '''
        self.base_dir_data = None

        self.cols_require = {
            'df_dates': ['study', 'year', 'date_plant', 'date_emerge'],
            'df_exp': ['study', 'year', 'plot_id', 'rep', 'trt_id'],
            'df_trt': ['study', 'year', 'trt_id', 'trt_n', 'trt_var', 'trt_irr'],
            'df_n_apps': ['study', 'year', 'trt_n', 'date_applied', 'source_n', 'rate_n_kgha'],
            'df_n_crf': ['study', 'year', 'date_applied', 'source_n', 'b0', 'b1', 'b2'],

            'default': ['study', 'year', 'plot_id'],
            'dae': ['study', 'year', 'plot_id', 'date'],
            'dap': ['study', 'year', 'plot_id', 'date'],
            'rate_ntd': ['study', 'year', 'plot_id', 'date']
            }
        self.msg_require = {
            'df_dates': (
                'The following columns are required in ``df_dates``: {0}. '
                'Please check that each of these column names are in '
                '"df_dates.columns".'.format(self.cols_require['df_dates'])),
            'df_exp': (
                'The following columns are required in ``df_exp``: {0}. '
                'Please check that each of these column names are in '
                '"df_exp.columns".'.format(self.cols_require['df_exp'])),
            'df_trt': (
                'The following columns are required in ``df_trt``: {0}. '
                'Please check that each of these column names are in '
                '"df_trt.columns".'.format(self.cols_require['df_trt'])),
            'df_n_apps': (
                'The following columns are required in ``df_n_apps``: {0}. '
                'Please check that each of these column names are in '
                '"df_n_apps.columns".'.format(self.cols_require['df_n_apps'])),
            'df_n_crf': (
                'The following columns are required in ``df_n_crf``: {0}. '
                'Please check that each of these column names are in '
                '"df_n_crf.columns".'.format(self.cols_require['df_n_crf'])),

            'default': ('The following columns are required in ``df``: {0}. '
                     'Please check that each of these column names are in '
                     '"df.columns".'.format(self.cols_require['default'])),
            'dae': ('join_tables.dae() requires the following columns in '
                    '``df``: {0}. Please check that each of these column '
                    'names are in "df.columns".'
                    ''.format(self.cols_require['dae'])),
            'dap': ('join_tables.dap() requires the following columns in '
                    '``df``: {0}. Please check that each of these column '
                    'names are in "df.columns".'
                    ''.format(self.cols_require['dap'])),
            'rate_ntd': (
                'join_tables.rate_ntd() requires the following columns in '
                '``df``: {0}. Please check that each of these column '
                'names are in "df.columns".'
                ''.format(self.cols_require['rate_ntd']))
            }

        self._set_params_from_kwargs_jt(**kwargs)
        self._set_attributes_jt(**kwargs)

        self.load_tables(**kwargs)

    def _set_params_from_dict_jt(self, config_dict):
        '''
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        '''
        if config_dict is not None and 'JoinTables' in config_dict:
            params_jt = config_dict['JoinTables']
        elif config_dict is not None and 'JoinTables' not in config_dict:
            params_jt = config_dict
        else:  # config_dict is None
            return
        for k, v in params_jt.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, v)

    def _set_params_from_kwargs_jt(self, **kwargs):
        '''
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'config_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``FeatureData`` more explicitly.
        '''
        if 'config_dict' in kwargs:
            self._set_params_from_dict_jt(kwargs.get('config_dict'))
        if kwargs is not None:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)

    def _set_attributes_jt(self, **kwargs):
        '''
        Sets any class attribute to ``None`` that will be created in one of the
        user functions
        '''
        self.fnames = None

    def _cr_rate_ntd(self, df):
        '''
        join_tables.rate_ntd() must sum all the N rates before a particular
        date within each study/year/plot_id combination. Therefore, there can
        NOT be duplicate rows of the metadata when using
        join_tables.rate_ntd(). This function ensures that there is not
        duplicate metadata information in df.
        '''
        msg = ('There can NOT be duplicate rows of the metadata in the ``df`` '
               'passed to ``join_tables.rate_ntd()``. Please filter ``df`` so '
               'so there are not duplicate metadata rows.\n\nHint: is ``df`` '
               'in a long format with multiple types of data (e.g., vine N '
               'and tuber N)?\n..or does ``df`` contain subsamples?')
        cols = ['study', 'year', 'plot_id', 'date']
        if df.groupby(cols).size()[0].max() > 1:
            raise AttributeError(msg)

    def _check_requirements(self, df, f=None, date_format='%Y-%m-%d'):
        '''
        Checks that ``df`` has all of the correct columns and that they contain
        the correct data types

        Parameters:
            df (``pandas.DataFrame``): the input DataFrame
            f (``str``): the function calling the _check_requirements()
                function. This is used to access join_tables.msg_require, which
                contains all the messages to be raised if the correct columns
                are not in ``df``. If ``None``, just assumes ``df`` should
                contain ["study", "year", and "plot_id"]
        '''
        if f is None:
            msg = self.msg_require['default']
            cols_require = self.cols_require['default']
        else:
            msg = self.msg_require[f]
            cols_require = self.cols_require[f]

        # if cols_require not in df.columns:
        if not all(i in df.columns for i in cols_require):
            raise AttributeError(msg)

        if 'date' in cols_require:  # must be a string containing only "date"
            df['date'] = pd.to_datetime(df['date'], format=date_format)
        if 'date_plant' in cols_require:
            df['date_plant'] = pd.to_datetime(df['date_plant'], format=date_format)
        if 'date_emerge' in cols_require:
            df['date_emerge'] = pd.to_datetime(df['date_emerge'], format=date_format)
        if 'date_applied' in cols_require:
            df['date_applied'] = pd.to_datetime(df['date_applied'])

        if f == 'rate_ntd':
            self._cr_rate_ntd(df)

        return df

    def _check_requirements_custom(
            self, df, date_cols=['date'], by=['study', 'year', 'plot_id'],
            date_format='%Y-%m-%d'):
        '''
        Checks that ``df`` has all of the correct columns and that they contain
        the correct data types

        Parameters:
            df (``pandas.DataFrame``): the input DataFrame
            f (``str``): the function calling the _check_requirements()
                function. This is used to access join_tables.msg_require, which
                contains all the messages to be raised if the correct columns
                are not in ``df``. If ``None``, just assumes ``df`` should
                contain ["study", "year", and "plot_id"]
        '''
        if not isinstance(date_cols, list):
            date_cols = [date_cols]
        cols_require = by.copy()
        cols_require.extend(date_cols)

        msg = ('The following columns are required in ``df``: {0}. Please '
               'check that each of these column names are in "df.columns".'
               ''.format(cols_require))
        if not all(i in df.columns for i in cols_require):
            raise AttributeError(msg)

        n_dt = len(df.select_dtypes(include=[np.datetime64]).columns)
        if n_dt < len(date_cols):
            for d in date_cols:
                df[d] = pd.to_datetime(df[d], format=date_format)
        return df

        # TODO: check each of the column data types if they must be particular (e.g., datetime)

    def _read_dfs(self, date_format='%Y-%m-%d'):
        '''
        Read in all to dataframe tables and convert date columns to datetime
        '''
        df_dates = pd.read_csv(self.fnames['dates'])
        df_dates = self._check_requirements(df_dates, f='df_dates', date_format=date_format)
        self.df_dates = df_dates

        df_exp = pd.read_csv(self.fnames['experiments'])
        df_exp = self._check_requirements(df_exp, f='df_exp', date_format=date_format)
        self.df_exp = df_exp

        df_trt = pd.read_csv(self.fnames['treatments'])
        df_trt = self._check_requirements(df_trt, f='df_trt', date_format=date_format)
        self.df_trt = df_trt

        df_n_apps = pd.read_csv(self.fnames['n_apps'])
        df_n_apps = self._check_requirements(df_n_apps, f='df_n_apps', date_format=date_format)
        self.df_n_apps = df_n_apps

        df_n_crf = pd.read_csv(self.fnames['n_crf'])
        df_n_crf = self._check_requirements(df_n_crf, f='df_n_crf', date_format=date_format)
        self.df_n_crf = df_n_crf

    def load_tables(self, **kwargs):
        '''
        Loads all of the tables required to take full advantage of the rest of
        the functions in this class

        Parameters:
            base_dir_data (``str``): The base directory containing all the tables
                available to be joined.
        '''
        self._set_params_from_kwargs_jt(**kwargs)

        if self.base_dir_data is not None:
            self.fnames = {
                # 'cropscan': os.path.join(self.base_dir_data, 'cropscan.csv'),
                'dates': os.path.join(self.base_dir_data, 'metadata_dates.csv'),
                'experiments': os.path.join(self.base_dir_data, 'metadata_exp.csv'),
                'treatments': os.path.join(self.base_dir_data, 'metadata_trt.csv'),
                'n_apps': os.path.join(self.base_dir_data, 'metadata_trt_n.csv'),
                'n_crf': os.path.join(self.base_dir_data, 'metadata_trt_n_crf.csv')}
                # 'petiole_no3': os.path.join(self.base_dir_data, 'tissue_petiole_NO3_ppm.csv'),
                # 'total_n': os.path.join(self.base_dir_data, 'tissue_wp_N_pct.csv')}
            self._read_dfs()

        else:
            print('WARNING: ``base_dir_data`` was not passed. Functions may not '
                  'perform as expected.\n')

    def join_closest_date(
            self, df_left, df_right, left_on='date', right_on='date',
            tolerance=3, by=['study', 'year', 'plot_id'], direction='nearest'):
        '''
        Joins ``df_left`` to ``df_right`` by the closest date (after first
        joining by the ``by`` columns).

        Parameters:
            df_left (``pd.DataFrame``):
            df_right (``pd.DataFrame``):
            left_on (``str``):
            right_on (``str``):
            tolerance (``int``): Number of days away to still allow join (if dates
                are greater than ``tolerance``, the join will not occur).
            by (``str`` or ``list``): Match on these columns before performing
                merge operation.
            direction (``str``): Whether to search for prior, subsequent, or
                closest matches.

        Note:
            Parameter names closely follow the pandas.merge_asof function:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_asof.html
        '''
        df_left = self._check_requirements_custom(
            df_left, date_cols=[left_on], by=by, date_format='%Y-%m-%d')
        df_right = self._check_requirements_custom(
            df_right, date_cols=[right_on], by=by, date_format='%Y-%m-%d')

        df_left.sort_values(left_on, inplace=True)
        df_right.sort_values(right_on, inplace=True)
        left_on2 = left_on + '_l'
        right_on2 = right_on + '_r'
        df_join = pd.merge_asof(
            df_left.rename(columns={left_on:left_on2}),
            df_right.rename(columns={right_on:right_on2}),
            left_on=left_on2, right_on=right_on2, by=by,
            tolerance=pd.Timedelta(tolerance, unit='D'), direction=direction)

        idx_delta = df_join.columns.get_loc(left_on2)
        df_join.insert(idx_delta+1, 'date_delta', None)
        df_join['date_delta'] = (df_join[left_on2]-df_join[right_on2]).astype('timedelta64[D]')
        df_join = df_join[pd.notnull(df_join['date_delta'])]
        df_join = df_join.rename(columns={left_on2:left_on})
        df_join = df_join.drop(right_on2, 1)
        return df_join

    def dae(self, df):
        '''
        Adds a days after emergence (DAE) column to df

        Parameters:
            df (``pandas.DataFrame``): The input dataframe to add DAE column
                to. Must have the following column to perform the appropriate
                joins: "study", "year", "plot_id", and "date" (must be able to
                be converted to datetime).

        Example:
            >>> import os
            >>> import pandas as pd
            >>> from research_tools import JoinTables

            >>> base_dir_data = os.path.join(os.getcwd(), 'research_tools', 'tests', 'testdata')
            >>> fname_petiole = os.path.join(base_dir_data, 'tissue_petiole_NO3_ppm.csv')
            >>> my_join = JoinTables(base_dir_data=base_dir_data)
            >>> df_pet_no3 = pd.read_csv(fname_petiole)
            >>> df_pet_no3.head(3)
              study  year  plot_id        date   tissue  measure         value
            0   NNI  2019      101  2019-06-25  Petiole  NO3_ppm  18142.397813
            1   NNI  2019      101  2019-07-09  Petiole  NO3_ppm   2728.023000
            2   NNI  2019      101  2019-07-23  Petiole  NO3_ppm   1588.190000

            >>> df_pet_no3 = my_join.dae(df_pet_no3)
            >>> df_pet_no3.head(3)
              study  year  plot_id       date   tissue  measure         value  dae
            0   NNI  2019      101 2019-06-25  Petiole  NO3_ppm  18142.397813   33
            1   NNI  2019      101 2019-07-09  Petiole  NO3_ppm   2728.023000   47
            2   NNI  2019      101 2019-07-23  Petiole  NO3_ppm   1588.190000   61
        '''
        df = self._check_requirements(df, 'dae')

        df_join = df.merge(self.df_dates, on=['study', 'year'],
                           validate='many_to_one')
        # df_join['dap'] = (df_join['date']-df_join['date_plant']).dt.days
        df_join['dae'] = (df_join['date']-df_join['date_emerge']).dt.days
        df_out = df_join[['study', 'year', 'plot_id', 'date', 'dae']]
        df_out = df.merge(df_out, on=['study', 'year', 'plot_id', 'date'])
        return df_out

    def dap(self, df):
        '''
        Adds a days after planting (DAP) column to df

        Parameters:
            df (``pandas.DataFrame``): The input dataframe to add DAP column
                to. Must have the following column to perform the appropriate
                joins: "study", "year", "plot_id", and "date" (must be able to
                be converted to datetime).

        Example:
            >>> import os
            >>> import pandas as pd
            >>> from research_tools import JoinTables

            >>> base_dir_data = os.path.join(os.getcwd(), 'research_tools', 'tests', 'testdata')
            >>> fname_petiole = os.path.join(base_dir_data, 'tissue_petiole_NO3_ppm.csv')
            >>> my_join = JoinTables(base_dir_data=base_dir_data)
            >>> df_pet_no3 = pd.read_csv(fname_petiole)
            >>> df_pet_no3.head(3)
              study  year  plot_id        date   tissue  measure         value
            0   NNI  2019      101  2019-06-25  Petiole  NO3_ppm  18142.397813
            1   NNI  2019      101  2019-07-09  Petiole  NO3_ppm   2728.023000
            2   NNI  2019      101  2019-07-23  Petiole  NO3_ppm   1588.190000

            >>> df_pet_no3 = my_join.dap(df_pet_no3)
            >>> df_pet_no3.head(3)
              study  year  plot_id       date   tissue  measure         value  dap
            0   NNI  2019      101 2019-06-25  Petiole  NO3_ppm  18142.397813   54
            1   NNI  2019      101 2019-07-09  Petiole  NO3_ppm   2728.023000   68
            2   NNI  2019      101 2019-07-23  Petiole  NO3_ppm   1588.190000   82
        '''
        df = self._check_requirements(df, 'dap')
        df_join = df.merge(self.df_dates, on=['study', 'year'],
                           validate='many_to_one')
        df_join['dap'] = (df_join['date']-df_join['date_plant']).dt.days
        df_out = df_join[['study', 'year', 'plot_id', 'date', 'dap']]
        df_out = df.merge(df_out, on=['study', 'year', 'plot_id', 'date'])
        return df_out

    def rate_ntd(self, df, col_rate_n='rate_n_kgha', unit_str='kgha'):
        '''
        Adds a column "rate_ntd" indicating the amount of N applied up to the
        date in the df['date'] column (not inclusive).

        First joins df_exp to get unique treatment ids for each plot
        Second joins df_trt to get "breakout" treatment ids for "trt_n"
        Third, joins df_n_apps to get date_applied and rate information
        Fourth, calculates the rate N applied to date (sum of N applied before the
        "date" column; "date" must be a column in df_left)

        Parameters:
            df (``pandas.DataFrame``): The input dataframe to add DAP column
                to. Must have the following column to perform the appropriate
                joins: "study", "year", "plot_id", and "date" (must be able to
                be converted to datetime).
            col_rate_n (``str``): the column name in ``df`` that contains N
                rates
            unit_str (``str``): A string to be appended to the new column name.
                If "kgha", the column name will be "rate_ntd_kgha"

        Example:
            >>> import os
            >>> import pandas as pd
            >>> from research_tools import JoinTables

            >>> base_dir_data = os.path.join(os.getcwd(), 'research_tools', 'tests', 'testdata')
            >>> fname_petiole = os.path.join(base_dir_data, 'tissue_petiole_NO3_ppm.csv')
            >>> my_join = JoinTables(base_dir_data=base_dir_data)
            >>> df_pet_no3 = pd.read_csv(fname_petiole)
            >>> df_pet_no3.head(3)
              study  year  plot_id        date   tissue  measure         value
            0   NNI  2019      101  2019-06-25  Petiole  NO3_ppm  18142.397813
            1   NNI  2019      101  2019-07-09  Petiole  NO3_ppm   2728.023000
            2   NNI  2019      101  2019-07-23  Petiole  NO3_ppm   1588.190000

            >>> df_pet_no3 = my_join.rate_ntd(df_pet_no3)
            >>> df_pet_no3.head(3)
              study  year  plot_id        date   tissue  measure         value rate_ntd_kgha
            0   NNI  2019      101  2019-06-25  Petiole  NO3_ppm  18142.397813       156.919
            1   NNI  2019      101  2019-07-09  Petiole  NO3_ppm   2728.023000       156.919
            2   NNI  2019      101  2019-07-23  Petiole  NO3_ppm   1588.190000       179.336
        '''
        df = self._check_requirements(df, 'rate_ntd')
        df_join = df.merge(self.df_exp, on=['study', 'year', 'plot_id'])
        df_join = df_join.merge(
            self.df_trt[['study', 'year', 'trt_id', 'trt_n']],
            on=['study', 'year', 'trt_id'])
        df_join = df_join.merge(
            self.df_n_apps[['study', 'year', 'trt_n', 'date_applied', col_rate_n]],
            on=['study', 'year', 'trt_n'], validate='many_to_many')

        # remove all rows where date_applied is after date
        df_join = df_join[df_join['date'] >= df_join['date_applied']]
        # TODO: open up to dfs that do not contain "tissue" and "measure", such as cropscan data
        # This is required in the first place because there are potentially
        # multiple types of observations (e.g., vine N and tuber N)
        cols_sum = ['study','year', 'plot_id', 'date']
        df_sum = df_join.groupby(cols_sum)[col_rate_n].sum().reset_index()

        col_name = 'rate_ntd_' + str(unit_str)
        df_sum.rename(columns={col_rate_n: col_name}, inplace=True)
        df_out = df.merge(df_sum, on=['study', 'year', 'plot_id', 'date'])
        return df_out
