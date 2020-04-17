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

from research_tools import feature_groups
from research_tools import train_prep
from research_tools import join_tables


class rtio(object):
    '''
    Class that provides file management functionality specifically for research
    data that are used for the basic training of supervised regression models.
    This class assists in the loading, filtering, and preparation of research
    data for its use in a supervised regression model.
    '''
    def __init__(self, base_dir_data,
                 fname_petiole='tissue_petiole_NO3_ppm.csv',
                 fname_total_n='tissue_wp_N_pct.csv',
                 fname_cropscan='cropscan.csv'):
        '''
        '''
        self.base_dir_data = base_dir_data
        self.fname_petiole = fname_petiole
        self.fname_total_n = fname_total_n
        self.fname_cropscan = fname_cropscan

        self.df_pet_no3 = None
        self.df_vine_n_pct = None
        self.df_tuber_n_pct = None
        self.df_cs = None

        self._load_tables()
        self.join_info = join_tables(self.base_dir_data)

    def _get_x_labels(self, group_feats, cols=None):
        '''
        Parses ``group_feats`` and returns a list of column headings so that
        a df can be subset to build the X matrix
        '''
        x_labels = []
        for key in group_feats:
            if 'cropscan_wl_range' in key:
                wl_range = group_feats[key]

                assert cols is not None, ('``cols`` must be passed.')
                for c in cols:
                    if (c.isnumeric() and int(c) > wl_range[0] and
                        int(c) < wl_range[1]):
                        x_labels.append(c)
            elif 'cropscan_bands' in key:
                x_labels.extend(group_feats[key])
            elif 'rate_ntd' in key:
                x_labels.append(group_feats[key]['col_out'])
            else:
                x_labels.append(group_feats[key])
        return x_labels

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
        Joins predictors to ``df`` based on the contents of x_lables
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
            y_label = 'value'
            return self.df_vine_n_pct.copy(), y_label
        if ground_truth == 'pet_no3_ppm':
            y_label = 'value'
            return self.df_pet_no3.copy(), y_label
        if ground_truth == 'tuber_n_pct':
            y_label = 'value'
            return self.df_tuber_n_pct.copy(), y_label


    def get_feat_group_X(self, group_feats=feature_groups.cs_test2,
                         ground_truth='vine_n_pct',
                         date_tolerance=3, random_seed=None):
        '''
        Retrieves all the necessary columns in ``group_feats``, then filters and
        manipulates the dataframe and splits into ``X`` and ``y``

        Parameters:
            group_feats (``list`` or ``dict``): The column headings to include in
                the X matrix.
            ground_truth (``str``): Must be one of "vine_n_pct", "pet_no3_ppm",
                or "tuber_n_pct"; dictates which table to access to retrieve
                the relevant training data.
            date_tolerance (``int``): Number of days away to still allow join
                between response data and predictor features (if dates are
                greater than ``date_tolerance``, the join will not occur and
                data will be neglected). Only relevant if predictor features
                were collected on a different day than response features.

        Example:
            >>> from research_tools import rtio

            >>> base_dir_data = 'I:/Shared drives/NSF STTR Phase I – Potato Remote Sensing/Historical Data/Rosen Lab/Small Plot Data/Data'
            >>> my_rt = rtio(base_dir_data)
            >>> my_rt.get_feat_group_X(group_feats=feature_groups.cs_test2, ground_truth='vine_n_pct')
        '''
        df, y_label = self._get_response_df(ground_truth)
        df = self._join_group_feats(df, group_feats, date_tolerance)
        x_labels = self._get_x_labels(group_feats, cols=df.columns)

        # df, x_labels = self._filter_df_bands(df, group_feats)
        X, y, x_labels = train_prep.get_X_and_y(
            df, x_labels, y_label, random_seed=random_seed)
        return X, y, df, x_labels

    def split_by_cs_band_config(df, tissue='Petiole', measure='NO3_ppm', band='1480'):
        df_full = df[(df['tissue']==tissue) & (df['measure']==measure)].dropna(axis=1)
        df_visnir = self.filter_df_bands(df, wl_range=[400, 900])
        df_swir = df[(df['tissue']==tissue) & (df['measure']==measure) & (pd.notnull(df[band]))].dropna(axis=1, how='all')
        df_re = df[(df['tissue']==tissue) & (df['measure']==measure) & (pd.isnull(df[band]))].dropna(axis=1, how='all')
