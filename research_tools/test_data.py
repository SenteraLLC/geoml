# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:31:26 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import os
import pandas as pd


class test_data(object):
    '''
    Class that gets test data and makes available for unit tests
    '''
    def __init__(self, data_dir):
        '''
        '''
        self.data_dir = data_dir
        self.df_pet_no3 = None
        self.df_vine_n = None
        self.df_cs = None

        self._get_data()

    def _get_data(self):
        '''
        Gets all the sample data that can be used for testing
        '''
        fname_petiole = os.path.join(self.data_dir, 'tissue_petiole_NO3_ppm.csv')
        fname_total_n = os.path.join(self.data_dir, 'tissue_wp_N_pct.csv')
        fname_cropscan = os.path.join(self.data_dir, 'cropscan.csv')

        df_pet_no3 = pd.read_csv(fname_petiole)
        df_pet_no3 = df_pet_no3[pd.notnull(df_pet_no3['value'])]

        df_total_n = pd.read_csv(fname_total_n)
        df_total_n = df_total_n[pd.notnull(df_total_n['value'])]
        df_vine_n = df_total_n[df_total_n['tissue'] == 'Vine']

        df_cs = pd.read_csv(fname_cropscan)
        df_cs = df_cs.groupby(['study', 'year', 'plot_id', 'date']).mean().reset_index()

        self.df_pet_no3 = df_pet_no3
        self.df_vine_n = df_vine_n
        self.df_cs = df_cs

