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

from research_tools.tests import config

base_dir_data = config.config_dict['JoinTables']['base_dir_data']
fname_petiole = os.path.join(
    base_dir_data, config.config_dict['FeatureData']['fname_petiole'])
fname_total_n = os.path.join(
    base_dir_data, config.config_dict['FeatureData']['fname_total_n'])
fname_cropscan = os.path.join(
    base_dir_data, config.config_dict['FeatureData']['fname_cropscan'])

df_pet_no3 = pd.read_csv(fname_petiole)
df_pet_no3 = df_pet_no3[pd.notnull(df_pet_no3['value'])]

df_total_n = pd.read_csv(fname_total_n)
df_total_n = df_total_n[pd.notnull(df_total_n['value'])]
df_vine_n = df_total_n[df_total_n['tissue'] == 'Vine']

df_cs = pd.read_csv(fname_cropscan)
df_cs = df_cs.groupby(['study', 'year', 'plot_id', 'date']).mean().reset_index()
