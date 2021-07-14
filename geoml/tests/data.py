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

from geoml.tests import config

base_dir_data = config.config_dict['Tables']['base_dir_data']
fname_obs_tissue = os.path.join(
    base_dir_data, config.config_dict['Tables']['table_names']['obs_tissue'])
fname_cropscan = os.path.join(
    base_dir_data, config.config_dict['Tables']['table_names']['rs_cropscan'])

df_obs_tissue = pd.read_csv(fname_obs_tissue)

measure_col = 'measure'
tissue_col = 'tissue'

# get all unique combinations of tissue and measure cols
tissue = df_obs_tissue.groupby(by=[measure_col, tissue_col], as_index=False).first()[tissue_col].tolist()
measure = df_obs_tissue.groupby(by=[measure_col, tissue_col], as_index=False).first()[measure_col].tolist()
for tissue, measure in zip(tissue, measure):
    df = df_obs_tissue[(df_obs_tissue[measure_col] == measure) &
                       (df_obs_tissue[tissue_col] == tissue)]
    if tissue == 'tuber' and measure == 'biomdry_Mgha':
        df_tuber_biomdry_Mgha = df.copy()
    elif tissue == 'vine' and measure == 'biomdry_Mgha':
        df_vine_biomdry_Mgha = df.copy()
    elif tissue == 'wholeplant' and measure == 'biomdry_Mgha':
        df_wholeplant_biomdry_Mgha = df.copy()
    elif tissue == 'tuber' and measure == 'biomfresh_Mgha':
        df_tuber_biomfresh_Mgha = df.copy()
    elif tissue == 'canopy' and measure == 'cover_pct':
        df_canopy_cover_pct = df.copy()
    elif tissue == 'tuber' and measure == 'n_kgha':
        df_tuber_n_kgha = df.copy()
    elif tissue == 'vine' and measure == 'n_kgha':
        df_vine_n_kgha = df.copy()
    elif tissue == 'wholeplant' and measure == 'n_kgha':
        df_wholeplant_n_kgha = df.copy()
    elif tissue == 'tuber' and measure == 'n_pct':
        df_tuber_n_pct = df.copy()
    elif tissue == 'vine' and measure == 'n_pct':
        df_vine_n_pct = df.copy()
    elif tissue == 'wholeplant' and measure == 'n_pct':
        df_wholeplant_n_pct = df.copy()
    elif tissue == 'petiole' and measure == 'no3_ppm':
        df_petiole_no3_ppm = df.copy()

df_cs = pd.read_csv(fname_cropscan)
df_cs = df_cs.groupby(['owner', 'study', 'year', 'plot_id', 'date']).mean().reset_index()
