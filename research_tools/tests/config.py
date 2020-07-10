# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:22:39 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
import numpy as np
import os

from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression


test_dir = os.path.dirname(os.path.abspath(__file__))

cs_test1 = {
    'dap': 'dap',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'cropscan_bands': ['460', '510', '560', '610', '660', '680', '710', '720',
                       '740', '760', '810', '870', '900']}

cs_test2 = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'cropscan_wl_range1': [400, 900]}

cs_test3 = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    # 'cropscan_bands': ['710', '740', '760', '810'],
    'cropscan_wl_range1': [400, 900],
    'wx': ['precip_cumsum_plant_to_date', 'gdd_cumsum_plant_to_date',
           'precip_diff_cumsum_plant_to_date', 'gdd_diff_cumsum_plant_to_date']}
    # 'wx': ['gdd_cumsum_plant_to_date', 'precip_cumsum_plant_to_date',
    #        'et_rs_cumsum_plant_to_date', 'solar_rad_cumsum_plant_to_date',
    #        'gdd_diff_cumsum_plant_to_date', 'precip_diff_cumsum_plant_to_date',
    #        'et_rs_diff_cumsum_plant_to_date',
    #        'solar_rad_diff_cumsum_plant_to_date']}

summer_2020 = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'cropscan_wl_range1': [400, 900],
    'wx': ['gdd_cumsum_plant_to_date', 'precip_cumsum_plant_to_date',
            'et_rs_cumsum_plant_to_date', 'solar_rad_cumsum_plant_to_date',
            'gdd_diff_cumsum_plant_to_date', 'precip_diff_cumsum_plant_to_date',
            'et_rs_diff_cumsum_plant_to_date',
            'solar_rad_diff_cumsum_plant_to_date',
            'temp_diff_cummean_plant_to_date',
            'temp_diff_diff_cummean_plant_to_date',
           # 'gdd_cumsum_emerge_to_date',
           # 'precip_cumsum_emerge_to_date', 'et_rs_cumsum_emerge_to_date',
           # 'solar_rad_cumsum_emerge_to_date', 'gdd_diff_cumsum_emerge_to_date',
           # 'precip_diff_cumsum_emerge_to_date', 'et_rs_diff_cumsum_emerge_to_date',
           # 'solar_rad_diff_cumsum_emerge_to_date',
           # 'temp_diff_cummean_emerge_to_date',
           # 'temp_diff_diff_cummean_emerge_to_date',
           # 'gdd_cumsum_last7',
           # 'precip_cumsum_last7', 'et_rs_cumsum_last7', 'solar_rad_cumsum_last7',
           # 'gdd_diff_cumsum_last7', 'precip_diff_cumsum_last7',
           # 'et_rs_diff_cumsum_last7', 'solar_rad_diff_cumsum_last7',
           # 'temp_diff_cummean_last7',
           # 'temp_diff_diff_cummean_last7',
            'gdd_cumsum_last14', 'precip_cumsum_last14', 'et_rs_cumsum_last14',
            'solar_rad_cumsum_last14'
            'gdd_diff_cumsum_last14',
            'precip_diff_cumsum_last14', 'et_rs_diff_cumsum_last14',
            'solar_rad_diff_cumsum_last14', 'temp_diff_cummean_last14',
            'temp_diff_diff_cummean_last14'
           # 'gdd_cumsum_last28', 'precip_cumsum_last28',
           # 'et_rs_cumsum_last28', 'solar_rad_cumsum_last28',
           # 'gdd_diff_cumsum_last28', 'precip_diff_cumsum_last28',
           # 'et_rs_diff_cumsum_last28', 'solar_rad_diff_cumsum_last28',
           # 'temp_diff_cummean_last28', 'temp_diff_diff_cummean_last28'
            ]}

nni1_diff = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'cropscan_wl_range1': [400, 900],
    'wx': ['gdd_cumsum_plant_to_date',
           'gdd_diff_cumsum_plant_to_date']}

nni2_diff = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'cropscan_wl_range1': [400, 900],
    'wx': ['gdd_cumsum_plant_to_date',
           'gdd_diff_cumsum_plant_to_date',  # 1st
           'precip_cumsum_plant_to_date', 'et_rs_cumsum_plant_to_date',
           'precip_diff_cumsum_plant_to_date', 'et_rs_diff_cumsum_plant_to_date']}

nni3_diff = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'cropscan_wl_range1': [400, 900],
    'wx': ['gdd_cumsum_plant_to_date',
           'gdd_diff_cumsum_plant_to_date',  # 1st
           'precip_cumsum_plant_to_date', 'et_rs_cumsum_plant_to_date',
           'precip_diff_cumsum_plant_to_date', 'et_rs_diff_cumsum_plant_to_date',  # 2nd
           'ipar_cumsum_plant_to_date', 'temp_diff_cummean_bulk_to_date', 'et_rs_cumsum_bulk_to_date', 'solar_rad_cumsum_bulk_to_date',
           'ipar_diff_cumsum_plant_to_date', 'temp_diff_diff_cummean_bulk_to_date', 'et_rs_diff_cumsum_bulk_to_date', 'solar_rad_diff_cumsum_bulk_to_date']}

biomass1_diff = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'wx': ['gdd_cumsum_plant_to_date', 'ipar_cumsum_plant_to_date',
           'gdd_diff_cumsum_plant_to_date', 'ipar_diff_cumsum_plant_to_date']}

biomass2_diff = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'wx': ['gdd_cumsum_plant_to_date', 'ipar_cumsum_plant_to_date',
           'gdd_diff_cumsum_plant_to_date', 'ipar_diff_cumsum_plant_to_date',  # 1st
           'temp_diff_cummean_bulk_to_date', 'et_rs_cumsum_bulk_to_date', 'solar_rad_cumsum_bulk_to_date',
           'temp_diff_diff_cummean_bulk_to_date', 'et_rs_diff_cumsum_bulk_to_date', 'solar_rad_diff_cumsum_bulk_to_date']}

biomass3_diff = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'wx': ['gdd_cumsum_plant_to_date', 'ipar_cumsum_plant_to_date',
           'gdd_diff_cumsum_plant_to_date', 'ipar_diff_cumsum_plant_to_date',  # 1st
           'temp_diff_cummean_bulk_to_date', 'et_rs_cumsum_bulk_to_date', 'solar_rad_cumsum_bulk_to_date',
           'temp_diff_diff_cummean_bulk_to_date', 'et_rs_diff_cumsum_bulk_to_date', 'solar_rad_diff_cumsum_bulk_to_date',  # 2nd
           'precip_cumsum_plant_to_date', 'et_rs_cumsum_plant_to_date',
           'precip_diff_cumsum_plant_to_date', 'et_rs_diff_cumsum_plant_to_date'],
    'cropscan_wl_range1': [400, 900]}

biomass1 = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'wx': ['gdd_cumsum_plant_to_date', 'ipar_cumsum_plant_to_date']}

biomass2 = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'wx': ['gdd_cumsum_plant_to_date', 'ipar_cumsum_plant_to_date',  # 1st
           'temp_diff_cummean_bulk_to_date', 'et_rs_cumsum_bulk_to_date', 'solar_rad_cumsum_bulk_to_date']}

biomass3 = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'wx': ['gdd_cumsum_plant_to_date', 'ipar_cumsum_plant_to_date',  # 1st
           'temp_diff_cummean_bulk_to_date', 'et_rs_cumsum_bulk_to_date', 'solar_rad_cumsum_bulk_to_date',  # 2nd
           'precip_cumsum_plant_to_date', 'et_rs_cumsum_plant_to_date'],
    'cropscan_wl_range1': [400, 900]}

config_dict = {
    'JoinTables': {
        'base_dir_data': os.path.join(test_dir, 'testdata')},
    'FeatureData': {
        'base_dir_data': os.path.join(test_dir, 'testdata'),
        'random_seed': 999,
        'fname_obs_tissue': 'obs_tissue.csv',
        'fname_cropscan': 'rs_cropscan.csv',
        'fname_wx': 'calc_weather.csv',
        'dir_results': None,
        'group_feats': cs_test2,
        'ground_truth_tissue': 'petiole',  # must coincide with obs_tissue.csv "tissue" column
        'ground_truth_measure': 'no3_ppm',  # must coincide with obs_tissue.csv "measure" column
        'date_tolerance': 3,
        'test_size': 0.4,
        'stratify': ['owner', 'study', 'date'],
        'impute_method': 'iterative',
        'n_splits': 4,
        'n_repeats': 3,
        'train_test': 'train',
        'print_out_fd': False},
    'FeatureSelection': {
        'base_dir_data': os.path.join(test_dir, 'testdata'),
        'model_fs': Lasso(),
        'model_fs_params_set': {'max_iter': 100000, 'selection': 'cyclic', 'warm_start': True},
        'model_fs_params_adjust_min': {'alpha': 1},  # these are initial values to begin
        'model_fs_params_adjust_max': {'alpha': 1e-3},  # the search for the range of parameters
        'n_feats': 12,
        'n_linspace': 150,
        # 'exit_on_stagnant_n': 5,
        # 'step_pct': 0.1,
        'print_out_fs': False},
    'Training': {
        'base_dir_data': os.path.join(test_dir, 'testdata'),
        'regressor': TransformedTargetRegressor(regressor=Lasso(), transformer=PowerTransformer(copy=True, method='yeo-johnson', standardize=True)),
        'regressor_params': {'max_iter': 100000, 'selection': 'cyclic', 'warm_start': True},
        'param_grid': {'alpha': list(np.logspace(-4, 0, 5))},
        'n_jobs_tune': 2,  # this should be chosen with care in context of rest of parallel processing
        'scoring': ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'),
        'refit': 'neg_mean_absolute_error',
        'rank_scoring': 'neg_mean_absolute_error',
        'print_out_train': False}
    }

