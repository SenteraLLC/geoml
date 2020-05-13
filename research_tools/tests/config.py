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

config_dict = {
    'JoinTables': {
        'base_dir_data': os.path.join(test_dir, 'testdata')},
    'FeatureData': {
        'base_dir_data': os.path.join(test_dir, 'testdata'),
        'random_seed': 999,
        'fname_petiole': 'tissue_petiole_NO3_ppm.csv',
        'fname_total_n': 'tissue_wp_N_pct.csv',
        'fname_cropscan': 'cropscan.csv',
        'dir_results': None,
        'group_feats': cs_test2,
        'ground_truth': 'vine_n_pct',  # 'vine_n_pct', 'pet_no3_ppm'
        'date_tolerance': 3,
        'test_size': 0.4,
        'stratify': ['study', 'date'],
        'impute_method': 'iterative',
        'n_splits': 4,
        'n_repeats': 3,
        'train_test': 'train',
        'print_out_fd': False},
    'FeatureSelection': {
        'model_fs': Lasso(),
        'model_fs_params_set': {'max_iter': 100000, 'selection': 'cyclic', 'warm_start': True},
        'model_fs_params_adjust_min': {'alpha': 1},  # these are initial values to begin
        'model_fs_params_adjust_max': {'alpha': 1e-3},  # the search for the range of parameters
        'n_feats': 5,
        'n_linspace': 100,
        # 'method_alpha_min': 'full',
        'exit_on_stagnant_n': 5,
        'step_pct': 0.1,
        'print_out_fs': True},
    'Training': {
        'regressor': TransformedTargetRegressor(regressor=Lasso(), transformer=PowerTransformer(copy=True, method='yeo-johnson', standardize=True)),
        'regressor_params': {'max_iter': 100000, 'selection': 'cyclic', 'warm_start': True},
        'param_grid': {'alpha': list(np.logspace(-4, 0, 5))},
        'n_jobs_tune': 2,  # this should be chosen with care in context of rest of parallel processing
        'scoring': ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'),
        'refit': 'neg_mean_absolute_error',
        'rank_scoring': 'neg_mean_absolute_error',
        'print_out_train': False}
    }

