# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:22:39 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
from datetime import datetime
import numpy as np
import os

from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression


test_dir = os.path.dirname(os.path.abspath(__file__))

sentinel_test1 = {
    'dap': 'dap',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                  'col_out': 'rate_ntd_kgha'},
    'sentinel_wl_range': [400, 2200],
    'weather_derived': ['gdd_cumsum_plant_to_date']
    }

cs_test = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'cropscan_wl_range1': [400, 900]}

rosen_trts = {
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

nni = {
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

biomass = {
    'dae': 'dae',
    'rate_ntd': {'col_rate_n': 'rate_n_kgha',
                 'col_out': 'rate_ntd_kgha'},
    'wx': ['gdd_cumsum_plant_to_date', 'ipar_cumsum_plant_to_date',  # 1st
           'temp_diff_cummean_bulk_to_date', 'et_rs_cumsum_bulk_to_date', 'solar_rad_cumsum_bulk_to_date',  # 2nd
           'precip_cumsum_plant_to_date', 'et_rs_cumsum_plant_to_date'],
    'cropscan_wl_range1': [400, 900]}

config_dict = {
    'Tables': {
        'db_name': 'db_test',
        'db_host': 'localhost',
        'db_user': 'postgres',
        'password': None,  # Note: password does not have to be passsed if stored in local keyring
        'db_schema': 'dev_client',
        'db_port': 5432,
        'db': None,
        'base_dir_data': os.path.join(test_dir, 'testdata'),
        'table_names': {  # if not connected to a DB, these should point to files that contain the join data.
            'experiments': 'experiments.geojson',
            'dates_res': 'dates_research.csv',
            'trt': 'trt.csv',
            'trt_n': 'trt_n.csv',
            'trt_n_crf': 'trt_n_crf.csv',
            'obs_tissue_res': 'obs_tissue_research.geojson',
            'obs_soil_res': 'obs_soil_research.geojson',
            'rs_cropscan_res': 'rs_cropscan.csv',
            'field_bounds': 'field_bounds.geojson',
            'dates': 'dates.csv',
            'as_planted': 'as_planted.geojson',
            'n_applications': 'n_applications.geojson',
            'obs_tissue': 'obs_tissue.geojson',
            'obs_soil': 'obs_soil.geojson',
            'rs_sentinel': 'rs_sentinel.geojson',
            'weather': 'weather.csv',
            'weather_derived': 'calc_weather.csv'}
        },
    'FeatureData': {
        'random_seed': 999,
        'dir_results': None,
        'group_feats': sentinel_test1,
        'ground_truth_tissue': 'petiole',  # must coincide with obs_tissue.csv "tissue" column
        'ground_truth_measure': 'no3_ppm',  # must coincide with obs_tissue.csv "measure" column
        'date_tolerance': 3,
        'cv_method': LeavePGroupsOut,
        'cv_method_kwargs': {'n_groups': 1},  # will be passed as ['cv_method'](**['cv_method_kwargs'])
        'cv_split_kwargs': {'groups': 'df["year"] != 2020'},  # explicitly places <groups> == True (e.g., 2018 and 2019) in train set
        # 'cv_method': LeavePGroupsOut,
        # 'cv_method_kwargs': {'n_groups': 1},
        # 'cv_split_kwargs': {'groups': ['year']},  # randomly (?) chooses <n_groups> <groups> (e.g., 1 "year") for test set
        # 'cv_method': ShuffleSplit,
        # 'cv_method_kwargs': {'test_size': 0.4},
        # 'cv_split_kwargs': {'groups': 'df["year"] != 2020'},
        # 'cv_method': train_test_split,
        # 'cv_method_kwargs': {'test_size': '0.4', 'stratify': 'df[["owner", "year"]]'},  # to pass a str, wrap in double quotes
        # 'cv_split_kwargs': None,
        'impute_method': 'iterative',
        # 'kfold_stratify': ['owner', 'year'],
        # 'n_splits': 4,
        # 'n_repeats': 3,
        'train_test': 'train',
        'cv_method_tune': RepeatedStratifiedKFold,
        'cv_method_tune_kwargs': {'n_splits': 4, 'n_repeats': 3},
        'cv_split_tune_kwargs': {'y': ['farm', 'year']},
        # 'cv_method_tune': RepeatedKFold,
        # 'cv_method_tune_kwargs': {'n_splits': 4, 'n_repeats': 2},
        # 'cv_split_tune_kwargs': None,
        # 'cv_method_tune': LeavePGroupsOut, # if method is a "group" method, then split_tune_kwargs should have a "gropus" parameter
        # 'cv_method_tune_kwargs': {'n_groups': 1},
        # 'cv_split_tune_kwargs': {'groups': 'df["year"] != 2019'},
        'print_out_fd': False,
        'print_splitter_info': False},
    'FeatureSelection': {
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
        'regressor': TransformedTargetRegressor(regressor=Lasso(), transformer=PowerTransformer(copy=True, method='yeo-johnson', standardize=True)),
        'regressor_params': {'max_iter': 100000, 'selection': 'cyclic', 'warm_start': True},
        'param_grid': {'alpha': list(np.logspace(-4, 0, 5))},
        'n_jobs_tune': 2,  # this should be chosen with care in context of rest of parallel processing
        'scoring': ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'),
        'refit': 'neg_mean_absolute_error',
        'rank_scoring': 'neg_mean_absolute_error',
        'print_out_train': False},
    'Predict': {
        'date_predict': datetime(2020, 7, 13),
        'gdf_pred': None,  # if left to None, <primary_keys_pred> should be set
        'primary_keys_pred': {'owner': 'css-farms-dalhart',
                              'farm': 'cabrillas',
                              'field_id': 'c-06',
                              'year': 2020},  # year isn't necessary; overwritten by date_predict.year
        'image_search_method': 'past',  # must be one of ['past', 'future', 'nearest']
        }
    }

