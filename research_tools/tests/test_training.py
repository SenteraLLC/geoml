# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:57:57 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor

from research_tools import feature_groups
from research_tools import Training


import pytest

@pytest.fixture
def test_training_init_fixture():
    my_train = Training(param_dict=feature_groups.param_dict_test, print_out=False)
    return my_train

@pytest.fixture
def test_training_train_fixture():
    my_train = Training(param_dict=feature_groups.param_dict_test, print_out=False)
    my_train.train()
    return my_train


# @pytest.fixture
# def test_training_PLSR_fixture():
#     my_train = Training(param_dict=feature_groups.param_dict_test,
#                       regressor=PLSRegression(), print_out=False)
#     return my_train


class Test_training_self:
    def test_kwargs_override_regressor(self):
        '''<regressor_params> must also be set with <regressor>'''
        tune_pls = Training(param_dict=feature_groups.param_dict_test,
                          regressor=PLSRegression(), regressor_params=None)
        assert isinstance(tune_pls.regressor, PLSRegression)

    def test_kwargs_override_regressor_params(self):
        regressor_params = {'n_components': 3, 'max_iter': 10000}
        tune_pls = Training(param_dict=feature_groups.param_dict_test,
                          regressor=PLSRegression(),
                          regressor_params=regressor_params)
        assert tune_pls.regressor_params == regressor_params

    def test_kwargs_override_param_grid(self):
        param_grid = {'alpha': list(np.logspace(-4, 0, 10))}
        my_train = Training(param_dict=feature_groups.param_dict_test,
                         param_grid=param_grid)
        assert my_train.param_grid == param_grid

    def test_kwargs_override_n_jobs_tune(self):
        n_jobs_tune = 4
        my_train = Training(param_dict=feature_groups.param_dict_test,
                         n_jobs_tune=n_jobs_tune)
        assert my_train.n_jobs_tune == n_jobs_tune

    def test_kwargs_override_scoring(self):
        scoring = ('neg_mean_squared_error', 'r2')
        my_train = Training(param_dict=feature_groups.param_dict_test,
                         scoring=scoring)
        assert my_train.scoring == scoring

    def test_kwargs_override_refit(self):
        scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2')
        refit = scoring[1]
        my_train = Training(param_dict=feature_groups.param_dict_test,
                         refit=refit)
        assert my_train.refit == refit

    def test_kwargs_override_rank_scoring(self):
        scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2')
        rank_scoring = scoring[1]
        my_train = Training(param_dict=feature_groups.param_dict_test,
                         rank_scoring=rank_scoring)
        assert my_train.rank_scoring == rank_scoring

    def test_kwargs_override_print_out_train(self):
        print_out_train = True
        my_train = Training(param_dict=feature_groups.param_dict_test,
                           print_out_train=print_out_train)
        my_train.train()  # for test coverage on print_out_train=True
        assert my_train.print_out_train == print_out_train


class Test_training_self_simple:
    def test_kwargs_override_regressor(self):
        '''
        Sets param_dict manually instead of using nested from feature_groups
        '''
        base_dir_data = feature_groups.param_dict_test['FeatureData']['base_dir_data']
        param_dict_tune = {
            'regressor': TransformedTargetRegressor(regressor=Lasso(), transformer=PowerTransformer(copy=False, method='yeo-johnson', standardize=True)),
            'regressor_params': {'max_iter': 100000, 'selection': 'cyclic', 'warm_start': True},
            'param_grid': {'alpha': list(np.logspace(-4, 0, 5))},
            'n_jobs_tune': 2,  # this should be chosen with care in context of rest of parallel processing
            'scoring': ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'),
            'refit': 'neg_mean_absolute_error',
            'rank_scoring': 'neg_mean_absolute_error',
            'print_out_train': True}
        my_train = Training(param_dict=param_dict_tune, base_dir_data=base_dir_data)
        assert my_train.print_out_train == True

class Test_training_df_tune_scores:
    def test_scoring_len(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        assert len(my_train.scoring) == 3

    def test_scores0(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
        for prefix in prefixes:
            col = prefix + my_train.scoring[0]
            assert col in my_train.df_tune

    def test_scores1(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
        for prefix in prefixes:
            col = prefix + my_train.scoring[1]
            assert col in my_train.df_tune

    def test_scores2(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
        for prefix in prefixes:
            col = prefix + my_train.scoring[2]
            assert col in my_train.df_tune

    def test_scores_no_rank_scoring(self, test_training_train_fixture):
        my_train = Training(param_dict=feature_groups.param_dict_test,
                         rank_scoring=None)
        my_train.train()
        assert my_train.rank_scoring == my_train.scoring[0]


class Test_training_df_test:
    def test_df_test_scores(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        prefixes = ['train_', 'test_']
        scoring = ['neg_mae', 'neg_rmse', 'r2']
        for obj_str in scoring:
            for prefix in prefixes:
                col = prefix + obj_str
                assert col in my_train.df_test

    def test_df_test_uid_unique(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        assert list(my_train.df_test['uid']) == list(my_train.df_test['uid'].unique())


class Test_training_df_test_filtered_features:
    def test_features_no_duplicates(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        n_rows = len(my_train.df_test_filtered['feat_n'])
        n_unique = len(set(my_train.df_test_filtered['feat_n']))
        assert n_rows == n_unique

    def test_features_unique_feat_n(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        feats_all = sorted(list(my_train.df_test_filtered['feat_n']))
        feats_unique = sorted(list(my_train.df_test_filtered['feat_n'].unique()))
        assert feats_all == feats_unique

    def test_features_len_feats_x_select(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        for idx, row in my_train.df_test_filtered.iterrows():
            assert row['feat_n'] == len(row['feats_x_select'])

    def test_features_len_rank_n_select(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        for idx, row in my_train.df_test_filtered.iterrows():
            assert row['feat_n'] == len(row['rank_x_select'])

class Test_training_df_tune_params:
    def test_df_tune_params_reg_tune_match1(
            self, test_training_train_fixture):
        '''"params_regressor" alpha must match "params_tuning" alpha'''
        my_train = test_training_train_fixture
        for idx, row in my_train.df_tune.iterrows():
            params_reg_alpha = row['params_regressor'][my_train.regressor_key+'alpha']
            params_tuning_alpha = row['params_tuning'][my_train.regressor_key+'alpha']
            assert params_reg_alpha == params_tuning_alpha

    def test_df_tune_set_params_from_params_regressor(
            self, test_training_train_fixture):
        '''
        If df_tune parameters are contradicting, ensure set_params()
        has the same result as the 'params_tuning' column.

        Not sure if it's a scikit-learn bug, but when ``set_params()`` is used on a
        nested estimator (e.g., Pipeline or TransformedTargetRegressor) e.g.,
        ``my_train.regressor.set_params(**{'regressor_alpha': 11})``,
        ``my_train.regressor.get_params()`` appears to change the first level
        parameter (e.g., 'regressor__alpha'), but NOT the nested regressor
        parameter (e.g., the parameter that shows up via the simple 'regressor'
        key). These tests should ensure that the parameters are being set as
        desired, no matter what the ``get_params()`` function returns (which is
        being stored in <df_tune>).
        '''
        my_train = test_training_train_fixture
        for idx, row in my_train.df_tune.iterrows():
            alpha_reg1 = row['params_regressor']['regressor'].get_params()['alpha']
            alpha_tuning = row['params_tuning'][my_train.regressor_key+'alpha']
            if alpha_reg1 == alpha_tuning:
                pass
            else:
                my_train.regressor.set_params(**row['params_regressor'])
                alpha_reg2 = my_train.regressor.regressor.get_params()['alpha']
                assert alpha_reg2 == alpha_tuning


class Test_training_set_kwargs:
    def test_set_kwargs_no_base_dir_data(self):
        with pytest.raises(ValueError):
            my_train = Training()

    def test_set_kwargs_param_dict_none(self):
        base_dir_data = feature_groups.param_dict_test['FeatureData']['base_dir_data']
        my_train = Training(param_dict=None, base_dir_data=base_dir_data)
        assert my_train.base_dir_data == base_dir_data

    def test_set_kwargs_regressor(self):
        regressor = TransformedTargetRegressor(
            regressor=PLSRegression(), transformer=PowerTransformer(
                copy=False, method='yeo-johnson', standardize=True))
        regressor_params = {'n_components': 3, 'max_iter': 10000}
        param_grid = {'n_components': list(np.linspace(2, 10, 9, dtype=int)), 'scale': [True, False]}
        my_train = Training(param_dict=feature_groups.param_dict_test,
                         regressor=regressor, regressor_params=regressor_params,
                         param_grid=param_grid)
        my_train.train()
        assert 'PLSRegression' in my_train.df_test_filtered['regressor_name'].unique()

    def test_set_kwargs_regressor_two_models(self, test_training_train_fixture):
        my_train = test_training_train_fixture
        regressor = TransformedTargetRegressor(
            regressor=PLSRegression(), transformer=PowerTransformer(
                copy=True, method='yeo-johnson', standardize=True))
        regressor_params = {'n_components': 3, 'max_iter': 10000}
        param_grid = {'n_components': list(np.linspace(2, 5, 4, dtype=int)), 'scale': [True, False]}
        my_train.train(regressor=regressor,
                               regressor_params=regressor_params,
                               param_grid=param_grid)
        assert 'Lasso' in my_train.df_test_filtered['regressor_name'].unique()
        assert 'PLSRegression' in my_train.df_test_filtered['regressor_name'].unique()

