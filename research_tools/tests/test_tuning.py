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
from research_tools import Tuning

import pytest

@pytest.fixture
def test_tuning_init_fixture():
    my_tune = Tuning(param_dict=feature_groups.param_dict_test, print_out=False)
    return my_tune

@pytest.fixture
def test_tuning_tune_regressor_fixture():
    my_tune = Tuning(param_dict=feature_groups.param_dict_test, print_out=False)
    my_tune.tune_regressor()
    return my_tune


# @pytest.fixture
# def test_tuning_PLSR_fixture():
#     my_tune = Tuning(param_dict=feature_groups.param_dict_test,
#                       regressor=PLSRegression(), print_out=False)
#     return my_tune


class Test_tuning_self:
    def test_kwargs_override_regressor(self):
        '''<regressor_params> must also be set with <regressor>'''
        tune_pls = Tuning(param_dict=feature_groups.param_dict_test,
                          regressor=PLSRegression(), regressor_params=None)
        assert isinstance(tune_pls.regressor, PLSRegression)

    def test_kwargs_override_regressor_params(self):
        regressor_params = {'n_components': 3, 'max_iter': 10000}
        tune_pls = Tuning(param_dict=feature_groups.param_dict_test,
                          regressor=PLSRegression(),
                          regressor_params=regressor_params)
        assert tune_pls.regressor_params == regressor_params

    def test_kwargs_override_param_grid(self):
        param_grid = {'alpha': list(np.logspace(-4, 0, 10))}
        my_tune = Tuning(param_dict=feature_groups.param_dict_test,
                         param_grid=param_grid)
        assert my_tune.param_grid == param_grid

    def test_kwargs_override_n_jobs_tune(self):
        n_jobs_tune = 4
        my_tune = Tuning(param_dict=feature_groups.param_dict_test,
                         n_jobs_tune=n_jobs_tune)
        assert my_tune.n_jobs_tune == n_jobs_tune

    def test_kwargs_override_scoring(self):
        scoring = ('neg_mean_squared_error', 'r2')
        my_tune = Tuning(param_dict=feature_groups.param_dict_test,
                         scoring=scoring)
        assert my_tune.scoring == scoring

    def test_kwargs_override_refit(self):
        scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2')
        refit = scoring[1]
        my_tune = Tuning(param_dict=feature_groups.param_dict_test,
                         refit=refit)
        assert my_tune.refit == refit

    def test_kwargs_override_rank_scoring(self):
        scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2')
        rank_scoring = scoring[1]
        my_tune = Tuning(param_dict=feature_groups.param_dict_test,
                         rank_scoring=rank_scoring)
        assert my_tune.rank_scoring == rank_scoring

    def test_kwargs_override_print_out_tune(self):
        print_out_tune = True
        my_tune = Tuning(param_dict=feature_groups.param_dict_test,
                         print_out_tune=print_out_tune)
        my_tune.tune_regressor()  # for test coverage on print_out_tune=True
        assert my_tune.print_out_tune == print_out_tune


class Test_tuning_self_simple:
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
            'print_out_tune': True}
        my_tune = Tuning(param_dict=param_dict_tune, base_dir_data=base_dir_data)
        assert my_tune.print_out_tune == True

class Test_tuning_df_tune_filtered_scores:
    def test_scoring_len(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        assert len(my_tune.scoring) == 3

    def test_scores0(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
        for prefix in prefixes:
            col = prefix + my_tune.scoring[0]
            assert col in my_tune.df_tune_filtered

    def test_scores1(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
        for prefix in prefixes:
            col = prefix + my_tune.scoring[1]
            assert col in my_tune.df_tune_filtered

    def test_scores2(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
        for prefix in prefixes:
            col = prefix + my_tune.scoring[2]
            assert col in my_tune.df_tune_filtered

    def test_scores_no_rank_scoring(self, test_tuning_tune_regressor_fixture):
        my_tune = Tuning(param_dict=feature_groups.param_dict_test,
                         rank_scoring=None)
        my_tune.tune_regressor()
        assert my_tune.rank_scoring == my_tune.scoring[0]


class Test_tuning_df_tune_filtered_features:
    def test_features_no_duplicates(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        n_rows = len(my_tune.df_tune_filtered['feat_n'])
        n_unique = len(set(my_tune.df_tune_filtered['feat_n']))
        assert n_rows == n_unique

    def test_features_unique_feat_n(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        feats_all = sorted(list(my_tune.df_tune_filtered['feat_n']))
        feats_unique = sorted(list(my_tune.df_tune_filtered['feat_n'].unique()))
        assert feats_all == feats_unique

    def test_features_len_feats_x_select(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        for idx, row in my_tune.df_tune_filtered.iterrows():
            assert row['feat_n'] == len(row['feats_x_select'])

    def test_features_len_rank_n_select(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        for idx, row in my_tune.df_tune_filtered.iterrows():
            assert row['feat_n'] == len(row['rank_x_select'])

class Test_tuning_df_tune_filtered_params:
    def test_df_tune_filtered_params_reg_tune_match1(
            self, test_tuning_tune_regressor_fixture):
        '''"params_regressor" alpha must match "params_tuning" alpha'''
        my_tune = test_tuning_tune_regressor_fixture
        for idx, row in my_tune.df_tune_filtered.iterrows():
            params_reg_alpha = row['params_regressor'][my_tune.regressor_key+'alpha']
            params_tuning_alpha = row['params_tuning'][my_tune.regressor_key+'alpha']
            assert params_reg_alpha == params_tuning_alpha

    def test_df_tune_filtered_set_params_from_params_regressor(
            self, test_tuning_tune_regressor_fixture):
        '''
        If df_tune_filtered parameters are contradicting, ensure set_params()
        has the same result as the 'params_tuning' column.

        Not sure if it's a scikit-learn bug, but when ``set_params()`` is used on a
        nested estimator (e.g., Pipeline or TransformedTargetRegressor) e.g.,
        ``my_tune.regressor.set_params(**{'regressor_alpha': 11})``,
        ``my_tune.regressor.get_params()`` appears to change the first level
        parameter (e.g., 'regressor__alpha'), but NOT the nested regressor
        parameter (e.g., the parameter that shows up via the simple 'regressor'
        key). These tests should ensure that the parameters are being set as
        desired, no matter what the ``get_params()`` function returns (which is
        being stored in <df_tune>).
        '''
        my_tune = test_tuning_tune_regressor_fixture
        for idx, row in my_tune.df_tune_filtered.iterrows():
            alpha_reg1 = row['params_regressor']['regressor'].get_params()['alpha']
            alpha_tuning = row['params_tuning'][my_tune.regressor_key+'alpha']
            if alpha_reg1 == alpha_tuning:
                pass
            else:
                my_tune.regressor.set_params(**row['params_regressor'])
                alpha_reg2 = my_tune.regressor.regressor.get_params()['alpha']
                assert alpha_reg2 == alpha_tuning


class Test_tuning_set_kwargs:
    def test_set_kwargs_no_base_dir_data(self):
        with pytest.raises(ValueError):
            my_tune = Tuning()

    def test_set_kwargs_param_dict_none(self):
        base_dir_data = feature_groups.param_dict_test['FeatureData']['base_dir_data']
        my_tune = Tuning(param_dict=None, base_dir_data=base_dir_data)
        assert my_tune.base_dir_data == base_dir_data

    def test_set_kwargs_regressor(self):
        regressor = TransformedTargetRegressor(
            regressor=PLSRegression(), transformer=PowerTransformer(
                copy=False, method='yeo-johnson', standardize=True))
        regressor_params = {'n_components': 3, 'max_iter': 10000}
        param_grid = {'n_components': list(np.linspace(2, 10, 9, dtype=int)), 'scale': [True, False]}
        my_tune = Tuning(param_dict=feature_groups.param_dict_test,
                         regressor=regressor, regressor_params=regressor_params,
                         param_grid=param_grid)
        my_tune.tune_regressor()
        assert 'PLSRegression' in my_tune.df_tune_filtered['regressor'].unique()

    def test_set_kwargs_regressor_two_models(self, test_tuning_tune_regressor_fixture):
        my_tune = test_tuning_tune_regressor_fixture
        regressor = TransformedTargetRegressor(
            regressor=PLSRegression(), transformer=PowerTransformer(
                copy=False, method='yeo-johnson', standardize=True))
        regressor_params = {'n_components': 3, 'max_iter': 10000}
        param_grid = {'n_components': list(np.linspace(2, 10, 9, dtype=int)), 'scale': [True, False]}
        my_tune.tune_regressor(regressor=regressor,
                               regressor_params=regressor_params,
                               param_grid=param_grid)
        assert 'Lasso' in my_tune.df_tune_filtered['regressor'].unique()
        assert 'PLSRegression' in my_tune.df_tune_filtered['regressor'].unique()

