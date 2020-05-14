# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:38:17 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
from copy import deepcopy
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression

import pytest
from research_tools.tests import config
from research_tools import FeatureSelection


@pytest.fixture
def test_fs_lasso_init_fixture():
    my_config = deepcopy(config.config_dict)
    myfs = FeatureSelection(config_dict=my_config, print_out_fs=False)
    return myfs

@pytest.fixture
def test_fs_pls_init_fixture():
    my_config = deepcopy(config.config_dict)
    myfs = FeatureSelection(config_dict=my_config, model_fs=PLSRegression(),
                            model_fs_params_set=None)
    return myfs

@pytest.fixture
def test_feature_selection_find_params_fixture():
    my_config = deepcopy(config.config_dict)
    myfs = FeatureSelection(config_dict=my_config, print_out_fs=False)
    myfs.fs_find_params()
    return myfs

@pytest.fixture
def test_feature_selection_get_X_select_fixture():
    my_config = deepcopy(config.config_dict)
    myfs = FeatureSelection(config_dict=my_config, print_out_fs=False)
    myfs.fs_find_params()
    idx = 2
    X_train_select, X_test_select = myfs.fs_get_X_select(df_fs_params_idx=idx)
    return myfs, X_train_select, X_test_select, idx

class Test_feature_selection_self:
    def test_model_fs_name(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        assert type(myfs.model_fs).__name__ == 'Lasso'

    def test_model_fs_params_set(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        assert myfs.model_fs_params_set == {'max_iter': 100000, 'selection': 'cyclic', 'warm_start': True}

    def test_model_fs_precompute_not_set(self, test_fs_lasso_init_fixture):
        '''
        As long as 'precompute' is set in __init__, "config_dict" should
        override it. This tests to be sure this happens in the proper order.
        '''
        myfs = test_fs_lasso_init_fixture
        assert myfs.model_fs.get_params()['precompute'] == False

    def test_model_fs_params_adjust_min(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        assert myfs.model_fs_params_adjust_min == {'alpha': 1}

    def test_model_fs_params_adjust_max(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        assert myfs.model_fs_params_adjust_max == {'alpha': 1e-3}

    def test_n_feats(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        assert myfs.n_feats == 5

    def test_n_linspace(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        assert myfs.n_linspace == 100

    def test_exit_on_stagnant_n(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        assert myfs.exit_on_stagnant_n == 5

    def test_step_pct(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        assert myfs.step_pct == 0.1


class Test_feature_selection_find_params:
    def test_find_params_n_feats(self, test_feature_selection_find_params_fixture):
        myfs = test_feature_selection_find_params_fixture
        df = myfs.df_fs_params
        n_feats = df['feat_n'].max()
        msg = ('May fail if convergence is not reached. Try adjusting '
               '<model_fs_params_set>, <n_feats>, <n_linspace>, '
               '<exit_on_stagnant_n>, or <step_pct>.')
        assert myfs.n_feats == n_feats, msg

    def test_find_params_n_feats_4(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        n_feats = 4
        myfs.fs_find_params(n_feats=n_feats, step_pct=0.05, exit_on_stagnant_n=10, print_out_fs=False)
        msg = ('May fail if convergence is not reached. Try adjusting '
               '<model_fs_params_set>, <n_feats>, <n_linspace>, '
               '<exit_on_stagnant_n>, or <step_pct>.')
        assert myfs.n_feats in myfs.df_fs_params['feat_n'], msg
        assert myfs.n_feats == n_feats

    def test_find_params_n_feats_6(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        n_feats = 6
        myfs.fs_find_params(n_feats=n_feats, step_pct=0.05, exit_on_stagnant_n=10, print_out_fs=False)
        msg = ('May fail if convergence is not reached. Try adjusting '
               '<model_fs_params_set>, <n_feats>, <n_linspace>, '
               '<exit_on_stagnant_n>, or <step_pct>.')
        assert myfs.n_feats in myfs.df_fs_params['feat_n'], msg
        assert myfs.n_feats == n_feats

    def test_find_params_n_feats_8(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        n_feats = 8
        myfs.fs_find_params(n_feats=n_feats, step_pct=0.05, exit_on_stagnant_n=10, print_out_fs=False)
        msg = ('May fail if convergence is not reached. Try adjusting '
               '<model_fs_params_set>, <n_feats>, <n_linspace>, '
               '<exit_on_stagnant_n>, or <step_pct>.')
        assert myfs.n_feats in myfs.df_fs_params['feat_n'], msg
        assert myfs.n_feats == n_feats

    def test_find_params_n_feats_10(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        n_feats = 10
        myfs.fs_find_params(n_feats=n_feats, step_pct=0.05, exit_on_stagnant_n=5, print_out_fs=False)
        msg = ('May fail if convergence is not reached. Try adjusting '
               '<model_fs_params_set>, <n_feats>, <n_linspace>, '
               '<exit_on_stagnant_n>, or <step_pct>.')
        assert myfs.n_feats in myfs.df_fs_params['feat_n'], msg
        assert myfs.n_feats == n_feats

    def test_find_params_n_feats_0(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        n_feats = 0
        with pytest.raises(AssertionError):
            myfs.fs_find_params(n_feats=n_feats)

    def test_find_params_n_feats_None(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        n_feats = None
        myfs.fs_find_params(n_feats=n_feats)
        assert myfs.n_feats == 14 == myfs.X_train.shape[1]

    def test_find_params_n_feats_16_over(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        n_feats = 16
        myfs.fs_find_params(n_feats=n_feats)
        assert myfs.n_feats == 14 == myfs.X_train.shape[1]

    def test_find_params_non_lasso_model(self, test_fs_pls_init_fixture):
        myfs = test_fs_pls_init_fixture
        with pytest.raises(NotImplementedError):
            myfs.fs_find_params()


class Test_feature_selection_get_X_select:
    def test_X_train_select_shape(self, test_feature_selection_get_X_select_fixture):
        myfs, X_train_select, X_test_select, idx = test_feature_selection_get_X_select_fixture
        assert len(X_train_select.shape) == 2

    def test_X_test_select_shape(self, test_feature_selection_get_X_select_fixture):
        myfs, X_train_select, X_test_select, idx = test_feature_selection_get_X_select_fixture
        assert len(X_test_select.shape) == 2

    def test_X_train_select_feat_dims(self, test_feature_selection_get_X_select_fixture):
        myfs, X_train_select, X_test_select, idx = test_feature_selection_get_X_select_fixture
        n_feats_select = myfs.df_fs_params.iloc[idx]['feat_n']
        assert X_train_select.shape[1] == n_feats_select

    def test_X_test_select_feat_dims(self, test_feature_selection_get_X_select_fixture):
        myfs, X_train_select, X_test_select, idx = test_feature_selection_get_X_select_fixture
        n_feats_select = myfs.df_fs_params.iloc[idx]['feat_n']
        assert X_test_select.shape[1] == n_feats_select


class Test_feature_selection_X_select_labels_x:
    def test_labels_x_exists(self, test_feature_selection_get_X_select_fixture):
        myfs, X_train_select, X_test_select, idx = test_feature_selection_get_X_select_fixture
        assert myfs.labels_x_select is not None


class Test_feature_selection_set_kwargs:
    def test_set_kwargs_no_base_dir_data(self):
        with pytest.raises(ValueError):
            myfs = FeatureSelection()

    def test_set_kwargs_model_fs(self, test_fs_pls_init_fixture):
        myfs = test_fs_pls_init_fixture
        assert isinstance(myfs.model_fs, PLSRegression)


class Test_feature_selection_find_features_max_edges:
    '''
    For better code coverage, these tests try to execute edge code of the
    find_features_max() function
    '''
    def test_exit_on_stagnant_n(self, test_fs_lasso_init_fixture):
        myfs = test_fs_lasso_init_fixture
        myfs.fs_find_params(n_feats=6, step_pct=0.05, exit_on_stagnant_n=2, print_out_fs=False)
        df = myfs.df_fs_params
        n_feats = df['feat_n'].max()
        assert myfs.n_feats >= n_feats

    def test_overshoot_n_feats(self, test_fs_lasso_init_fixture):
        '''
        These settings should overshoot n_feats; if it fails, try adjusting the
        arguments to fs_find_params().
        '''
        myfs = test_fs_lasso_init_fixture
        myfs.fs_find_params(n_feats=10, step_pct=0.02, exit_on_stagnant_n=6, print_out_fs=False)
        df = myfs.df_fs_params
        n_feats = df['feat_n'].max()
        assert n_feats > myfs.n_feats


class Test_feature_selection_find_features_min_edges:
    '''
    For better code coverage, these tests try to execute edge code of the
    find_features_min() function
    '''
    def test_find_alpha_min(self, test_fs_lasso_init_fixture):
        '''
        Sets model_fs_params_adjust_min so that we have exactly 1 feature
        right away.
        '''
        myfs = test_fs_lasso_init_fixture
        myfs.fs_find_params(model_fs_params_adjust_min={'alpha': 100}, print_out_fs=False)
        df = myfs.df_fs_params
        n_feats = df['feat_n'].max()
        assert myfs.n_feats == n_feats
