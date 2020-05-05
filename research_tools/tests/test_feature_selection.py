# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:38:17 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import pytest

from research_tools import feature_groups
from research_tools import FeatureSelection


@pytest.fixture
def test_feature_selection_init_fixture():
    myfs = FeatureSelection(param_dict=feature_groups.param_dict_test, print_out=False)
    return myfs

@pytest.fixture
def test_feature_selection_find_params_fixture():
    myfs = FeatureSelection(param_dict=feature_groups.param_dict_test, print_out=False)
    myfs.fs_find_params()
    return myfs

@pytest.fixture
def test_feature_selection_get_X_select_fixture():
    myfs = FeatureSelection(param_dict=feature_groups.param_dict_test, print_out=False)
    myfs.fs_find_params()
    idx = 2
    X_train_select, X_test_select = myfs.fs_get_X_select(df_fs_params_idx=idx)
    return myfs, X_train_select, X_test_select, idx


class Test_feature_selection_self:
    def test_model_fs_name(self, test_feature_selection_init_fixture):
        myfs = test_feature_selection_init_fixture
        assert type(myfs.model_fs).__name__ == 'Lasso'

    def test_model_fs_params_set(self, test_feature_selection_init_fixture):
        myfs = test_feature_selection_init_fixture
        assert myfs.model_fs_params_set == {'max_iter': 100000, 'selection': 'cyclic', 'warm_start': True}

    def test_model_fs_precompute_not_set(self, test_feature_selection_init_fixture):
        '''
        As long as 'precompute' is set in __init__, "param_dict_test" should
        override it. This tests to be sure this happens in the proper order.
        '''
        myfs = test_feature_selection_init_fixture
        assert myfs.model_fs.get_params()['precompute'] == False

    def test_model_fs_params_adjust_min(self, test_feature_selection_init_fixture):
        myfs = test_feature_selection_init_fixture
        assert myfs.model_fs_params_adjust_min == {'alpha': 1}

    def test_model_fs_params_adjust_max(self, test_feature_selection_init_fixture):
        myfs = test_feature_selection_init_fixture
        assert myfs.model_fs_params_adjust_max == {'alpha': 1e-3}

    def test_n_feats(self, test_feature_selection_init_fixture):
        myfs = test_feature_selection_init_fixture
        assert myfs.n_feats == 5

    def test_n_linspace(self, test_feature_selection_init_fixture):
        myfs = test_feature_selection_init_fixture
        assert myfs.n_linspace == 100

    def test_exit_on_stagnant_n(self, test_feature_selection_init_fixture):
        myfs = test_feature_selection_init_fixture
        assert myfs.exit_on_stagnant_n == 5

    def test_step_pct(self, test_feature_selection_init_fixture):
        myfs = test_feature_selection_init_fixture
        assert myfs.step_pct == 0.01


class Test_feature_selection_find_params:
    def test_find_params_n_feats(self, test_feature_selection_find_params_fixture):
        myfs = test_feature_selection_find_params_fixture
        df = myfs.df_fs_params
        n_feats = df['feat_n'].max()
        msg = ('May fail if convergence is not reached. Try adjusting '
               '<model_fs_params_set>, <n_feats>, <n_linspace>, '
               '<exit_on_stagnant_n>, or <step_pct>.')
        assert myfs.n_feats == n_feats, msg

    # Beef up these tests specific to Lasso feature selection


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
