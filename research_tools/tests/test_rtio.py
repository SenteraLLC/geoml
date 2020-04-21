# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:10:47 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import pandas as pd
import pytest
from research_tools import rtio
from research_tools import feature_groups


@pytest.fixture
def test_rtio_init_fixture():
    random_seed = 0
    base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    my_rtio = rtio(base_dir_data, random_seed=random_seed)
    return my_rtio

@pytest.fixture
def test_rtio_get_feat_group_X_y_fixture():
    random_seed = 0
    base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    my_rtio = rtio(base_dir_data, random_seed=random_seed)
    group_feats = feature_groups.cs_test2

    my_rtio.get_feat_group_X_y(
        group_feats, ground_truth='vine_n_pct', date_tolerance=3,
        test_size=0.4, stratify=['study', 'date'], impute_method='iterative')
    return my_rtio


class Test_rtio_exist_df:
    def test_df_pet_no3(self, test_rtio_init_fixture):
        my_rtio = test_rtio_init_fixture
        cols_require = ['study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(my_rtio.df_pet_no3.columns)
        assert len(my_rtio.df_pet_no3) > 1000

    def test_df_vine_n_pct(self, test_rtio_init_fixture):
        my_rtio = test_rtio_init_fixture
        cols_require = ['study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(my_rtio.df_vine_n_pct.columns)
        assert len(my_rtio.df_vine_n_pct) > 500

    def test_df_tuber_n_pct(self, test_rtio_init_fixture):
        my_rtio = test_rtio_init_fixture
        cols_require = ['study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(my_rtio.df_tuber_n_pct.columns)
        assert len(my_rtio.df_tuber_n_pct) > 500

    def test_df_cs(self, test_rtio_init_fixture):
        my_rtio = test_rtio_init_fixture
        cols_require = ['study', 'year', 'plot_id', 'date']
        assert set(cols_require).issubset(my_rtio.df_cs.columns)
        assert len(my_rtio.df_cs.columns) > 8
        assert len(my_rtio.df_cs) > 3


class Test_rtio_self:
    def test_group_feats(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert my_rtio.group_feats['dae'] == 'dae'
        assert my_rtio.group_feats['rate_ntd'] == {'col_rate_n': 'rate_n_kgha', 'col_out': 'rate_ntd_kgha'}
        assert my_rtio.group_feats['cropscan_wl_range1'] == [400, 900]

    def test_ground_truth(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert my_rtio.ground_truth == 'vine_n_pct'

    def test_date_tolerance(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert my_rtio.date_tolerance == 3

    def test_test_size(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert my_rtio.test_size == 0.4

    def test_stratify(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert my_rtio.stratify == ['study', 'date']


class Test_rtio_X_and_y:
    def test_X_train_shape(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert len(my_rtio.X_train.shape) == 2

    def test_X_test_shape(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert len(my_rtio.X_test.shape) == 2

    def test_y_train_shape(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert len(my_rtio.y_train.shape) == 1

    def test_y_test_shape(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert len(my_rtio.y_test.shape) == 1


class Test_rtio_labels:
    def test_labels_id(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        labels_id = ['study', 'year', 'plot_id', 'train_test']
        assert my_rtio.labels_id == labels_id

    def test_labels_x(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        labels_x = ['dae', 'rate_ntd_kgha', '460', '510', '560', '610', '660',
                     '680', '710', '720', '740', '760', '810', '870']
        assert my_rtio.labels_x == labels_x

    def test_labels_y_id(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        labels_y_id = ['tissue', 'measure']
        assert my_rtio.labels_y_id == labels_y_id

    def test_label_y(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert isinstance(my_rtio.label_y, str)
        assert my_rtio.label_y == 'value'


class Test_rtio_df_X_and_df_y:
    def test_train_test_df_X_col(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert 'train_test' in my_rtio.df_X.columns

    def test_train_test_df_y_col(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert 'train_test' in my_rtio.df_y.columns

    def test_train_test_df_X_vals(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert sorted(my_rtio.df_X['train_test'].unique()) == ['test', 'train']

    def test_train_test_df_y_vals(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        assert sorted(my_rtio.df_y['train_test'].unique()) == ['test', 'train']

    def test_train_test_df_X_proportion(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        df = my_rtio.df_X
        n_train = df[df['train_test'] == 'train']['train_test'].count()
        n_test = df[df['train_test'] == 'test']['train_test'].count()
        assert pytest.approx(n_train / (n_train + n_test), 0.01) == 0.6

    def test_train_test_df_y_proportion(self, test_rtio_get_feat_group_X_y_fixture):
        my_rtio = test_rtio_get_feat_group_X_y_fixture
        df = my_rtio.df_y
        n_train = df[df['train_test'] == 'train']['train_test'].count()
        n_test = df[df['train_test'] == 'test']['train_test'].count()
        assert pytest.approx(n_train / (n_train + n_test), 0.01) == 0.6


if __name__ == '__main__':
    '''Test from Python console'''
    pytest.main([__file__])