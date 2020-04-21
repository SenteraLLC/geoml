# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:10:47 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import os
import pandas as pd
import pytest

from research_tools import feature_data
from research_tools import feature_groups


@pytest.fixture
def test_feature_data_init_fixture():
    random_seed = 0
    base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    feat_data_cs = feature_data(base_dir_data, random_seed=random_seed)
    return feat_data_cs

@pytest.fixture
def test_feature_data_dir_results_fixture(tmp_path):
    random_seed = 0
    base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    dir_results = os.path.join(tmp_path, 'test_feature_data_dir_results')
    feat_data_cs = feature_data(base_dir_data, random_seed=random_seed,
                                dir_results=dir_results)
    return feat_data_cs

@pytest.fixture
def test_feature_data_get_feat_group_X_y_fixture():
    random_seed = 0
    base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    feat_data_cs = feature_data(base_dir_data, random_seed=random_seed)
    group_feats = feature_groups.cs_test2

    feat_data_cs.get_feat_group_X_y(
        group_feats, ground_truth='vine_n_pct', date_tolerance=3,
        test_size=0.4, stratify=['study', 'date'], impute_method='iterative')
    return feat_data_cs

@pytest.fixture
def test_feature_data_get_feat_group_X_y_dir_results_fixture(tmp_path):
    random_seed = 0
    base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    dir_results = os.path.join(tmp_path, 'test_feature_data_dir_results')
    feat_data_cs = feature_data(base_dir_data, random_seed=random_seed,
                                dir_results=dir_results)
    group_feats = feature_groups.cs_test2

    feat_data_cs.get_feat_group_X_y(
        group_feats, ground_truth='vine_n_pct', date_tolerance=3,
        test_size=0.4, stratify=['study', 'date'], impute_method='iterative')
    return feat_data_cs


class Test_feature_data_exist_df:
    def test_df_pet_no3(self, test_feature_data_init_fixture):
        feat_data_cs = test_feature_data_init_fixture
        cols_require = ['study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_pet_no3.columns)
        assert len(feat_data_cs.df_pet_no3) > 1000

    def test_df_vine_n_pct(self, test_feature_data_init_fixture):
        feat_data_cs = test_feature_data_init_fixture
        cols_require = ['study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_vine_n_pct.columns)
        assert len(feat_data_cs.df_vine_n_pct) > 500

    def test_df_tuber_n_pct(self, test_feature_data_init_fixture):
        feat_data_cs = test_feature_data_init_fixture
        cols_require = ['study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_tuber_n_pct.columns)
        assert len(feat_data_cs.df_tuber_n_pct) > 500

    def test_df_cs(self, test_feature_data_init_fixture):
        feat_data_cs = test_feature_data_init_fixture
        cols_require = ['study', 'year', 'plot_id', 'date']
        assert set(cols_require).issubset(feat_data_cs.df_cs.columns)
        assert len(feat_data_cs.df_cs.columns) > 8
        assert len(feat_data_cs.df_cs) > 3


class Test_feature_data_self:
    def test_group_feats(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert feat_data_cs.group_feats['dae'] == 'dae'
        assert feat_data_cs.group_feats['rate_ntd'] == {'col_rate_n': 'rate_n_kgha', 'col_out': 'rate_ntd_kgha'}
        assert feat_data_cs.group_feats['cropscan_wl_range1'] == [400, 900]

    def test_ground_truth(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert feat_data_cs.ground_truth == 'vine_n_pct'

    def test_date_tolerance(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert feat_data_cs.date_tolerance == 3

    def test_test_size(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert feat_data_cs.test_size == 0.4

    def test_stratify(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert feat_data_cs.stratify == ['study', 'date']


class Test_feature_data_X_and_y:
    def test_X_train_shape(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert len(feat_data_cs.X_train.shape) == 2

    def test_X_test_shape(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert len(feat_data_cs.X_test.shape) == 2

    def test_y_train_shape(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert len(feat_data_cs.y_train.shape) == 1

    def test_y_test_shape(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert len(feat_data_cs.y_test.shape) == 1


class Test_feature_data_labels:
    def test_labels_id(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        labels_id = ['study', 'year', 'plot_id', 'train_test']
        assert feat_data_cs.labels_id == labels_id

    def test_labels_x(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        labels_x = ['dae', 'rate_ntd_kgha', '460', '510', '560', '610', '660',
                     '680', '710', '720', '740', '760', '810', '870']
        assert feat_data_cs.labels_x == labels_x

    def test_labels_y_id(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        labels_y_id = ['tissue', 'measure']
        assert feat_data_cs.labels_y_id == labels_y_id

    def test_label_y(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert isinstance(feat_data_cs.label_y, str)
        assert feat_data_cs.label_y == 'value'


class Test_feature_data_df_X_and_df_y:
    def test_train_test_df_X_col(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert 'train_test' in feat_data_cs.df_X.columns

    def test_train_test_df_y_col(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert 'train_test' in feat_data_cs.df_y.columns

    def test_train_test_df_X_vals(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert sorted(feat_data_cs.df_X['train_test'].unique()) == ['test', 'train']

    def test_train_test_df_y_vals(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        assert sorted(feat_data_cs.df_y['train_test'].unique()) == ['test', 'train']

    def test_train_test_df_X_proportion(self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        df = feat_data_cs.df_X
        n_train = df[df['train_test'] == 'train']['train_test'].count()
        n_test = df[df['train_test'] == 'test']['train_test'].count()
        assert pytest.approx(n_train / (n_train + n_test), 0.01) == 0.6

    def test_train_test_df_y_proportion(
            self, test_feature_data_get_feat_group_X_y_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_fixture
        df = feat_data_cs.df_y
        n_train = df[df['train_test'] == 'train']['train_test'].count()
        n_test = df[df['train_test'] == 'test']['train_test'].count()
        assert pytest.approx(n_train / (n_train + n_test), 0.01) == 0.6

class Test_feature_data_dir_results:
    def test_dir_results_new_dir(self, test_feature_data_dir_results_fixture):
        feat_data_cs = test_feature_data_dir_results_fixture
        assert os.path.isdir(feat_data_cs.dir_results)

    def test_dir_results_readme_exists(
            self, test_feature_data_dir_results_fixture):
        feat_data_cs = test_feature_data_dir_results_fixture
        fname_readme = os.path.join(feat_data_cs.dir_results, 'README.txt')
        assert os.path.isfile(fname_readme)

    def test_dir_results_df_X_exists(
            self, test_feature_data_get_feat_group_X_y_dir_results_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_dir_results_fixture
        dir_out = os.path.join(feat_data_cs.dir_results, feat_data_cs.label_y)
        fname_out_X = os.path.join(dir_out, 'data_X_' + feat_data_cs.label_y + '.csv')
        assert os.path.isfile(fname_out_X)

    def test_dir_results_df_y_exists(
            self, test_feature_data_get_feat_group_X_y_dir_results_fixture):
        feat_data_cs = test_feature_data_get_feat_group_X_y_dir_results_fixture
        dir_out = os.path.join(feat_data_cs.dir_results, feat_data_cs.label_y)
        fname_out_y = os.path.join(dir_out, 'data_y_' + feat_data_cs.label_y + '.csv')
        assert os.path.isfile(fname_out_y)


if __name__ == '__main__':
    '''Test from Python console'''
    pytest.main([__file__])