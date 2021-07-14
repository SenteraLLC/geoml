# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:49:40 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
from datetime import datetime
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

from geoml.tests import config
from geoml import FeatureData
import pytest


@pytest.fixture(scope="class")
def test_fd_basic_args():
    base_dir_data = config.config_dict['Tables']['base_dir_data']
    # base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    return base_dir_data

@pytest.fixture(scope="class")
def test_fd_init_fixture():
    base_dir_data = config.config_dict['Tables']['base_dir_data']
    # base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    feat_data_cs = FeatureData(
        config_dict=config.config_dict, base_dir_data=base_dir_data,
        random_seed=0)
    return feat_data_cs

@pytest.fixture(scope="class")
def test_fd_init_config_dict_simple_fixture():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_dict_fd = {
        # 'base_dir_data': os.path.join(test_dir, 'testdata'),
        'random_seed': 999,
        'dir_results': None,
        'group_feats': config.cs_test,
        'ground_truth_tissue': 'vine',  # must coincide with obs_tissue.csv "tissue" column
        'ground_truth_measure': 'n_pct',  # must coincide with obs_tissue.csv "measure" column
        'date_train': datetime.now().date(),  # ignores training data after this date
        'date_tolerance': 3,
        'cv_method': train_test_split,
        'cv_method_kwargs': {'test_size': '0.4', 'stratify': 'df[["owner", "year"]]'},
        'cv_split_kwargs': None,
        'impute_method': 'iterative',
        'train_test': 'train',
        'cv_method_tune': RepeatedStratifiedKFold,
        'cv_method_tune_kwargs': {'n_splits': 4, 'n_repeats': 3},
        'cv_split_tune_kwargs': {'y': ['farm', 'year']},
        'print_out_fd': False,
        'print_splitter_info': False}
    feat_data_cs = FeatureData(
        config_dict=config_dict_fd, random_seed=0)
    return feat_data_cs

@pytest.fixture(scope="function")
def test_fd_dir_results_fixture(tmp_path):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir_data = os.path.join(test_dir, 'testdata')
    dir_results = os.path.join(tmp_path, 'test_feature_data_dir_results')
    feat_data_cs = FeatureData(
        config_dict=config.config_dict, base_dir_data=base_dir_data,
        random_seed=0, dir_results=dir_results)
    return feat_data_cs

@pytest.fixture(scope="class")
def test_fd_get_feat_group_X_y_fixture():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir_data = os.path.join(test_dir, 'testdata')
    feat_data_cs = FeatureData(
        config_dict=config.config_dict, base_dir_data=base_dir_data,
        random_seed=0)
    group_feats = config.cs_test2
    feat_data_cs.get_feat_group_X_y(
        group_feats=group_feats, ground_truth_tissue='vine',
        ground_truth_measure='n_pct', date_tolerance=3,
        test_size=0.4, stratify=['owner', 'study', 'date'], impute_method='iterative')
    return feat_data_cs

@pytest.fixture(scope="function")
def test_fd_get_feat_group_X_y_dir_results_fixture(tmp_path):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir_data = os.path.join(test_dir, 'testdata')
    dir_results = os.path.join(tmp_path, 'test_feature_data_dir_results')
    feat_data_cs = FeatureData(
        config_dict=config.config_dict, base_dir_data=base_dir_data,
        random_seed=0, dir_results=dir_results)
    group_feats = config.cs_test2
    feat_data_cs.get_feat_group_X_y(
        group_feats=group_feats, ground_truth_tissue='vine',
        ground_truth_measure='n_pct', date_tolerance=3,
        test_size=0.4, stratify=['owner', 'study', 'date'], impute_method='iterative')
    return feat_data_cs


class Test_feature_data_exist_df:
    def test_df_pet_no3(self, test_fd_init_fixture):
        feat_data_cs = test_fd_init_fixture
        cols_require = ['owner', 'study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_petiole_no3_ppm.columns)
        assert len(feat_data_cs.df_petiole_no3_ppm) > 1000

    def test_df_vine_n_pct(self, test_fd_init_fixture):
        feat_data_cs = test_fd_init_fixture
        cols_require = ['owner', 'study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_vine_n_pct.columns)
        assert len(feat_data_cs.df_vine_n_pct) > 300

    def test_df_tuber_n_pct(self, test_fd_init_fixture):
        feat_data_cs = test_fd_init_fixture
        cols_require = ['owner', 'study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_tuber_n_pct.columns)
        assert len(feat_data_cs.df_tuber_n_pct) > 300

    def test_df_cs(self, test_fd_init_fixture):
        feat_data_cs = test_fd_init_fixture
        cols_require = ['owner', 'study', 'year', 'plot_id', 'date']
        assert set(cols_require).issubset(feat_data_cs.df_cs.columns)
        assert len(feat_data_cs.df_cs.columns) > 8
        assert len(feat_data_cs.df_cs) > 3

class Test_feature_data_exist_simple_df:
    def test_simple_df_pet_no3(self, test_fd_init_config_dict_simple_fixture):
        feat_data_cs = test_fd_init_config_dict_simple_fixture
        cols_require = ['owner', 'study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_petiole_no3_ppm.columns)
        assert len(feat_data_cs.df_petiole_no3_ppm) > 1000

    def test_simple_df_vine_n_pct(self, test_fd_init_config_dict_simple_fixture):
        feat_data_cs = test_fd_init_config_dict_simple_fixture
        cols_require = ['owner', 'study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_vine_n_pct.columns)
        assert len(feat_data_cs.df_vine_n_pct) > 300

    def test_simple_df_tuber_n_pct(self, test_fd_init_config_dict_simple_fixture):
        feat_data_cs = test_fd_init_config_dict_simple_fixture
        cols_require = ['owner', 'study', 'year', 'plot_id', 'date', 'tissue', 'measure',
                        'value']
        assert set(cols_require).issubset(feat_data_cs.df_tuber_n_pct.columns)
        assert len(feat_data_cs.df_tuber_n_pct) > 300

    def test_simple_df_cs(self, test_fd_init_config_dict_simple_fixture):
        feat_data_cs = test_fd_init_config_dict_simple_fixture
        cols_require = ['owner', 'study', 'year', 'plot_id', 'date']
        assert set(cols_require).issubset(feat_data_cs.df_cs.columns)
        assert len(feat_data_cs.df_cs.columns) > 8
        assert len(feat_data_cs.df_cs) > 3


class Test_feature_data_self:
    def test_group_feats(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert feat_data_cs.group_feats['dae'] == 'dae'
        assert feat_data_cs.group_feats['rate_ntd'] == {'col_rate_n': 'rate_n_kgha', 'col_out': 'rate_ntd_kgha'}
        assert feat_data_cs.group_feats['cropscan_wl_range1'] == [400, 900]

    def test_ground_truth(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert feat_data_cs.ground_truth_tissue=='vine'
        assert feat_data_cs.ground_truth_measure=='n_pct'

    def test_date_tolerance(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert feat_data_cs.date_tolerance == 3

    def test_test_size(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert feat_data_cs.test_size == 0.4

    def test_stratify(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert feat_data_cs.stratify == ['owner', 'study', 'date']


class Test_feature_data_self_other_X_and_y:
    '''Modifies the X and y variables to increase coverage'''
    def test_group_cropscan_bands(self, test_fd_init_fixture):
        feat_data_cs = test_fd_init_fixture
        group_feats = config.cs_test1
        feat_data_cs.get_feat_group_X_y(group_feats=group_feats)
        cs_bands_expected = ['460', '510', '560', '610', '660', '680', '710',
                              '720', '740', '760', '810', '870', '900']
        assert feat_data_cs.group_feats['dap'] == 'dap'
        assert feat_data_cs.group_feats['rate_ntd'] == {'col_rate_n': 'rate_n_kgha', 'col_out': 'rate_ntd_kgha'}
        assert feat_data_cs.group_feats['cropscan_bands'] == cs_bands_expected
        with pytest.raises(KeyError):
            assert feat_data_cs.group_feats['cropscan_wl_range1']

    def test_get_y_pet_no3_ppm(self, test_fd_init_fixture):
        feat_data_cs = test_fd_init_fixture
        group_feats = config.cs_test1
        feat_data_cs.get_feat_group_X_y(group_feats=group_feats,
                                        ground_truth_tissue='petiole',
                                        ground_truth_measure='no3_ppm')
        assert 'petiole' in feat_data_cs.df_y['tissue'].unique()

    def test_get_y_tuber_n_pct(self, test_fd_init_fixture):
        feat_data_cs = test_fd_init_fixture
        group_feats = config.cs_test1
        feat_data_cs.get_feat_group_X_y(group_feats=group_feats,
                                        ground_truth_tissue='tuber',
                                        ground_truth_measure='n_pct')
        assert 'tuber' in feat_data_cs.df_y['tissue'].unique()


class Test_feature_data_X_and_y:
    def test_X_train_shape(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert len(feat_data_cs.X_train.shape) == 2

    def test_X_test_shape(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert len(feat_data_cs.X_test.shape) == 2

    def test_y_train_shape(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert len(feat_data_cs.y_train.shape) == 1

    def test_y_test_shape(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert len(feat_data_cs.y_test.shape) == 1

    def test_get_feat_group_knn(self, test_fd_init_fixture):
        feat_data_cs = test_fd_init_fixture
        group_feats = config.cs_test2
        feat_data_cs.get_feat_group_X_y(
            group_feats=group_feats, ground_truth_tissue='vine',
            ground_truth_measure='n_pct', date_tolerance=3,
            test_size=0.4, stratify=['owner', 'study', 'date'], impute_method='knn')
        assert feat_data_cs.impute_method == 'knn'


class Test_feature_data_labels:
    def test_labels_id(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        labels_id = ['owner', 'study', 'year', 'plot_id', 'date', 'train_test']
        assert feat_data_cs.labels_id == labels_id

    def test_labels_x(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        labels_x = ['dae', 'rate_ntd_kgha', '460', '510', '560', '610', '660',
                      '680', '710', '720', '740', '760', '810', '870']
        assert feat_data_cs.labels_x == labels_x

    def test_labels_y_id(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        labels_y_id = ['tissue', 'measure']
        assert feat_data_cs.labels_y_id == labels_y_id

    def test_label_y(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert isinstance(feat_data_cs.label_y, str)
        assert feat_data_cs.label_y == 'value'


class Test_feature_data_df_X_and_df_y:
    def test_train_test_df_X_col(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert 'train_test' in feat_data_cs.df_X.columns

    def test_train_test_df_y_col(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert 'train_test' in feat_data_cs.df_y.columns

    def test_train_test_df_X_vals(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert sorted(feat_data_cs.df_X['train_test'].unique()) == ['test', 'train']

    def test_train_test_df_y_vals(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        assert sorted(feat_data_cs.df_y['train_test'].unique()) == ['test', 'train']

    def test_train_test_df_X_proportion(self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        df = feat_data_cs.df_X
        n_train = df[df['train_test'] == 'train']['train_test'].count()
        n_test = df[df['train_test'] == 'test']['train_test'].count()
        assert pytest.approx(n_train / (n_train + n_test), 0.01) == 0.6

    def test_train_test_df_y_proportion(
            self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        df = feat_data_cs.df_y
        n_train = df[df['train_test'] == 'train']['train_test'].count()
        n_test = df[df['train_test'] == 'test']['train_test'].count()
        assert pytest.approx(n_train / (n_train + n_test), 0.01) == 0.6

class Test_feature_data_cv_rep_strat:
    def test_cv_rep_strat_prop_75(
            self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        cv_rep_strat = feat_data_cs.kfold_repeated_stratified(
            n_splits=4, n_repeats=3, train_test='train')
        for train_index, val_index in cv_rep_strat:
            train_fold = feat_data_cs.stratify_train[train_index]
            val_fold = feat_data_cs.stratify_train[val_index]
            break
        n = len(train_fold) + len(val_fold)
        assert pytest.approx(len(train_fold) / n, 0.01) == 0.75

    def test_cv_rep_strat_prop_67(
            self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        cv_rep_strat = feat_data_cs.kfold_repeated_stratified(
            n_splits=3, n_repeats=3, train_test='train')
        for train_index, val_index in cv_rep_strat:
            train_fold = feat_data_cs.stratify_train[train_index]
            val_fold = feat_data_cs.stratify_train[val_index]
            break
        n = len(train_fold) + len(val_fold)
        assert pytest.approx(len(train_fold) / n, 0.01) == 0.667

    def test_cv_rep_strat_prop_50(
            self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        cv_rep_strat = feat_data_cs.kfold_repeated_stratified(
            n_splits=2, n_repeats=3, train_test='train', print_out_fd=True)
        for train_index, val_index in cv_rep_strat:
            train_fold = feat_data_cs.stratify_train[train_index]
            val_fold = feat_data_cs.stratify_train[val_index]
            break
        n = len(train_fold) + len(val_fold)
        assert pytest.approx(len(train_fold) / n, 0.01) == 0.50

    def test_cv_rep_strat_n_unique_strats_train(
            self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        cv_rep_strat = feat_data_cs.kfold_repeated_stratified(
            n_splits=2, n_repeats=3, train_test='train')
        for train_index, val_index in cv_rep_strat:
            train_fold = feat_data_cs.stratify_train[train_index]
            val_fold = feat_data_cs.stratify_train[val_index]
            break
        n_strats1 = sorted(np.unique(train_fold))
        n_strats2 = sorted(feat_data_cs.df_X.groupby(feat_data_cs.stratify
                                                      ).ngroup().unique())
        assert n_strats1 == n_strats2

    def test_cv_rep_strat_n_unique_strats_val(
            self, test_fd_get_feat_group_X_y_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_fixture
        cv_rep_strat = feat_data_cs.kfold_repeated_stratified(
            n_splits=2, n_repeats=3, train_test='train')
        for train_index, val_index in cv_rep_strat:
            train_fold = feat_data_cs.stratify_train[train_index]
            val_fold = feat_data_cs.stratify_train[val_index]
            break
        n_strats1 = sorted(np.unique(val_fold))
        n_strats2 = sorted(feat_data_cs.df_X.groupby(feat_data_cs.stratify
                                                      ).ngroup().unique())
        assert n_strats1 == n_strats2


class Test_feature_data_dir_results:
    def test_dir_results_new_dir(self, test_fd_dir_results_fixture):
        feat_data_cs = test_fd_dir_results_fixture
        assert os.path.isdir(feat_data_cs.dir_results)

    def test_dir_results_readme_exists(
            self, test_fd_dir_results_fixture):
        feat_data_cs = test_fd_dir_results_fixture
        fname_readme = os.path.join(feat_data_cs.dir_results, 'README.txt')
        assert os.path.isfile(fname_readme)

    def test_dir_results_df_X_exists(
            self, test_fd_get_feat_group_X_y_dir_results_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_dir_results_fixture
        dir_out = os.path.join(feat_data_cs.dir_results, feat_data_cs.label_y)
        fname_out_X = os.path.join(dir_out, 'data_X_' + feat_data_cs.label_y + '.csv')
        assert os.path.isfile(fname_out_X)

    def test_dir_results_df_y_exists(
            self, test_fd_get_feat_group_X_y_dir_results_fixture):
        feat_data_cs = test_fd_get_feat_group_X_y_dir_results_fixture
        dir_out = os.path.join(feat_data_cs.dir_results, feat_data_cs.label_y)
        fname_out_y = os.path.join(dir_out, 'data_y_' + feat_data_cs.label_y + '.csv')
        assert os.path.isfile(fname_out_y)


class Test_feature_data_set_kwargs:
    def test_set_kwargs_no_base_dir_data(self):
        with pytest.raises(ValueError):
            feat_data_cs = FeatureData()

    def test_set_kwargs_config_dict_none(self, test_fd_basic_args):
        base_dir_data = test_fd_basic_args
        feat_data_cs = FeatureData(config_dict=None, base_dir_data=base_dir_data)
        assert feat_data_cs.base_dir_data == base_dir_data

    def test_set_kwargs_base_dir_data(self, test_fd_basic_args):
        base_dir_data = test_fd_basic_args
        feat_data_cs = FeatureData(base_dir_data=base_dir_data)
        assert feat_data_cs.base_dir_data == base_dir_data

    def test_set_kwargs_random_seed(self, test_fd_basic_args):
        base_dir_data = test_fd_basic_args
        feat_data_cs = FeatureData(base_dir_data=base_dir_data, random_seed=0)
        assert feat_data_cs.random_seed == 0

    def test_set_kwargs_by_config_dict_random_seed(self):
        feat_data_cs = FeatureData(config_dict=config.config_dict)
        random_seed = config.config_dict['FeatureData']['random_seed']
        assert feat_data_cs.random_seed == random_seed

    def test_set_kwargs_config_dict_override_random_seed(self):
        feat_data_cs = FeatureData(config_dict=config.config_dict,
                                    random_seed=0)
        assert feat_data_cs.random_seed == 0

    def test_set_kwargs_get_feat_group_X_y_override_random_seed(self):
        feat_data_cs = FeatureData(config_dict=config.config_dict,
                                    random_seed=0)
        feat_data_cs.get_feat_group_X_y(group_feats=config.cs_test2,
                                        random_seed=100)
        assert feat_data_cs.random_seed == 100

    def test_set_kwargs_kfold_repeated_stratified_override_random_seed(self):
        feat_data_cs = FeatureData(config_dict=config.config_dict,
                                    random_seed=0)
        feat_data_cs.get_feat_group_X_y(group_feats=config.cs_test2)
        cv_rep_strat = feat_data_cs.kfold_repeated_stratified(random_seed=100)
        assert feat_data_cs.random_seed == 100

    def test_set_kwargs_kfold_repeated_stratified_override_train_test(self):
        feat_data_cs = FeatureData(config_dict=config.config_dict,
                                    train_test='train')
        feat_data_cs.get_feat_group_X_y(group_feats=config.cs_test2)
        cv_rep_strat = feat_data_cs.kfold_repeated_stratified(train_test='test')
        assert feat_data_cs.train_test == 'test'


if __name__ == '__main__':
    '''Test from Python console'''
    pytest.main([__file__])