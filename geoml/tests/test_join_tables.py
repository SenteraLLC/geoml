# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:29:24 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import pandas as pd
from geoml import JoinTables

import pytest
from geoml.tests import config
from geoml.tests import data


@pytest.fixture(scope="class")
def test_data_fixture():
    # data_dir = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    my_join = JoinTables(config_dict=config.config_dict)
    # data = testdata(my_join.base_dir_data)
    return data.df_petiole_no3_ppm, data.df_vine_n_pct, data.df_cs, my_join

@pytest.fixture(scope="class")
def test_data_fixture_pseudo():
    df = pd.DataFrame.from_dict({
        'owner': ['rosen lab'],
        'study': ['refl'],
        'year': [2010],
        'plot_id': [101]})
    df_left = df.copy()
    df_right = df.copy()
    my_join = JoinTables()
    return df_left, df_right, my_join


class Test_join_tables_exist:
    def test_dates(self, test_data_fixture):
        _, _, _, my_join = test_data_fixture
        cols_require = ['owner', 'study','year', 'date_plant', 'date_emerge']
        assert set(cols_require).issubset(my_join.df_dates.columns)
        assert len(my_join.df_dates) > 3

    def test_exp(self, test_data_fixture):
        _, _, _, my_join = test_data_fixture
        cols_require = ['owner', 'study','year', 'plot_id', 'trt_id']
        assert set(cols_require).issubset(my_join.df_exp.columns)
        assert len(my_join.df_exp) > 3

    def test_trt(self, test_data_fixture):
        _, _, _, my_join = test_data_fixture
        cols_require = ['owner', 'study','year', 'trt_id', 'trt_n', 'trt_var', 'trt_irr']
        assert set(cols_require).issubset(my_join.df_trt.columns)
        assert len(my_join.df_trt) > 3

    def test_n_apps(self, test_data_fixture):
        _, _, _, my_join = test_data_fixture
        cols_require = ['owner', 'study','year', 'trt_n', 'date_applied', 'source_n',
                        'rate_n_kgha']
        assert set(cols_require).issubset(my_join.df_n_apps.columns)
        assert len(my_join.df_n_apps) > 3

    def test_n_crf(self, test_data_fixture):
        _, _, _, my_join = test_data_fixture
        cols_require = ['owner', 'study','year', 'date_applied', 'source_n', 'b0', 'b1',
                        'b2']
        assert set(cols_require).issubset(my_join.df_n_crf.columns)
        assert len(my_join.df_n_crf) > 3


class Test_join_tables_dap_dae:
    def test_dap_column(self, test_data_fixture):
        df_petiole_no3_ppm, df_vine_n_pct, df_cs, my_join = test_data_fixture
        assert 'dap' in my_join.dap(df_petiole_no3_ppm).columns

    def test_dae_column(self, test_data_fixture):
        df_petiole_no3_ppm, df_vine_n_pct, df_cs, my_join = test_data_fixture
        assert 'dae' in my_join.dae(data.df_petiole_no3_ppm).columns

    def test_join_closest_date_column(self, test_data_fixture):
        df_petiole_no3_ppm, df_vine_n_pct, df_cs, my_join = test_data_fixture
        assert 'date_delta' in  my_join.join_closest_date(
            df_petiole_no3_ppm, df_cs, left_on='date', right_on='date', tolerance=3).columns

    def test_dap_calc_correct(self, test_data_fixture):
        df_petiole_no3_ppm, df_vine_n_pct, df_cs, my_join = test_data_fixture
        df = pd.DataFrame.from_dict({'owner': ['rosen lab'],
                                     'study': ['nni'],
                                     'year': [2019],
                                     'plot_id': [101],
                                     'date': ['2019-06-25']})
        assert my_join.dap(df)['dap'][0] == 54

    def test_dae_calc_correct(self, test_data_fixture):
        df_petiole_no3_ppm, df_vine_n_pct, df_cs, my_join = test_data_fixture
        df = pd.DataFrame.from_dict({'owner': ['rosen lab'],
                                     'study': ['nni'],
                                     'year': [2019],
                                     'plot_id': [101],
                                     'date': ['2019-06-25']})
        assert my_join.dae(df)['dae'][0] == 33


class Test_join_tables_join_closest:
    def test_join_closest_date_same(self, test_data_fixture_pseudo):
        df_left, df_right, my_join = test_data_fixture_pseudo
        df_left['date'] = ['2019-07-01']
        df_right['date'] = ['2019-07-01']
        df_join = my_join.join_closest_date(
            df_left, df_right, left_on='date', right_on='date', tolerance=3)
        assert df_join.loc[0, 'date_delta'] == 0

    def test_join_closest_date_up1(self, test_data_fixture_pseudo):
        '''Ground truth (left) collected one day before predictor'''
        df_left, df_right, my_join = test_data_fixture_pseudo
        df_left['date'] = ['2019-07-01']
        df_right['date'] = ['2019-06-30']
        df_join = my_join.join_closest_date(
            df_left, df_right, left_on='date', right_on='date', tolerance=3)
        assert df_join.loc[0, 'date_delta'] == 1

    def test_join_closest_date_up2(self, test_data_fixture_pseudo):
        '''Ground truth (left) collected two days before predictor'''
        df_left, df_right, my_join = test_data_fixture_pseudo
        df_left['date'] = ['2019-07-01']
        df_right['date'] = ['2019-06-29']
        df_join = my_join.join_closest_date(
            df_left, df_right, left_on='date', right_on='date', tolerance=3)
        assert df_join.loc[0, 'date_delta'] == 2

    def test_join_closest_date_up3(self, test_data_fixture_pseudo):
        '''Ground truth (left) collected three days before predictor'''
        df_left, df_right, my_join = test_data_fixture_pseudo
        df_left['date'] = ['2019-07-01']
        df_right['date'] = ['2019-06-28']
        df_join = my_join.join_closest_date(
            df_left, df_right, left_on='date', right_on='date', tolerance=3)
        assert df_join.loc[0, 'date_delta'] == 3

    def test_join_closest_date_down2(self, test_data_fixture_pseudo):
        '''Ground truth (left) collected two days after predictor'''
        df_left, df_right, my_join = test_data_fixture_pseudo
        df_left['date'] = ['2019-07-01']
        df_right['date'] = ['2019-07-03']
        df_join = my_join.join_closest_date(
            df_left, df_right, left_on='date', right_on='date', tolerance=3)
        assert df_join.loc[0, 'date_delta'] == -2

    def test_join_closest_date_tol0(self, test_data_fixture_pseudo):
        '''Ground truth collected one day after predictor, with zero tolerance'''
        df_left, df_right, my_join = test_data_fixture_pseudo
        df_left['date'] = ['2019-07-01']
        df_right['date'] = ['2019-07-02']
        df_join = my_join.join_closest_date(
            df_left, df_right, left_on='date', right_on='date', tolerance=0)
        assert len(df_join) == 0

if __name__ == '__main__':
    '''Test from Python console'''
    pytest.main([__file__])