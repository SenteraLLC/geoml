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
import pytest
from research_tools import join_tables
from research_tools import test_data


@pytest.fixture
def test_data_fixture():
    data_dir = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
    my_join = join_tables(data_dir)
    data = test_data(data_dir)
    df_pet_no3, df_vine_n, df_cs = data.df_pet_no3, data.df_vine_n, data.df_cs
    return df_pet_no3, df_vine_n, df_cs, my_join

@pytest.fixture
def test_data_fixture_pseudo():
    df = pd.DataFrame.from_dict({
        'study': ['Refl'],
        'year': [2010],
        'plot_id': [101]})
    df_left = df.copy()
    df_right = df.copy()
    my_join = join_tables()
    return df_left, df_right, my_join


def test_dap_column(test_data_fixture):
    df_pet_no3, df_vine_n, df_cs, my_join = test_data_fixture
    assert 'dap' in my_join.dap(df_pet_no3).columns

def test_dae_column(test_data_fixture):
    df_pet_no3, df_vine_n, df_cs, my_join = test_data_fixture
    assert 'dae' in my_join.dae(df_pet_no3).columns

def test_join_closest_date_column(test_data_fixture):
    df_pet_no3, df_vine_n, df_cs, my_join = test_data_fixture
    assert 'date_delta' in  my_join.join_closest_date(
        df_pet_no3, df_cs, left_on='date', right_on='date', tolerance=3).columns

def test_dap_calc_correct(test_data_fixture):
    df_pet_no3, df_vine_n, df_cs, my_join = test_data_fixture
    df = pd.DataFrame.from_dict({'study': ['NNI'],
                                'year': [2019],
                                'plot_id': [101],
                                'date': ['2019-06-25']})
    assert my_join.dap(df)['dap'][0] == 54

def test_dae_calc_correct(test_data_fixture):
    df_pet_no3, df_vine_n, df_cs, my_join = test_data_fixture
    df = pd.DataFrame.from_dict({'study': ['NNI'],
                                'year': [2019],
                                'plot_id': [101],
                                'date': ['2019-06-25']})
    assert my_join.dae(df)['dae'][0] == 33

def test_join_closest_date_same(test_data_fixture_pseudo):
    df_left, df_right, my_join = test_data_fixture_pseudo
    df_left['date'] = ['2019-07-01']
    df_right['date'] = ['2019-07-01']
    df_join = my_join.join_closest_date(
        df_left, df_right, left_on='date', right_on='date', tolerance=3)
    assert df_join.loc[0, 'date_delta'] == 0

def test_join_closest_date_up1(test_data_fixture_pseudo):
    '''Ground truth (left) collected one day before predictor'''
    df_left, df_right, my_join = test_data_fixture_pseudo
    df_left['date'] = ['2019-07-01']
    df_right['date'] = ['2019-06-30']
    df_join = my_join.join_closest_date(
        df_left, df_right, left_on='date', right_on='date', tolerance=3)
    assert df_join.loc[0, 'date_delta'] == 1

def test_join_closest_date_up2(test_data_fixture_pseudo):
    '''Ground truth (left) collected two days before predictor'''
    df_left, df_right, my_join = test_data_fixture_pseudo
    df_left['date'] = ['2019-07-01']
    df_right['date'] = ['2019-06-29']
    df_join = my_join.join_closest_date(
        df_left, df_right, left_on='date', right_on='date', tolerance=3)
    assert df_join.loc[0, 'date_delta'] == 2

def test_join_closest_date_up3(test_data_fixture_pseudo):
    '''Ground truth (left) collected three days before predictor'''
    df_left, df_right, my_join = test_data_fixture_pseudo
    df_left['date'] = ['2019-07-01']
    df_right['date'] = ['2019-06-28']
    df_join = my_join.join_closest_date(
        df_left, df_right, left_on='date', right_on='date', tolerance=3)
    assert df_join.loc[0, 'date_delta'] == 3

def test_join_closest_date_down2(test_data_fixture_pseudo):
    '''Ground truth (left) collected two days after predictor'''
    df_left, df_right, my_join = test_data_fixture_pseudo
    df_left['date'] = ['2019-07-01']
    df_right['date'] = ['2019-07-03']
    df_join = my_join.join_closest_date(
        df_left, df_right, left_on='date', right_on='date', tolerance=3)
    assert df_join.loc[0, 'date_delta'] == -2

def test_join_closest_date_tol0(test_data_fixture_pseudo):
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