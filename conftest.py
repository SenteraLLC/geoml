# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:31:26 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
import os
import pytest


test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'geoml', 'tests')
data_dir = os.path.join(test_dir, 'testdata')


# @pytest.fixture(scope="session")
# def data_path():
#     return lambda filename: join(data_dir, filename)
