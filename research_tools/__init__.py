# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:43:52 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]

This software contains confidential and proprietary information of Insight
Sensing Corporation and is protected by copyright, trade secret, and other
State and Federal laws. Its receipt or possession does not convey any rights to
reproduce, disclose its contents, or to manufacture, use or sell anything it
may describe. Reproduction, disclosure, or use without specific written
authorization of Insight Sensing Corporation is strictly forbidden.
"""

__copyright__ = ('Copyright (c) Insight Sensing Corporation, 2020. All rights '
                 'reserved.')
__author__ = 'Tyler J. Nigon'
__license__ = ('TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.')
__email__ = 'tyler@insight-sensing.com'

from .join_tables import join_tables
from .rtio import rtio
from .test_data import test_data

name = 'research_tools'
__version__ = '0.0.1'

__all__ = ['join_tables',
           'test_data']
