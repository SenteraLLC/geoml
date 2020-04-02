# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:54:22 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""

import os
import pandas as pd


class batch_cv(object):
    '''
    Class that provides functionality to perform batch cross-validation.
    '''
    def __init__(self):
        '''
        '''

    def hs_grid_search(settings_dict, dir_out=None,
                       fname_out='batch_cv_settings.csv'):
        '''
        Reads ``settings_dict`` and returns a dataframe with all the necessary
        information to execute each specific processing scenario.

        Folder name will be the index of df_grid for each set of outputs, so
        df_grid must be referenced to know which folder corresponds to which
        scenario.

        Parameters:
            settings_dict (``dict``): A dictionary describing all the
                processing scenarios.
            dir_out (``str``): The folder directory to save the resulting
                DataFrame to (default: ``None``).
            fname_out (``str``): Filename to save the resulting DataFrame as
                (default: 'batch_cv_settings.csv').
        '''
        df_grid = pd.DataFrame(columns=settings_dict.keys())
        keys = settings_dict.keys()
        values = (settings_dict[key] for key in keys)
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        for i in combinations:
            data = []
            for col in df_grid.columns:
                data.append(i[col])
            df_temp = pd.DataFrame(data=[data], columns=df_grid.columns)
            df_grid = df_grid.append(df_temp)
        df_grid = df_grid.reset_index(drop=True)
        # if csv is True:
        if dir_out is not None and os.path.isdir(dir_out):
            df_grid.to_csv(os.path.join(dir_out, fname_out), index=False)
        return df_grid