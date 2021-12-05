# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:42:05 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
from copy import deepcopy
import inspect
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore

from copy import deepcopy
from sklearn.model_selection import train_test_split  # type: ignore

from ...db.db import utilities as db_utils
from ..utils import AnyDataFrame
from .util import cv_method_check_random_seed, check_sklearn_splitter, stratify_set, splitter_eval

from typing import Tuple, List, Set, Optional, Any, Dict
from typing import cast

# TODO: Function to filter cropscan data (e.g., low irradiance, etc.)


def load_df_response(tables               : Dict[str, AnyDataFrame],
                     ground_truth_tissue  : str,
                     ground_truth_measure : str,
                     tissue_col           : str = 'tissue',
                     measure_col          : str = 'measure',
                     value_col            : str = 'value',
                    ) -> Tuple[AnyDataFrame, List[str], str]:
    '''
    Loads the response DataFrame based on <ground_truth_tissue> and
    <ground_truth_measure>. The observations are retrieved from the
    <obs_tissue> table.
    '''
    tissue = ground_truth_tissue
    measure = ground_truth_measure
    print('\nLoading response dataframe...\nTissue: {0}\nMeasure: {1}\n'
          ''.format(tissue, measure))
    if "obs_tissue_res" in tables:
        obs_tissue = tables["obs_tissue_res"].copy()
        if "obs_tissue" in tables:
            raise ValueError('Both <obs_tissue> and <obs_tissue_res> are '
                             'populated, so we are unsure which table to '
                             'load. Please be sure only one of <obs_tissue> '
                             'or <obs_tissue_res> is in <base_dir_data> or '
                             '<db_schema>.')

    elif "obs_tissue" in tables:
        obs_tissue = tables["obs_tissue"].copy()
    else:
        raise ValueError('Both <obs_tissue> and <obs_tisue_res> are None. '
                         'Please be sure either <obs_tissue> or '
                         '<obs_tissue_res> is in <base_dir_data> or '
                         '<db_schema>.')
    obs_tissue = obs_tissue[pd.notnull(obs_tissue[value_col])]
    result = obs_tissue[(obs_tissue[measure_col] == measure) &
                        (obs_tissue[tissue_col] == tissue)]
    labels_y_id = [tissue_col, measure_col]
    label_y = value_col
    return result, labels_y_id, label_y


def _save_df_X_y(dir_results : str,
                 label_y     : str,
                 df_X        : AnyDataFrame,
                 df_y        : AnyDataFrame,
                ) -> None:
    '''
    Saves both ``FeatureData.df_X`` and ``FeatureData.df_y`` to
    ``FeatureData.dir_results``.
    '''
    dir_out = os.path.join(dir_results, label_y)
    os.makedirs(dir_out, exist_ok=True)

    fname_out_X = os.path.join(dir_out, 'data_X_' + label_y + '.csv')
    fname_out_y = os.path.join(dir_out, 'data_y_' + label_y + '.csv')
    df_X.to_csv(fname_out_X, index=False)
    df_y.to_csv(fname_out_y, index=False)



def get_X_and_y(df : AnyDataFrame,
                labels_x : List[str],
                label_y  : str,
                labels_y_id : List[str],
                dir_results : Optional[str],
              ) -> Tuple[AnyDataFrame, AnyDataFrame]:
    subset = db_utils.get_primary_keys(df)
    labels_id = subset + ['date', 'train_test']
    df_X = df[labels_id + labels_x]
    df_y = df[labels_id + labels_y_id + [label_y]]

    if dir_results is not None:
        _save_df_X_y(dir_results, label_y, df_X, df_y)

    return df_X, df_y


def get_tuning_splitter(df                    : AnyDataFrame,
                        df_X                  : AnyDataFrame,
                        cv_method_tune        : Any,
                        cv_method_tune_kwargs : Dict[str, Any],
                        cv_split_tune_kwargs  : Dict[str, Any],
                        random_seed           : int,
                       ) -> Any:
    cv_method = deepcopy(cv_method_tune)
    cv_method_kwargs = deepcopy(cv_method_tune_kwargs)
    cv_split_kwargs = deepcopy(cv_split_tune_kwargs)
    cv_method_kwargs = cv_method_check_random_seed(
                         cv_method, cv_method_kwargs, random_seed)

    if cv_method.__name__ == 'train_test_split':
        cv_method_kwargs['random_state'] = random_seed
        if 'arrays' in cv_method_kwargs:  # I think can only be <df>?
            df = eval(cv_method_kwargs.pop('arrays', None))
        scope = locals()  # So it understands what <df> is inside func scope
        cv_method_kwargs_eval = dict(
            (k, eval(str(cv_method_kwargs[k]), scope)
             ) for k in cv_method_kwargs)
        return cv_method(df, **cv_method_kwargs_eval)
    else:
        cv_split_kwargs = check_sklearn_splitter(
            cv_method, cv_method_kwargs, cv_split_kwargs,
            raise_error=False)
        cv_split_tune_kwargs = cv_split_kwargs
        cv = cv_method(**cv_method_kwargs)
        for key in ['y', 'groups']:
            if key in cv_split_kwargs:
                if isinstance(cv_split_kwargs[key], list):
                    # assume these are columns to group by and adjust kwargs
                    cv_split_kwargs[key] = stratify_set(
                        stratify_cols=cv_split_kwargs[key],
                        train_test='train', df=df)

        # Now cv_split_kwargs should be ready to be evaluated
        df_X_train = df_X[df_X['train_test'] == 'train']
        cv_split_kwargs_eval = splitter_eval(
            cv_split_kwargs, df=df_X_train)

        if 'X' not in cv_split_kwargs_eval:  # sets X
            cv_split_kwargs_eval['X'] = df_X_train

    # TODO: Stderr
    if True:
        n_train = []
        n_val = []
        for idx_train, idx_val in cv.split(**cv_split_kwargs_eval):
            n_train.append(len(idx_train))
            n_val.append(len(idx_val))
        print('Tuning splitter: number of cross-validation splits: {0}'.format(cv.get_n_splits(**cv_split_kwargs_eval)))
        train_pct = (np.mean(n_train) / (np.mean(n_train) + np.mean(n_val))) * 100
        val_pct = (np.mean(n_val) / (np.mean(n_train) + np.mean(n_val))) * 100
        print('Number of observations in the (tuning) train set (avg): {0:.1f} ({1:.1f}%)'.format(np.mean(n_train), train_pct))
        print('Number of observations in the (tuning) validation set (avg): {0:.1f} ({1:.1f}%)\n'.format(np.mean(n_val), val_pct))

    return cv.split(**cv_split_kwargs_eval)

