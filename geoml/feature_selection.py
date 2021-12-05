# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:47:40 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
from copy import deepcopy
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.stats import rankdata  # type: ignore
from scipy import optimize  # type: ignore
from sklearn.linear_model import Lasso  # type: ignore
from sklearn.feature_selection import SelectFromModel  # type: ignore
import traceback
import json

from .utils import AnyDataFrame
from typing import Any, Dict, List, Optional, Tuple


def set_model_fs(model_fs            : Any,
                 model_fs_params_set : Dict[str, Any],
                 random_seed         : int,
                ) -> Tuple[Any, str]:
    '''
    Actually initializes the sklearn model based on the model ``str``. If
    the <random_seed> was not included in the <model_fs_params> (or if
    there is a discrepancy), the model's random state is set/reset to avoid
    any discrepancy.
    '''
    if model_fs_params_set is None:
        model_fs_params_set = {}
    model_fs.set_params(**model_fs_params_set)

    if 'regressor' in model_fs.get_params().keys():
        regressor_key_fs = 'regressor__'
        model_fs_name = type(model_fs.regressor).__name__
        try:
            model_fs.set_params(**{regressor_key_fs + 'random_state': random_seed})
        except ValueError:
            print('Invalid parameter <random_state> for estimator, thus '
                  '<random_state> cannot be set.\n')
    else:
        model_fs_name = type(model_fs).__name__
        try:
            model_fs.set_params(**{'random_state': random_seed})
        except ValueError:
            print('Invalid parameter <random_state> for estimator, thus '
                  '<random_state> cannot be set.\n')

    return model_fs, model_fs_name


def _f_feat_n_df(model_fs      : Any,
                 model_fs_name : str,
                 model_params  : Dict[str, Any],
                 X_train       : AnyDataFrame,
                 y_train       : AnyDataFrame,
                 labels_x      : List[str],
                ) -> AnyDataFrame:
    '''
    Uses an ``sklearn`` model to determine the number of features selected
    from a given set of parameters. The parameters to the model should be
    passed via <kwargs>, which will be passed to the sklearn model
    instance.
    '''
    cols = ['model_fs', 'params_fs', 'feat_n', 'feats_x_select',
            'labels_x_select', 'rank_x_select']
    model_fs.set_params(**model_params)
    model_fs.fit(X_train, y_train)
    model_bs = SelectFromModel(model_fs, prefit=True)
    feats = model_bs.get_support(indices=True)
    coefs = model_fs.coef_[feats]  # get ranking coefficients
    feat_ranking = rankdata(-np.abs(coefs), method='min')

    feats_x_select = [labels_x[i] for i in feats]
    data = [model_fs_name, model_fs.get_params(),
            len(feats), tuple(feats), tuple(feats_x_select),
            tuple(feat_ranking)]
    df = pd.DataFrame(data=[data], columns=cols)
    return df

def _params_adjust(config_dict : Dict[str, float],
                   key         : str,
                   increase    : bool  = True,
                   factor      : float = 10,
                  ) -> Dict[str, float]:
    val = config_dict[key]
    if increase is True:
        config_dict[key] = val*factor
    else:
        config_dict[key] = val/factor
    return config_dict


def _f_opt_n_feats(x             : float,
                   n_feats       : int,
                   model_fs      : Any,
                   model_fs_name : str,
                   X_train       : AnyDataFrame,
                   y_train       : AnyDataFrame,
                   labels_x      : List[str],
                  ) -> float:
    '''
    Returns the difference between n selected feats and desired n_feats.

    Zero is desired.
    '''
    df = _f_feat_n_df(model_fs, model_fs_name, {'alpha': x}, X_train, y_train, labels_x)
    feat_n_sel = df['feat_n'].values[0]
    return np.abs(feat_n_sel - n_feats)


def _find_features_max(n_feats       : int,
                       model_fs      : Any,
                       model_fs_name : str,
                       X_train       : AnyDataFrame,
                       y_train       : AnyDataFrame,
                       labels_x      : List[str],
                      ) -> Dict[str, Any]:
    '''
    Finds the model parameter(s) that result in the max n_feats.

    <FeatureSelection.model_fs_params_feats_max> can be passed to
    ``_f_feat_n_df()`` via ``**kwargs`` to return a dataframe with number of
    features, ranking, etc.
    '''
    # TODO: Get bracket min and max to get in ballpark
    alpha : float = 10000
    # TODO: remember _f_opt_n_feats should return 0 at convergence
    while _f_opt_n_feats(alpha, n_feats, model_fs, model_fs_name, X_train, y_train, labels_x) > 1:
        alpha *= 0.1
        if alpha < 1e-4:
            break
        # print(alpha)
    alpha_min = alpha / 10
    alpha_max = alpha
    args = (n_feats, model_fs, model_fs_name, X_train, y_train, labels_x)
    result = optimize.minimize_scalar(
                 _f_opt_n_feats,
                 args = args,
                 bracket=(alpha_min, alpha_max),
                 method='Golden',
                 options={'maxiter': 3},
              )

    model_fs_params_feats_max = {'alpha': alpha_min}
    if result['success'] is True:
        model_fs_params_feats_max = {'alpha': result['x']}
    elif result['success'] is False and result['fun'] == 0:
        model_fs_params_feats_max = {'alpha': result['x']}

    # TODO: Does any of this even get used?
    df = _f_feat_n_df(model_fs, model_fs_name, model_fs_params_feats_max, X_train, y_train, labels_x)
    feat_n_sel = df['feat_n'].values[0]

    # TODO: Log to stderr
    print('Using up to {0} selected features\n'.format(feat_n_sel))

    return model_fs_params_feats_max


def _find_features_min(model_fs      : Any,
                       model_fs_name : str,
                       model_fs_params_adjust_min : Dict[str, float],
                       X_train       : AnyDataFrame,
                       y_train       : AnyDataFrame,
                       labels_x      : List[str],
                      ) -> Dict[str, Any]:
    '''
    Finds the model parameters that will result in having just a single
    feature. <model_fs_params_feats_min> can be passed to ``_f_feat_n_df()``
    via ``**kwargs`` to return a dataframe with number of features,
    ranking, etc.
    '''
    df = _f_feat_n_df(model_fs, model_fs_name, model_fs_params_adjust_min, X_train, y_train, labels_x)
    feat_n_sel = df['feat_n'].values[0]

    if feat_n_sel <= 1:  # the initial value already results in 1 (or 0) feats
        while feat_n_sel <= 1:
            params_last = model_fs_params_adjust_min
            model_fs_params_adjust_min = _params_adjust(
                model_fs_params_adjust_min, key='alpha',
                increase=False, factor=1.2)
            df = _f_feat_n_df(model_fs, model_fs_name, model_fs_params_adjust_min, X_train, y_train, labels_x)
            feat_n_sel = df['feat_n'].values[0]
        model_fs_params_adjust_min = params_last  # set it back to 1 feat
    else:
        while feat_n_sel > 1:
            model_fs_params_adjust_min = _params_adjust(
                model_fs_params_adjust_min, key='alpha',
                increase=True, factor=1.2)
            df = _f_feat_n_df(model_fs, model_fs_name, model_fs_params_adjust_min, X_train, y_train, labels_x)
            feat_n_sel = df['feat_n'].values[0]
    return model_fs_params_adjust_min

def _lasso_fs_df(model_fs      : Any,
                 model_fs_name : str,
                 model_fs_params_adjust_min : Dict[str, float],
                 X_train       : AnyDataFrame,
                 y_train       : AnyDataFrame,
                 labels_x      : List[str],
                 n_feats       : int,
                 n_linspace    : int,
                ) -> AnyDataFrame:
    '''
    Creates a "template" dataframe that provides all the necessary
    information for the ``Tuning`` class to achieve the specific number of
    features determined by the ``FeatureSelection`` class.
    '''
    if n_feats > 1:
        model_fs_params_feats_min = _find_features_min(model_fs, model_fs_name, model_fs_params_adjust_min, X_train, y_train, labels_x)
        model_fs_params_feats_max = _find_features_max(n_feats, model_fs, model_fs_name, X_train, y_train, labels_x)

        params_max = np.log(model_fs_params_feats_max['alpha'])
        params_min = np.log(model_fs_params_feats_min['alpha'])
        param_val_list = list(np.logspace(params_min, params_max,
                                          num=n_linspace, base=np.e))
    else:
        model_fs_params_feats_max = {'alpha': 1e-4}
        model_fs_params_feats_min = {'alpha': 10}
        params_max = np.log(model_fs_params_feats_max['alpha'])
        params_min = np.log(model_fs_params_feats_min['alpha'])
        param_val_list = list(np.logspace(params_min, params_max,
                                          num=n_linspace, base=np.e) / 10)

    # minimization is the first point where minimum is reached; thus, when
    # using with logspace, it may not quite reach the max feats desired
    if _f_opt_n_feats(param_val_list[-1], n_feats, model_fs, model_fs_name, X_train, y_train, labels_x) != 0:
        param_val_list[-1] = model_fs_params_feats_max['alpha']

    param_adjust_temp = model_fs_params_feats_min.copy()
    param_adjust_temp['alpha'] = param_val_list[0]
    df = _f_feat_n_df(model_fs, model_fs_name, param_adjust_temp, X_train, y_train, labels_x)

    for val in param_val_list[1:]:
        param_adjust_temp['alpha'] = val
        df_temp = _f_feat_n_df(model_fs, model_fs_name, param_adjust_temp, X_train, y_train, labels_x)
        df = df.append(df_temp)

    df = df.drop_duplicates(subset=['feats_x_select'], ignore_index=True)
    msg = ('The alpha value that achieves selection of {0} feature(s) was '
           'not found. Instead, the max number of features to use will '
           'be: {1}')
    if n_feats < df['feat_n'].max():
        df = df[df['feat_n'] <= n_feats]
    if n_feats not in df['feat_n'].to_list():
        print(msg.format(n_feats, df['feat_n'].max()))

    return df

def fs_find_params(X_train       : AnyDataFrame,
                   y_train       : AnyDataFrame,
                   model_fs      : Any,
                   model_fs_name : str,
                   labels_x      : List[str],
                   n_feats       : int,
                   n_linspace    : int,
                   model_fs_params_adjust_min : Dict[str, float],
                  ) -> AnyDataFrame:
    '''
    Constructs a dataframe (df_fs_params) that contains all the information
    necessary to recreate unique feature selection scenarios (i.e.,
    duplicate feature list/feature ranking entries are dropped).

    The goal of this function is to cover the full range of features from 1
    to all features (max number of features varies by dataset), and be able
    to repeat the methods to derive any specific number of features.

    Returns:
        df_fs_params (``pd.DataFrame``): A "template" dataframe that
            provides the sklearn model (``str``), the number of features
            (``int``), the feature indices (``list``), the feature ranking
            (``list``), and the parameters to recreate the feature
            selection scenario (``dict``).

    Example:
        >>> from geoml import FeatureSelection
        >>> from geoml.tests import config

        >>> myfs = FeatureSelection(config_dict=config.config_dict)
        >>> myfs.fs_find_params()
        >>> X_train_select, X_test_select = myfs.fs_get_X_select(df_fs_params_idx=2)
        >>> print(X_train_select[0:3])
        [[ 49.         235.3785       0.6442    ]
         [ 57.         358.672        0.67396667]
         [ 63.         246.587        0.48595   ]]
    '''
    print('\nPerforming feature selection...')
    if n_feats is None:
        n_feats = X_train.shape[1]
    else:
        n_feats = int(n_feats)
    msg1 = ('``n_feats`` must be a non-negative number greater than zero.')
    assert n_feats > 0, msg1

    if n_feats > X_train.shape[1]:
        print('``n_feats`` must not be more than are available.\n'
              '``n_feats``: {0}\nFeatures available: {1}\nAdjusting '
              '``n_feats`` to {1}'
              ''.format(n_feats, X_train.shape[1]))
        n_feats = X_train.shape[1]

    if model_fs_name == 'Lasso':
        df_fs_params = _lasso_fs_df(model_fs,
                                    model_fs_name,
                                    model_fs_params_adjust_min,
                                    X_train,
                                    y_train,
                                    labels_x,
                                    n_feats,
                                    n_linspace,
                                   )
        return df_fs_params
    # elif model_fs_name == 'PCA':
    else:
        raise NotImplementedError('{0} is not implemented.'.format(model_fs_name))


def fs_get_X_select(X_train          : AnyDataFrame,
                    X_test           : AnyDataFrame,
                    df_fs_params     : AnyDataFrame,
                    df_fs_params_idx : int,
                   ) -> Tuple[AnyDataFrame, AnyDataFrame, List[str], List[int]]:
    '''
    References <df_fs_params> to provide a new matrix X that only includes
    the selected features.

    Parameters:
        df_fs_params_idx (``int``): The index of <df_fs_params> to retrieve
            sklearn model parameters (the will be stored in the "params"
            column).

    Example:
        >>> from geoml import FeatureSelection
        >>> from geoml.tests import config

        >>> myfs = FeatureSelection(config_dict=config.config_dict)
        >>> myfs.fs_find_params()
        >>> X_train_select, X_test_select = myfs.fs_get_X_select(2)
    '''
    msg1 = ('<df_fs_params> must be populated; be sure to execute '
            '``find_feat_selection_params()`` prior to running '
            '``get_X_select()``.')
    assert isinstance(df_fs_params, pd.DataFrame), msg1

    feats_x_select = df_fs_params.loc[df_fs_params_idx]['feats_x_select']

    X_train_select = X_train[:,feats_x_select]
    X_test_select = X_test[:,feats_x_select]
    X_train_select = X_train_select
    X_test_select = X_test_select

    # TODO: Are these ever used?
    labels_x_select = df_fs_params.loc[df_fs_params_idx]['labels_x_select']
    rank_x_select = df_fs_params.loc[df_fs_params_idx]['rank_x_select']

    return X_train_select, X_test_select, labels_x_select, rank_x_select

