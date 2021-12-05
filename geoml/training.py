# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:24:40 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
from copy import deepcopy
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.metrics import r2_score  # type: ignore

from typing import Any, Optional, Dict, List, Tuple

from .utils import AnyDataFrame
from .feature_selection import fs_get_X_select

REGRESSOR_KEY = 'regressor__'


def _set_regressor(regressor        : Any,
                   regressor_params : Optional[Dict[str, Any]],
                   random_seed      : int,
                  ) -> str:
    '''
    Applies tuning parameters to the sklearn model(s) listed in
    <params_dict>. If the <random_seed> was not included in
    <model_tun_params> (or if there is a discrepancy), the model's random
    state is set/reset to avoid any discrepancy.
    '''
    if regressor_params is None:
        regressor_params = {}

    if 'regressor' in regressor.get_params().keys():
        regressor_params = _param_grid_add_key(regressor_params, REGRESSOR_KEY)
        regressor.set_params(**regressor_params)
        regressor_name = type(regressor.regressor).__name__
        try:
            regressor.set_params(**{REGRESSOR_KEY + 'random_state': random_seed})
        except ValueError:
            print('Invalid parameter <random_state> for estimator, thus '
                  '<random_state> cannot be set.\n')
    else:
        regressor.set_params(**regressor_params)
        regressor_name = type(regressor).__name__
        try:
            regressor.set_params(**{'random_state': random_seed})
        except ValueError:
            print('Invalid parameter <random_state> for estimator, thus '
                  '<random_state> cannot be set.\n')
    return regressor_name


def _param_grid_add_key(param_grid_dict : Dict[str, Any],
                        key : str
                       ) -> Dict[str, Any]:
    '''
    Define tuning parameter grids for pipeline or transformed regressor
    key.

    Parameters:
        param_grid_dict (``dict``): The parameter dictionary.
        key (``str``, optional): String to prepend to the ``sklearn`` parameter
            keys; should either be "transformedtargetregressor__regressor__"
            for pipe key or "regressor__" for transformer key (default:
            "regressor__").

    Returns:
        param_grid_mod: A modified version of <param_grid_dict>
    '''
    param_grid_mod = {f'{key}{k}': v for k, v in param_grid_dict.items()}
    return param_grid_mod


def _tune_grid_search(n_jobs_tune : int,
                      regressor   : Any,
                      param_grid  : Any, # TODO: ?
                      scoring     : Any,
                      refit       : Any,
                      X_train_select  : AnyDataFrame,
                      y_train         : AnyDataFrame,
                      tuning_splitter : Any,
                     ) -> AnyDataFrame:
    '''
    Performs the CPU intensive hyperparameter tuning via GridSearchCV

    Returns:
        df_tune (``pd.DataFrame``): Results of ``GridSearchCV()``.
    '''
    if n_jobs_tune > 0:
        pre_dispatch = int(n_jobs_tune*2)
    param_grid = _param_grid_add_key(param_grid, REGRESSOR_KEY)


    kwargs_grid_search = {'estimator': regressor,
                          'param_grid': param_grid,
                          'scoring': scoring,
                          'n_jobs': n_jobs_tune,
                          'pre_dispatch': pre_dispatch,
                          'cv': tuning_splitter,
                          'refit': refit,
                          'return_train_score': True}
    clf = GridSearchCV(**kwargs_grid_search, verbose=0)

    try:
        clf.fit(X_train_select, y_train)
        df_tune = pd.DataFrame(clf.cv_results_)
        return df_tune
    except ValueError as e:
        print('Estimator was unable to fit due to "{0}"'.format(e))
        return None


def _get_df_tune_cols(scoring : Any) -> List[str]:
    '''
    Gets column names for tuning dataframe
    '''
    cols = ['uid', 'model_fs', 'feat_n', 'feats_x_select', 'rank_x_select',
            'regressor_name', 'regressor', 'params_regressor',
            'params_tuning']
    prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
    for obj_str in scoring:
        for prefix in prefixes:
            col = prefix + obj_str
            cols.extend([col])
    return cols


def _get_df_test_cols() -> List[str]:
    '''
    Gets column names for test dataframe
    '''
    cols = ['uid', 'model_fs', 'feat_n', 'feats_x_select', 'rank_x_select',
            'regressor_name', 'regressor', 'params_regressor']
    prefixes = ['train_', 'test_']
    scoring = ['neg_mae', 'neg_rmse', 'r2']
    for obj_str in scoring:
        for prefix in prefixes:
            col = prefix + obj_str
            cols.extend([col])
    return cols


# TODO: Types
def _get_tune_results(df              : AnyDataFrame,
                      regressor       : Any,
                      regressor_name  : str,
                      scoring         : List[str],
                      rank_scoring    : Optional[str],
                      model_fs_name   : str,
                      labels_x_select : List[str],
                      rank_x_select   : List[int],
                      rank            : int           = 1
                     ):
    '''
    Retrieves all training and validation scores for a given <rank>. The
    first scoring string (from ``scoring``) is used to

    Parameters:
        df (``pd.DataFrame``): Dataframe containing results from
            ``_tune_grid_search``.
        rank (``int``): The rank to retrieve values for (1 is highest rank).
    '''
    data = [np.nan, model_fs_name, len(labels_x_select),
            labels_x_select, rank_x_select,
            regressor_name, deepcopy(regressor)]
    if not isinstance(df, pd.DataFrame):
        data.extend([np.nan] * (len(_get_df_tune_cols(scoring)) - len(data)))
        df_tune1 = pd.DataFrame(
            data=[data], columns=_get_df_tune_cols(scoring))
        return df_tune1
    if rank_scoring not in scoring or rank_scoring is None:
        rank_scoring = scoring[0]
    rank_scoring = 'rank_test_' + rank_scoring

    params_tuning = df[df[rank_scoring] == rank]['params'].values[0]
    regressor.set_params(**params_tuning)
    params_all = regressor.get_params()
    data.extend([params_all, params_tuning])
    for s in scoring:
        score_train_s = 'mean_train_' + s
        std_train_s = 'std_train_' + s
        score_test_s = 'mean_test_' + s
        std_test_s = 'std_test_' + s
        score_train = df[df[rank_scoring] == rank][score_train_s].values[0]
        std_train = df[df[rank_scoring] == rank][std_train_s].values[0]
        score_val = df[df[rank_scoring] == rank][score_test_s].values[0]
        std_val = df[df[rank_scoring] == rank][std_test_s].values[0]
        data.extend([score_train, std_train, score_val, std_val])

    df_tune1 = pd.DataFrame(data=[data], columns=_get_df_tune_cols(scoring))
    return df_tune1

# def _prep_pred_dfs(df_test, feat_n_list, y_label='nup_kgha'):
#     cols_scores = ['feat_n', 'feats', 'score_train_mae', 'score_test_mae',
#                    'score_train_rmse', 'score_test_rmse',
#                    'score_train_r2', 'score_test_r2']

#     cols_meta = ['study', 'date', 'plot_id', 'trt', 'rate_n_pp_kgha',
#                  'rate_n_sd_plan_kgha', 'rate_n_total_kgha', 'growth_stage',
#     cols = list(df_y.columns)
#     cols.remove('value')
#     cols.extend(['value_obs', 'value_pred'])

#     feat_n_list = list(my_train.df_tune['feat_n'])
#     cols_preds = cols_meta + feat_n_list
#     df_pred = pd.DataFrame(columns=cols_preds)
#     df_pred[cols_meta] = df_test[cols_meta]
#     df_score = pd.DataFrame(columns=cols_scores)
#     return df_pred, df_score


def _error(X : AnyDataFrame,
           y : AnyDataFrame,
           regressor : Any,
          ):
    '''
    Returns the MAE, RMSE, MSLE, and R2 for a fit model
    '''
    y_pred = regressor.predict(X)
    # sns.scatterplot(x=my_train.y_test, y=y_pred)


    neg_mae = -mean_absolute_error(y, y_pred)
    neg_rmse = -np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return y_pred, neg_mae, neg_rmse, r2


def _fit_all_data(X_train_select : AnyDataFrame,
                  X_test_select  : AnyDataFrame,
                  y_train        : AnyDataFrame,
                  y_test         : AnyDataFrame,
                  regressor      : Any,
                 ) -> None:
    '''
    Fits ``Training.regressor`` using the full dataset (i.e., training and
    test dataset).

    Cross validation is required to be sure our model is not overfit (i.e,
    that the model is not too complex); after hyperparameter tuning and
    testing, it is fine to train the model with the full dataset as long
    as the hyperparameters are set according to the results from the
    cross-validation.

    Caution: there should not be any feature selection, tuning,
    optimization, etc. after this function is executed.
    '''
    X = np.concatenate((X_train_select, X_test_select))
    y = np.concatenate((y_train, y_test))
    regressor.fit(X, y)


def _get_test_results(df              : AnyDataFrame,
                      model_fs_name   : str,
                      labels_x_select : List[str],
                      rank_x_select   : List[int],
                      regressor_name  : str,
                      regressor       : Any,
                      X_train_select  : AnyDataFrame,
                      X_test_select   : AnyDataFrame,
                      X_train         : AnyDataFrame,
                      y_train         : AnyDataFrame,
                      X_test          : AnyDataFrame,
                      y_test          : AnyDataFrame,
                     ):

    '''
    Trains the model for "current" tuning scenario and computes the
    train and test errors. The train errors in df_train are different than
    that of the train errors in df_tune because tuning uses k-fold cross-
    validation of the training set, whereas df_train uses the full training
    set.

    Parameters:
        df (``pd.DataFrame``):
    '''
    data = [df.iloc[0]['uid'], model_fs_name, len(labels_x_select),
            labels_x_select, rank_x_select,
            regressor_name, deepcopy(regressor)]
    if pd.isnull(df.iloc[0]['params_regressor']):
        data.extend([np.nan] * (len(_get_df_test_cols()) - len(data)))
        df_test_full1 = pd.DataFrame(
            data=[data], index=[df.index[0]], columns=_get_df_test_cols())
        return df_test_full1, None, None

    msg = ('<params_regressor> are not equal. (this is a bug)')
    assert regressor.get_params() == df['params_regressor'].values[0], msg

    regressor.fit(X_train_select, y_train)

    y_pred_train, train_neg_mae, train_neg_rmse, train_r2 = _error(X_train_select, y_train, regressor)
    y_pred_test, test_neg_mae, test_neg_rmse, test_r2 = _error(X_test_select, y_test, regressor)

    _fit_all_data(X_train_select, X_test_select, y_train, y_test, regressor)  # Fit using both train and test data
    data[-1] = deepcopy(regressor)
    data.extend([regressor.get_params()])
    data.extend([train_neg_mae, test_neg_mae, train_neg_rmse, test_neg_rmse,
                 train_r2, test_r2])
    df_test_full1 = pd.DataFrame([data], index=[df.index[0]], columns=_get_df_test_cols())
    return df_test_full1, y_pred_test, y_pred_train


    # estimator = df_tune_filtered2.iloc[0]['regressor']
    # estimator1 = estimator.replace('\n', '')

# def _execute_tuning(X, y, model_list, param_grid_dict,
#                    alpha, standardize, scoring, scoring_refit,
#                    max_iter, random_seed, key, df_train, n_splits, n_repeats,
#                    print_results=False):
#     '''
#     Execute model tuning, saving gridsearch hyperparameters for each number
#     of features.
#     '''
#     df_tune = None
#     for idx in self.df_fs_params.index:
#         X_train_select, X_test_select = self.fs_get_X_select(idx)
#         print('Number of features: {0}'.format(len(feats)))

#         param_grid_dict = param_grid_add_key(param_grid_dict, key)

#         df_tune_grid = self._tune_grid_search()
#         df_tune_rank = self._get_tune_results(df_tune_grid, rank=1)
#         if df_tune is None:
#             df_tune = df_tune_rank.copy()
#         else:
#             df_tune.append(df_tune_rank)

#         if print_results is True:
#             print('{0}:'.format(self.regressor_name))
#             print('R2: {0:.3f}\n'.format(df_temp['score_val_r2'].values[0]))
#     df_tune = df_tune.sort_values('feat_n').reset_index(drop=True)
#     self.df_tune = df_tune


# def _execute_tuning_pp(
#         logspace_list, X1, y1, model_list, param_grid_dict, standardize,
#         scoring, scoring_refit, max_iter, random_seed, key, df_train,
#         n_splits, n_repeats, df_tune_all_list):
#     '''
#     Actual execution of hyperparameter tuning via multi-core processing
#     '''
#     # chunks = chunk_by_n(reversed(logspace_list))
#     chunk_size = int(len(logspace_list) / (os.cpu_count()*2)) + 1
#     with ProcessPoolExecutor() as executor:
#         # for alpha, df_tune_feat_list in zip(reversed(logspace_list), executor.map(execute_tuning, it.repeat(X1), it.repeat(y1), it.repeat(model_list), it.repeat(param_grid_dict), reversed(logspace_list),
#         #                                                                           it.repeat(standardize), it.repeat(scoring), it.repeat(scoring_refit), it.repeat(max_iter), it.repeat(random_seed),
#         #                                                                           it.repeat(key), it.repeat(df_train), it.repeat(n_splits), it.repeat(n_repeats))):
#         for df_tune_feat_list in executor.map(execute_tuning, it.repeat(X1), it.repeat(y1), it.repeat(model_list), it.repeat(param_grid_dict), reversed(logspace_list),
#                                               it.repeat(standardize), it.repeat(scoring), it.repeat(scoring_refit), it.repeat(max_iter), it.repeat(random_seed),
#                                               it.repeat(key), it.repeat(df_train), it.repeat(n_splits), it.repeat(n_repeats), chunksize=chunk_size):
#                 # chunksize=chunk_size))

#             # print('df: {0}'.format(df_tune_feat_list))

#             # print('type: {0}'.format(type(df_tune_feat_list[0])))
#             df_tune_all_list = append_tuning_results(df_tune_all_list, df_tune_feat_list)
#     return df_tune_all_list

# def _set_df_pred_idx(self):
#     df = self.df_test
#     idx_full = self.df_pred.columns.get_level_values(level=0)
#     idx_filtered = []
#     for i in idx_full:
#         # print(i)
#         if i in self.df_y.columns:
#             idx_filtered.append(i)
#         elif i in list(df['index_full']):
#             idx_filtered.append(df[df['index_full'] == i].index[0])
#         else:
#             idx_filtered.append(np.nan)  # keep -1
#     self.df_pred.columns = pd.MultiIndex.from_arrays([idx_full, idx_filtered], names=('full', 'filtered'))

def _filter_test_results(df_tune      : AnyDataFrame,
                         df_test_full : AnyDataFrame,
                         scoring      : str = 'test_neg_mae'):
    '''
    Remove dupilate number of features (keep only lowest error)

    Parameters:
        scoring (``str``): If there are multiple scenarios with the same
            number of features, <scoring> corresponds to the <df_test_full>
            column that will be used to determine which scenario to keep
            (keeps the highest).
    '''
    msg = ('<df_tune> must be populated with parameters and test scroes. '
           'Have you executed ``tune_and_train()`` yet?')
    assert isinstance(df_tune, pd.DataFrame), msg

    df = df_test_full
    idx = df.groupby(['regressor_name', 'feat_n'])[scoring].transform(max) == df[scoring]
    idx_feat1 = df['feat_n'].searchsorted(1, side='left')
    if np.isnan(df.iloc[idx_feat1][scoring]):
        idx.iloc[idx_feat1] = True

    df_filtered = df_test_full[idx].drop_duplicates(['regressor_name', 'feat_n'])
    # df_filtered.reset_index(level=df_filtered.index.names, inplace=True)
    # df_filtered = df_filtered.rename(columns={'index': 'index_full'})
    df_filtered.reset_index(drop=True, inplace=True)
    df_test = df_filtered
    # self._set_df_pred_idx()

def _get_uid(df_test_full : AnyDataFrame,
             idx : int
            ) -> int:
    if df_test_full is None:
        idx_max = 0
    else:
        idx_max = df_test_full.index.max() + 1
    return int(idx_max + idx)


# TODO: Is it ok to pass the tuning splitter here directly?
#       What logging am I sacrificing?
def fit(df_y            : AnyDataFrame,
        df_fs_params    : AnyDataFrame,
        X_train         : AnyDataFrame,
        y_train         : AnyDataFrame,
        X_test          : AnyDataFrame,
        y_test          : AnyDataFrame,
        tuning_splitter : Any,
        n_jobs_tune     : int,
        regressor       : Any,
        regressor_params : Dict[str, Any],
        refit           : str,
        rank_scoring    : str,
        model_fs_name   : str,
        param_grid      : Dict[str, Any],
        scoring         : List[str],
        random_seed     : int,
       ) -> Tuple[AnyDataFrame, AnyDataFrame, AnyDataFrame, AnyDataFrame]:
    '''
    Perform tuning for each unique scenario from ``FeatureSelection``
    (i.e., for each row in <df_fs_params>).

    Example:
        >>> from geoml import Training
        >>> from geoml.tests import config

        >>> my_train = Training(config_dict=config.config_dict)
        >>> my_train.train()
    '''
    print('Executing hyperparameter tuning and estimator training...')

    regressor_name = _set_regressor(regressor, regressor_params, random_seed)

    #_ = get_tuning_splitter()  # prints the number of obs
    df_tune : Optional[AnyDataFrame] = None
    df_test_full : Optional[AnyDataFrame] = None
    df_pred : Optional[AnyDataFrame] = None
    for idx in df_fs_params.index:
        X_train_select, X_test_select, labels_x_select, rank_x_select = fs_get_X_select(X_train, X_test, df_fs_params, idx)
        n_feats = len(df_fs_params.loc[idx]['feats_x_select'])
        # TODO: Stderr
        if True:
            print('Number of features: {0}'.format(n_feats))
        df_tune_grid = _tune_grid_search(n_jobs_tune, regressor, param_grid, scoring, refit, X_train_select, y_train, tuning_splitter)
        df_tune_rank = _get_tune_results(df_tune_grid, regressor, regressor_name, scoring, rank_scoring, model_fs_name, labels_x_select, rank_x_select, rank=1)
        uid = _get_uid(df_test_full, idx)
        df_tune_rank.loc[0, 'uid'] = uid
        df_tune_rank = df_tune_rank.rename(index={0: uid})
        if df_tune is None:
            df_tune = df_tune_rank.copy()
        else:
            df_tune = df_tune.append(df_tune_rank)
        df_test_full1, y_pred_test, y_pred_train = _get_test_results(df_tune_rank, model_fs_name, labels_x_select, rank_x_select, regressor_name, regressor, X_train_select, X_test_select, X_train, y_train, X_test, y_test)
        if df_test_full is None:
            df_test_full = df_test_full1.copy()
        else:
            df_test_full = df_test_full.append(df_test_full1)

        # TODO: Stderr
        if True:
            print('{0}:'.format(regressor_name))
            print('R2: {0:.3f}\n'.format(df_tune_rank['score_val_r2'].values[0]))

        if df_pred is None:
            df_pred = df_y[df_y['train_test'] == 'test'].copy()
            df_pred_full = df_y.copy()

        if y_pred_test is not None:  # have to store y_pred while we have it
            df_pred[uid] = y_pred_test
            df_pred_full[uid] = np.concatenate([y_pred_train, y_pred_test])
            # df_pred[(uid, np.nan)] = y_pred


    _filter_test_results(df_tune, df_test_full, scoring='test_neg_mae')

    return df_tune, df_test_full, df_pred, df_pred_full

