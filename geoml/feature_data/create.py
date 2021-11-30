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
from datetime import datetime, date
import inspect
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import warnings

from copy import deepcopy
from sklearn.model_selection import RepeatedStratifiedKFold  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.impute import KNNImputer  # type: ignore
from sklearn.experimental import enable_iterative_imputer  # type: ignore
from sklearn.impute import IterativeImputer  # type: ignore

from ...db.db import utilities as db_utils

# TODO: Factor out
from ..tables.new_columns import dae, dap, rate_ntd
from ..tables.join import join_closest_date

from ..utils import AnyDataFrame
from ..config import FeatureDataConfig, GroupFeatures

from typing import Tuple, List, Set, Optional, Any, Dict
from typing import cast

# TODO: Function to filter cropscan data (e.g., low irradiance, etc.)


# TODO: Fix config
#       date_train
#       dv_method LeavePGroupsOut
#       cv_method_tune RepeatedStrafifiedKFold

def _handle_wl_cols(c : str,
                    wl_range : Tuple[int, int],
                    labels_x : List[str],
                    prefix   : str = 'wl_'
                   ) -> List[str]:
    if not isinstance(c, str):
        c = str(c)
    col = c.replace(prefix, '') if prefix in c else c

    if (col.isnumeric() and int(col) >= wl_range[0] and
        int(col) <= wl_range[1]):
        labels_x.append(c)
    return labels_x


# TODO: This function isn't very type-safe
def _get_labels_x(group_feats : GroupFeatures,
                  cols        : Optional[Set[str]] = None
                 ) -> List[str]:
      '''
      Parses ``group_feats`` and returns a list of column headings so that
      a df can be subset to build the X matrix with only the appropriate
      features indicated by ``group_feats``.
      '''

      labels_x : List[str] = []

      for key, feature in group_feats.items():
          print('Loading <group_feats> key: {0}'.format(key))
          if 'wl_range' in key:
              wl_range = group_feats[key]  # type: ignore
              assert cols is not None, ('``cols`` must be passed.')
              for c in cols:
                  labels_x = _handle_wl_cols(c, wl_range, labels_x,
                                                  prefix='wl_')
          elif 'bands' in key or 'weather_derived' in key or 'weather_derived_res' in key:
              labels_x.extend(cast(List[str], feature))
          elif 'rate_ntd' in key:
              labels_x.append(cast(Dict[str, str], feature)['col_out'])
          else:
              labels_x.append(cast(str, feature))
      return labels_x


def _check_empty_geom(df : AnyDataFrame):
    if isinstance(df, gpd.GeoDataFrame):
        if all(df[~df[df.geometry.name].is_empty]):
            df = pd.DataFrame(df.drop(columns=[df.geometry.name]))
    return df


def _join_group_feats(df             : AnyDataFrame,
                      group_feats    : GroupFeatures,
                      date_tolerance : int,
                      tables         : Dict[str, AnyDataFrame],
                     ) -> AnyDataFrame:
    '''
    Joins predictors to ``df`` based on the contents of group_feats
    '''
    as_planted = tables.get("as_planted")
    dates_res  = tables.get("dates_res")

    if 'dae' in group_feats:
        field_bounds = tables["field_bounds"]
        df = dae(df, field_bounds, as_planted, dates_res)

    if 'dap' in group_feats:
        field_bounds = tables["field_bounds"]
        df = dap(df, field_bounds, as_planted, dates_res)

    n_applications = tables.get("n_applications")
    experiments    = tables.get("experiments")
    trt            = tables.get("trt")
    trt_n          = tables.get("trt_n")

    if 'rate_ntd' in group_feats:
        col_rate_n = group_feats['rate_ntd']['col_rate_n']
        col_rate_ntd_out = group_feats['rate_ntd']['col_out']
        df = rate_ntd(df,
                      n_applications,
                      experiments,
                      trt,
                      trt_n,
                      col_rate_n = col_rate_n,
                      col_rate_ntd_out = col_rate_ntd_out
                     )

    if 'weather_derived' in group_feats:
        weather_derived = tables["weather_derived"]
        weather_derived = _check_empty_geom(weather_derived)
        # join weather by closest date
        df = join_closest_date(
               df,
               weather_derived,
               left_on='date',
               right_on='date',
               tolerance=0,
               delta_label=None
             )

    if 'weather_derived_res' in group_feats:
        df = join_closest_date(  # join weather by closest date
               df,
               tables["weather_derived_res"],
               left_on='date',
               right_on='date',
               tolerance=0,
               delta_label=None
             )

    # necessary because 'cropscan_wl_range1' must be differentiated
    for key in group_feats:
        if 'cropscan' in key:
            df = join_closest_date(
                   df,
                   tables["rs_cropscan_res"],
                   left_on     = 'date',
                   right_on    = 'date',
                   tolerance   = date_tolerance,
                   delta_label = 'cropscan'
                 )

        if 'micasense' in key:
            # join micasense by closest date
            df = join_closest_date(
                   df,
                   tables["rs_micasense_res"],
                   left_on     = 'date',
                   right_on    = 'date',
                   tolerance   = date_tolerance,
                   delta_label = 'micasense'
                 )

        if 'spad' in key:
            # join spad by closest date
            df = join_closest_date(
                   df,
                   tables["rs_spad_res"],
                   left_on     = 'date',
                   right_on    = 'date',
                   tolerance   = date_tolerance,
                   delta_label = 'spad'
                 )

        if 'sentinel' in key:
            # join sentinel by closest date
            df = join_closest_date(
                   df,
                   tables["rs_sentinel"],
                   left_on     = 'date',
                   right_on    = 'acquisition_time',
                   tolerance   = date_tolerance,
                   delta_label = 'sentinel'
                 )
    return df


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


def _add_empty_geom(gdf          : AnyDataFrame,
                    field_bounds : AnyDataFrame
                   ) -> gpd.GeoDataFrame:
    '''Adds field_bounds geometry to all Empty gdf geometries'''
    subset = db_utils.get_primary_keys(gdf)

    # Split gdf into geom/no-geom
    gdf_geom = gdf[~gdf[gdf.geometry.name].is_empty]
    gdf_nogeom = gdf[gdf[gdf.geometry.name].is_empty]

    # Get field bounds where owner, farm, field_id, and year match obs_tissue_no_geom

    gdf_nogeom.drop(columns=[gdf.geometry.name], inplace=True)
    field_bounds.drop(columns=['id'], inplace=True)
    gdf_out = gdf_geom.append(gdf_nogeom.merge(field_bounds, on=subset))
    return gdf_out


def _get_response_df(df_response          : AnyDataFrame,
                     field_bounds         : AnyDataFrame,
                     ground_truth_tissue  : str,
                     ground_truth_measure : str,
                     tissue_col           : str = 'tissue',
                     measure_col          : str = 'measure',
                    ) -> AnyDataFrame:
    '''
    Gets the relevant response dataframe

    Parameters:
        ground_truth_tissue (``str``): The tissue to use for the response
            variable. Must be in "obs_tissue.csv", and dictates which table
            to access to retrieve the relevant training data.
        ground_truth_measure (``str``): The measure to use for the response
            variable. Must be in "obs_tissue.csv"
        tissue_col (``str``): The column name from "obs_tissue.csv" to look
            for ``tissue``.
        measure_col (``str``): The column name from "obs_tissue.csv" to
            look for ``measure``.
    '''
    tissue_list = df_response.groupby(by=[measure_col, tissue_col], as_index=False).first()[tissue_col].tolist()
    measure_list = df_response.groupby(by=[measure_col, tissue_col], as_index=False).first()[measure_col].tolist()
    avail_list = ['_'.join(map(str, i)) for i in zip(tissue_list, measure_list)]
    msg = ('``tissue``  and ``measure`` must be '
           'one of:\n{0}.\nPlease see "obs_tissue" table to be sure your '
           'intended data are available.'
           ''.format(list(zip(tissue_list, measure_list))))
    assert '_'.join((ground_truth_tissue, ground_truth_measure)) in avail_list, msg

    df = df_response[(df_response[measure_col] == ground_truth_measure) &
                     (df_response[tissue_col] == ground_truth_tissue)]
    df = _add_empty_geom(df, field_bounds)
    return df


def _stratify_set(stratify_cols : List[str] = ['owner', 'farm', 'year'],
                  train_test    : Optional[str] = None,
                  df            : Optional[AnyDataFrame] = None,
                  df_y          : Optional[pd.Series]    = None,
                 ) -> AnyDataFrame:
    '''
    Creates a 1-D array of the stratification IDs (to be used by k-fold)
    for both the train and test sets: <stratify_train> and <stratify_test>

    Returns:
        groups (``numpy.ndarray): Array that asssigns each observation to
            a stratification group.
    '''
    if df is None and df_y is not None:
        df = df_y.copy()
    msg1 = ('All <stratify> strings must be columns in <df_y>')
    if df is not None:
      for c in stratify_cols:
          assert c in df.columns, msg1
    if train_test is None:
        groups = df.groupby(stratify_cols).ngroup().values
    else:
        groups = df[df['train_test'] == train_test].groupby(
            stratify_cols).ngroup().values

    unique, counts = np.unique(groups, return_counts=True)
    print('\nStratification groups: {0}'.format(stratify_cols))
    print('Number of stratification groups:  {0}'.format(len(unique)))
    print('Minimum number of splits allowed: {0}'.format(min(counts)))
    return groups


def _check_sklearn_splitter(cv_method        : Any,
                            cv_method_kwargs : Dict[str, Any],
                            cv_split_kwargs  : Optional[Dict[str, Any]] = None,
                            raise_error      : bool  = False
                           ) -> Dict[str, Any]:
    '''
    Checks <cv_method>, <cv_method_kwargs>, and <cv_split_kwargs> for
    continuity.

    Displays a UserWarning or raises ValueError if an invalid parameter
    keyword is provided.

    Parameters:
        raise_error (``bool``): If ``True``, raises a ``ValueError`` if
            parameters do not appear to be available. Otherwise, simply
            issues a warning, and will try to move forward anyways. This
            exists because <inspect.getfullargspec(cv_method)[0]> is
            used to get the arguments, but certain scikit-learn functions/
            methods do not expose their arguments to be screened by
            <inspect.getfullargspec>. Thus, the only way to use certain
            splitter functions is to bypass this check.

    Note:
        Does not check for the validity of the keyword argument(s). Also,
        the warnings does not work as fully intended because when
        <inspect.getfullargspec(cv_method)[0]> returns an empty list,
        there is no either a warning or ValueError can be raised.
    '''
    if cv_split_kwargs is None:
        cv_split_kwargs = {}

    cv_method_args = inspect.getfullargspec(cv_method)[0]
    cv_split_args = inspect.getfullargspec(cv_method.split)[0]

    return cv_split_kwargs


def _cv_method_check_random_seed(cv_method : Any,
                                 cv_method_kwargs : Dict[str, Any],
                                 random_seed      : int,
                                ) -> Dict[str, Any]:
    '''
    If 'random_state' is a valid parameter in <cv_method>, sets from
    <random_seed>.
    '''
    # cv_method_args = inspect.getfullargspec(cv_method)[0]
    # TODO: Add tests for all supported <cv_method>s
    cv_method_args = inspect.signature(cv_method).parameters
    if 'random_state' in cv_method_args:  # ensure random_seed is set correctly
        cv_method_kwargs['random_state'] = random_seed  # if this will get passed to eval(), should be fine since it gets passed to str() first
    return cv_method_kwargs


def _splitter_eval(cv_split_kwargs : Dict[str, Any],
                   df : Optional[AnyDataFrame] = None
                  ) -> Dict[str, Any]:
    '''
    Preps the CV split keyword arguments (evaluates them to variables).
    '''
    if cv_split_kwargs is None:
        cv_split_kwargs = {}
    if 'X' not in cv_split_kwargs and df is not None:  # sets X to <df>
        cv_split_kwargs['X'] = 'df'
    scope = locals()

    if df is None and 'df' in [
            i for i in [a for a in cv_split_kwargs.values()]]:
        raise ValueError(
            '<df> is None, but is present in <cv_split_kwargs>. Please '
            'pass <df> or ajust <cv_split_kwargs>')
    # evaluate any str; keep anything else as is
    cv_split_kwargs_eval = dict(
        (k, eval(str(cv_split_kwargs[k]), scope))
        if isinstance(cv_split_kwargs[k], str)
        else (k, cv_split_kwargs[k])
        for k in cv_split_kwargs)

    return cv_split_kwargs_eval


def _train_test_split_df(df               : AnyDataFrame,
                         random_seed      : int,
                         cv_method        : Any,
                         cv_method_kwargs : Dict[str, Any],
                         cv_split_kwargs  : Optional[Dict[str, Any]] = None,
                        ) -> AnyDataFrame:
    '''
    Splits <df> into train and test sets.
    '''
    cv_method = deepcopy(cv_method)
    cv_method_kwargs = deepcopy(cv_method_kwargs)
    cv_split_kwargs = deepcopy(cv_split_kwargs)
    cv_method_kwargs = _cv_method_check_random_seed(
        cv_method, cv_method_kwargs, random_seed)
    if cv_method.__name__ == 'train_test_split':
        # Because train_test_split has **kwargs for options, random_state is not caught, so it should be set explicitly
        cv_method_kwargs['random_state'] = random_seed
        if 'arrays' in cv_method_kwargs:  # I think can only be <df>?
            df = eval(cv_method_kwargs.pop('arrays', None))
        scope = locals()  # So it understands what <df> is inside func scope
        cv_method_kwargs_eval = dict(
            (k, eval(str(cv_method_kwargs[k]), scope)
             ) for k in cv_method_kwargs)
        df_train, df_test = cv_method(df, **cv_method_kwargs_eval)
    else:
        cv_split_kwargs = _check_sklearn_splitter(
            cv_method, cv_method_kwargs, cv_split_kwargs,
            raise_error=False)
        cv = cv_method(**cv_method_kwargs)
        for key in ['y', 'groups']:
            if key in cv_split_kwargs:
                if isinstance(cv_split_kwargs[key], list):
                    # assume these are columns to group by and adjust kwargs
                    cv_split_kwargs[key] = _stratify_set(
                        stratify_cols=cv_split_kwargs[key],
                        train_test=None, df=df)

        # Now cv_split_kwargs should be ready to be evaluated
        cv_split_kwargs_eval = _splitter_eval(
            cv_split_kwargs, df=df)

        if 'X' not in cv_split_kwargs_eval:  # sets X
            cv_split_kwargs_eval['X'] = df

        train_idx, test_idx = next(cv.split(**cv_split_kwargs_eval))
        df_train, df_test = df.loc[train_idx], df.loc[test_idx]

    train_pct = (len(df_train) / (len(df_train) + len(df_test))) * 100
    test_pct = (len(df_test) / (len(df_train) + len(df_test))) * 100
    print('\nNumber of observations in the "training" set: {0} ({1:.1f}%)'.format(len(df_train), train_pct))
    print('Number of observations in the "test" set: {0} ({1:.1f}%)\n'.format(len(df_test), test_pct))

    df_train.insert(0, 'train_test', 'train')
    df_test.insert(0, 'train_test', 'test')
    df_out = df_train.copy()
    df_out = df_out.append(df_test).reset_index(drop=True)

    return df_out


def _impute_missing_data(X           : AnyDataFrame,
                         random_seed : int,
                         method      : str = 'iterative'
                        ) -> AnyDataFrame:
    '''
    Imputes missing data in X - sk-learn models will not work with null data

    Parameters:
        method (``str``): should be one of "iterative" (takes more time)
            or "knn" (default: "iterative").
    '''
    if pd.isnull(X).any() is False:
        return X

    if method == 'iterative':
        imp = IterativeImputer(max_iter=10, random_state=random_seed)
    elif method == 'knn':
        imp = KNNImputer(n_neighbors=2, weights='uniform')
    elif method == None:
        return X
    X_out = imp.fit_transform(X)

    return X_out


def _get_X_and_y(df            : AnyDataFrame,
                 label_y       : str,
                 group_feats   : GroupFeatures,
                 random_seed   : int,
                 impute_method : str = 'iterative'
                ) -> Tuple[AnyDataFrame, AnyDataFrame, pd.Series, pd.Series, AnyDataFrame, List[str]]:
    msg = ('``impute_method`` must be one of: ["iterative", "knn", None]')
    assert impute_method in ['iterative', 'knn', None], msg

    if impute_method is None:
        df = df.dropna()
    else:
        df = df[pd.notnull(df[label_y])]
    labels_x = _get_labels_x(group_feats, cols=df.columns)

    df_train = df[df['train_test'] == 'train']
    df_test = df[df['train_test'] == 'test']

    # If number of cols are different, then remove from both and update labels_x
    cols_nan_train = df_train.columns[df_train.isnull().all(0)]  # gets columns with all nan
    cols_nan_test = df_test.columns[df_test.isnull().all(0)]
    if len(cols_nan_train) > 0 or len(cols_nan_test) > 0:
        df.drop(list(cols_nan_train) + list(cols_nan_test), axis='columns', inplace=True)
        df_train = df[df['train_test'] == 'train']
        df_test = df[df['train_test'] == 'test']
        labels_x = _get_labels_x(group_feats, cols=df.columns)

    X_train = df_train[labels_x].values
    X_test = df_test[labels_x].values
    y_train = df_train[label_y].values
    y_test = df_test[label_y].values

    X_train = _impute_missing_data(X_train, random_seed, method=impute_method)
    X_test = _impute_missing_data(X_test, random_seed, method=impute_method)

    msg = ('There is a different number of columns in <X_train> than in '
           '<X_test>.')
    assert X_train.shape[1] == X_test.shape[1], msg

    return X_train, X_test, y_train, y_test, df, labels_x

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


def get_feat_group_X_y(df_response : AnyDataFrame,
                       tables      : Dict[str, AnyDataFrame],
                       ground_truth_tissue : str,
                       ground_truth_measure : str,
                       group_feats : GroupFeatures,
                       date_tolerance : int,
                       date_train : date,
                       label_y  : str,
                       labels_y_id : List[str],
                       dir_results : Optional[str],
                       random_seed : int,
                       impute_method : str,
                       cv_method   : Any,
                       cv_method_kwargs : Dict[str, Any],
                       cv_split_kwargs  : Optional[Dict[str, Any]] = None,
                      ) -> Tuple[AnyDataFrame, AnyDataFrame, List[str]]:
    '''
    Retrieves all the necessary columns in ``group_feats``, then filters
    the dataframe so that it is left with only the identifying columns
    (i.e., study, year, and plot_id), a column indicating if each
    observation belongs to the train or test set (i.e., train_test), and
    the feature columns indicated by ``group_feats``.

    Note:
        This function is designed to handle any of the many scikit-learn
        "splitter classes". Documentation available at
        https://scikit-learn.org/stable/modules/classes.html#splitter-classes.
        All parameters used by the <cv_method> function or the
        <cv_method.split> method should be set via
        <cv_method_kwargs> and <cv_split_kwargs>.
    '''
    df = _get_response_df(df_response,
                          tables["field_bounds"],
                          ground_truth_tissue,
                          ground_truth_measure,
                         )
    df = _join_group_feats(df, group_feats, date_tolerance, tables)
    msg = ('After joining feature data with response data and filtering '
           'by <date_tolerance={0}>, there are no observations left. '
           'Check that there are a sufficient number of observations with '
           'both feature and response data within the date tolerance.'
           ''.format(date_tolerance))

    # TODO: Why do I have to cast date_train now?
    df = df[df['date'] < pd.to_datetime(date_train)].reset_index()

    assert len(df) > 0, msg

    df = _train_test_split_df(df, random_seed, cv_method, cv_method_kwargs, cv_split_kwargs)

    X_train, X_test, y_train, y_test, df, labels_x, = _get_X_and_y(
        df, label_y, group_feats, random_seed, impute_method)

    subset = db_utils.get_primary_keys(df)
    labels_id = subset + ['date', 'train_test']
    df_X = df[labels_id + labels_x]
    df_y = df[labels_id + labels_y_id + [label_y]]

    if dir_results is not None:
        _save_df_X_y(dir_results, label_y, df_X, df_y)

    return df_X, df_y, labels_id


def get_tuning_splitter(cv_method_tune        : Any,
                        cv_method_tune_kwargs : Dict[str, Any],
                        cv_split_tune_kwargs  : Dict[str, Any],
                        random_seed           : int,
                        df_X                  : AnyDataFrame,
                        print_splitter_info   : bool,
                       ) -> Any:
    cv_method = deepcopy(cv_method_tune)
    cv_method_kwargs = deepcopy(cv_method_tune_kwargs)
    cv_split_kwargs = deepcopy(cv_split_tune_kwargs)
    cv_method_kwargs = _cv_method_check_random_seed(
                         cv_method, cv_method_kwargs, random_seed)

    if cv_method.__name__ == 'train_test_split':
        # Because train_test_split has **kwargs for options, random_state is not caught, so it should be set explicitly
        cv_method_kwargs['random_state'] = random_seed
        if 'arrays' in cv_method_kwargs:  # I think can only be <df>?
            df = eval(cv_method_kwargs.pop('arrays', None))
        scope = locals()  # So it understands what <df> is inside func scope
        cv_method_kwargs_eval = dict(
            (k, eval(str(cv_method_kwargs[k]), scope)
             ) for k in cv_method_kwargs)
        return cv_method(df, **cv_method_kwargs_eval)
    else:
        cv_split_kwargs = _check_sklearn_splitter(
            cv_method, cv_method_kwargs, cv_split_kwargs,
            raise_error=False)
        cv_split_tune_kwargs = cv_split_kwargs
        cv = cv_method(**cv_method_kwargs)
        for key in ['y', 'groups']:
            if key in cv_split_kwargs:
                if isinstance(cv_split_kwargs[key], list):
                    # assume these are columns to group by and adjust kwargs
                    cv_split_kwargs[key] = _stratify_set(
                        stratify_cols=cv_split_kwargs[key],
                        train_test='train')

        # Now cv_split_kwargs should be ready to be evaluated
        df_X_train = df_X[df_X['train_test'] == 'train']
        cv_split_kwargs_eval = _splitter_eval(
            cv_split_kwargs, df=df_X_train)

        if 'X' not in cv_split_kwargs_eval:  # sets X
            cv_split_kwargs_eval['X'] = df_X_train

    if print_splitter_info == True:
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

