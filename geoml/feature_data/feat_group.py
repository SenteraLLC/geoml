from typing import Any, Dict, Optional

from datetime import date
from copy import deepcopy

import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore

from ..utils import AnyDataFrame
from ...db.db import utilities as db_utils
from ..config.config import GroupFeatures

from .util import cv_method_check_random_seed, check_sklearn_splitter, stratify_set, splitter_eval

# TODO: Factor out
from ..tables.new_columns import dae, dap, rate_ntd
from ..tables.join import join_closest_date



def _check_empty_geom(df : AnyDataFrame) -> AnyDataFrame:
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

    field_bounds = tables["field_bounds"]
    as_planted = tables.get("as_planted")
    dates_res  = tables.get("dates_res")

    if 'dae' in group_feats:
        df = dae(df, field_bounds, as_planted, dates_res)

    if 'dap' in group_feats:
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
    cv_method_kwargs = cv_method_check_random_seed(
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
        cv_split_kwargs = check_sklearn_splitter(
            cv_method, cv_method_kwargs, cv_split_kwargs,
            raise_error=False)
        cv = cv_method(**cv_method_kwargs)
        for key in ['y', 'groups']:
            if key in cv_split_kwargs:
                if isinstance(cv_split_kwargs[key], list):
                    # assume these are columns to group by and adjust kwargs
                    cv_split_kwargs[key] = stratify_set(
                        stratify_cols=cv_split_kwargs[key],
                        train_test=None, df=df)

        # Now cv_split_kwargs should be ready to be evaluated
        cv_split_kwargs_eval = splitter_eval(
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


def get_feat_group_X_y(df_response          : AnyDataFrame,
                       tables               : Dict[str, AnyDataFrame],
                       group_feats          : GroupFeatures,
                       ground_truth_tissue  : str,
                       ground_truth_measure : str,
                       date_tolerance       : int,
                       date_train           : date,
                       random_seed          : int,
                       cv_method            : Any,
                       cv_method_kwargs     : Dict[str, Any],
                       cv_split_kwargs      : Optional[Dict[str, Any]] = None,
                      ) -> AnyDataFrame:
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
    return df

