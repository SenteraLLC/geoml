import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore

from db import utilities as db_utils

from ..utils import AnyDataFrame
from ..utils import check_col_names

from typing import Optional


def get_geom_from_primary_keys(df           : AnyDataFrame,
                               field_bounds : AnyDataFrame,
                              ) -> gpd.GeoDataFrame:
    '''Adds field_bounds geom to <df> based on primary keys'''
    for g in ['goem', 'geometry']:
        if g in df.columns:
            df.drop(columns=[g], inplace=True)
    subset = db_utils.get_primary_keys(df)
    gdf = gpd.GeoDataFrame(
        df.merge(field_bounds[subset + [field_bounds.geometry.name]],
                 on=subset), geometry='geom')
    return gdf


def spatial_join_clean_keys(df_left  : AnyDataFrame,
                            df_right : AnyDataFrame
                           ) -> AnyDataFrame:
    '''
    Performs spatial join, then only keeps matched primary keys.

    Deletes any rows whose primary keys in the joined dataframes do not
    match, then renames primary keys with standard naming.
    '''

    df_join_s = gpd.tools.sjoin(df_left, df_right.to_crs(df_left.crs.to_epsg()), how='inner')

    subset = db_utils.get_primary_keys(df_left)
    if 'field_id' in subset:
        keys = [['owner_left', 'owner_right'],
                ['farm_left', 'farm_right'],
                ['field_id_left', 'field_id_right'],
                ['year_left', 'year_right']]
    else:
        keys = [['owner_left', 'owner_right'],
                ['study_left', 'study_right'],
                ['plot_id_left', 'plot_id_right'],
                ['year_left', 'year_right']]
    df_join = df_join_s.loc[(df_join_s[keys[0][0]] == df_join_s[keys[0][1]]) &
                            (df_join_s[keys[1][0]] == df_join_s[keys[1][1]]) &
                            (df_join_s[keys[2][0]] == df_join_s[keys[2][1]]) &
                            (df_join_s[keys[3][0]] == df_join_s[keys[3][1]])]

    df_join.rename(columns={
        keys[0][0]: keys[0][0].split('_left')[0],
        keys[1][0]: keys[1][0].split('_left')[0],
        keys[2][0]: keys[2][0].split('_left')[0],
        keys[3][0]: keys[3][0].split('_left')[0]},
        inplace=True)
    df_join.drop(columns={
        keys[0][1]: keys[0][1].split('_right')[0],
        keys[1][1]: keys[1][1].split('_right')[0],
        keys[2][1]: keys[2][1].split('_right')[0],
        keys[3][1]: keys[3][1].split('_right')[0]},
        inplace=True)

    return df_join



def dae(df           : AnyDataFrame,
        field_bounds : AnyDataFrame,
        as_planted   : Optional[AnyDataFrame],
        dates_res    : Optional[AnyDataFrame],
       ) -> AnyDataFrame:
    subset = db_utils.get_primary_keys(df)
    cols_require = [subset + ['date', df.geometry.name]
                    if isinstance(df, gpd.GeoDataFrame)
                    else subset + ['date']][0]
    check_col_names(df, cols_require)
    df.loc[:, 'date'] = df.loc[:, 'date'].apply(pd.to_datetime, errors='coerce')
    if not isinstance(df, gpd.GeoDataFrame):
        df = get_geom_from_primary_keys(df, field_bounds)  # Returns GeoDataFrame

    if 'field_id' in subset:
        if as_planted is not None:
            df_join = spatial_join_clean_keys(df, as_planted)
        else:
            raise RuntimeError("Expected 'as_planted' data for dae")
    elif 'plot_id' in subset:
        on = [i for i in subset if i != 'plot_id']
        if dates_res is not None:
            df_join = df.merge(dates_res, on=on,
                               validate='many_to_one')
        else:
            raise RuntimeError("Expected 'as_planted' data for dae")
    if len(df_join) == 0:
        raise RuntimeError(
            'Unable to calculate DAE. Check that "date_emerge" is set in '
            '<as_planted> table for primary keys:\n{0}'.format(df[subset]))

    df_join.loc[:, 'date_emerge'] = df_join.loc[:, 'date_emerge'].apply(pd.to_datetime, errors='coerce')
    df_join['dae'] = (df_join['date']-df_join['date_emerge']).dt.days
    df_out = df.merge(df_join[cols_require + ['dae']], on=cols_require)
    df_out = df_out.drop_duplicates()
    return df_out.reset_index(drop=True)


def dap(df           : AnyDataFrame,
        field_bounds : AnyDataFrame,
        as_planted   : AnyDataFrame,
        dates_res    : AnyDataFrame,
       ) -> AnyDataFrame:

    subset = db_utils.get_primary_keys(df)
    cols_require = [subset + ['date', df.geometry.name]
                    if isinstance(df, gpd.GeoDataFrame)
                    else subset + ['date']][0]
    check_col_names(df, cols_require)
    df.loc[:, 'date'] = df.loc[:, 'date'].apply(pd.to_datetime, errors='coerce')
    if not isinstance(df, gpd.GeoDataFrame):
        df = get_geom_from_primary_keys(df, field_bounds)  # Returns GeoDataFrame

    if 'field_id' in subset:
        df_join = spatial_join_clean_keys(df, as_planted)
    elif 'plot_id' in subset:
        on = [i for i in subset if i != 'plot_id']
        df_join = df.merge(dates_res, on=on,
                           validate='many_to_one')
    if len(df_join) == 0:
        raise RuntimeError(
            'Unable to calculate DAP. Check that "date_plant" is set in '
            '<as_planted> table for primary keys:\n{0}'.format(df[subset]))

    df_join.loc[:, 'date_plant'] = df_join.loc[:, 'date_plant'].apply(pd.to_datetime, errors='coerce')
    df_join['dap'] = (df_join['date']-df_join['date_plant']).dt.days
    df_out = df.merge(df_join[cols_require + ['dap']], on=cols_require)
    return df_out.reset_index(drop=True)


def _cr_rate_ntd(df : AnyDataFrame) -> None:
    '''
    join_tables.rate_ntd() must sum all the N rates before a particular
    date within each study/year/plot_id combination. Therefore, there can
    NOT be duplicate rows of the metadata when using
    join_tables.rate_ntd(). This function ensures that there is not
    duplicate metadata information in df.
    '''
    msg = ('There can NOT be duplicate rows of the metadata in the ``df`` '
           'passed to ``join_tables.rate_ntd()``. Please filter ``df`` so '
           'there are not duplicate metadata rows.\n\nHint: is ``df`` '
           'in a long format with multiple types of data (e.g., vine N '
           'and tuber N)?\n..or does ``df`` contain subsamples?')
    subset = db_utils.get_primary_keys(df)
    if df.groupby(subset + ['date']).size()[0].max() > 1:
        raise AttributeError(msg)


def _add_id_by_subset(df, subset, col_id='id'):
    '''
    Adds a new column named 'id' to <df> based on unique subset values.
    '''
    df_unique_id = df.drop_duplicates(
        subset=subset, keep='first').sort_values(
            by=subset)[subset]
    df_unique_id.insert(0, 'id', list(range(len(df_unique_id))))
    df_id = df.merge(df_unique_id, on=subset)
    df_id = df_id[['id'] + [c for c in df_id.columns if c != 'id' ]]
    return df_id



def rate_ntd(df               : AnyDataFrame,
             n_applications   : AnyDataFrame,
             experiments      : AnyDataFrame,
             trt              : AnyDataFrame,
             trt_n            : AnyDataFrame,
             col_rate_n       : str = 'rate_n_kgha',
             col_rate_ntd_out : str = 'rate_ntd_kgha',
            ) -> gpd.GeoDataFrame:
    subset = db_utils.get_primary_keys(df)
    cols_require = subset + ['date']
    check_col_names(df, cols_require)
    _cr_rate_ntd(df)  # raises an error if data aren't suitable

    if 'id' not in df.columns:  # required for multiple geometries
        df = _add_id_by_subset(df, subset=cols_require)

    if 'field_id' in subset:
        # We need spatial join, but then have to remove excess columns and rename
        subset_n_apps = ['date_applied', 'source_n', 'rate_n_kgha']
        df_join : Optional[AnyDataFrame] = None
        for y in sorted(df['year'].unique()):
            df_y = df[df['year'] == y]
            n_apps_y = n_applications[n_applications['year'] == y]
            if df_join is None:
                df_join = gpd.overlay(df_y[['id'] + cols_require + ['geom']],
                                      n_apps_y[subset_n_apps + ['geom']], how='intersection')
            else:
                df_join = df_join.append(gpd.overlay(df_y[['id'] + cols_require + ['geom']],
                                         n_apps_y[subset_n_apps + ['geom']], how='intersection'))

        # Don't drop index_right? Becasue it defines the duplicates from the right df
        if df_join is not None and 'index_right' in df_join.columns:
            df_join.drop(columns='index_right', inplace=True)

        # remove all rows where date_applied is after date
        # TODO: Don't ignore typing
        df_join = df_join[df_join['date_applied'] <= df_join['date']] # type: ignore


        df_join.loc[~df_join.geometry.is_valid, df_join.geometry.name] = df_join.loc[~df_join.geometry.is_valid].buffer(0)

        df_dissolve = df_join.dissolve(by=['id'] + cols_require, as_index=False, aggfunc='sum')

        df_out = df.merge(df_dissolve[cols_require + ['rate_n_kgha']], on=cols_require, how='left', indicator=True)
        df_out = df_out[df_out['_merge'] == 'both'].drop(columns='_merge').drop_duplicates()
        df_out.rename(columns={col_rate_n: col_rate_ntd_out}, inplace=True)

    elif 'plot_id' in subset:
        df_join = df.merge(experiments, on=subset).merge(
            trt, on=['owner', 'study', 'year', 'trt_id'], validate='many_to_many').merge(
                trt_n, on=['owner', 'study', 'year', 'trt_n'], validate='many_to_many')

        # remove all rows where date_applied is after date
        df_join = df_join[df_join['date_applied'] <= df_join['date']]
        df_sum = df_join.groupby(cols_require)[col_rate_n].sum().reset_index()
        df_sum.rename(columns={col_rate_n: col_rate_ntd_out}, inplace=True)
        df_out = df.merge(df_sum, on=cols_require)
    return df_out.reset_index(drop=True)
