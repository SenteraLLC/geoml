from typing import Set

from . import util as table_utils


def check_empty_geom(df : AnyDataFrame):
    '''Issues a warning if there is empty geometry in <df>.'''
    msg1 = ('Some geometries are empty and will not result in joins on those '
            'rows. Either pass as a normal dataframe with no geometry, or add '
            'the appropriate geometry for all rows.')
    msg2 = ('A geodataframe was passed to ``join_closest_date()``, but all '
            'geometries are empty and thus no rows will be joined. Please '
            'either pass as a normal dataframe with no geometry, or add in '
            'the appropriate geometry.')
    if all(df[df.geometry.name].is_empty):
        raise ValueError(msg2)
    if any(df[df.geometry.name].is_empty):
        warnings.warn(msg1, category=RuntimeWarning, stacklevel=1)


def add_date_delta(df_join     : AnyDataFrame,
                   left_on     : str,
                   right_on    : str,
                   delta_label : Optional[str] = None
                  ) -> Tuple[AnyDataFrame, str]:
    '''
    Adds "date_delta" column to <df_join>
    '''
    idx_delta = df_join.columns.get_loc(right_on)
    if delta_label is None:
        date_delta_str = 'date_delta'
    else:
        date_delta_str = 'date_delta_{0}'.format(delta_label)
    df_join.insert(idx_delta+1, date_delta_str, None)
    df_join[date_delta_str] = (df_join[left_on]-df_join[right_on]).astype('timedelta64[D]')
    df_join = df_join[pd.notnull(df_join[date_delta_str])]
    return df_join, date_delta_str



def dt_or_ts_to_date(df           : AnyDataFrame,
                      col_date    : str,
                      date_format : str ='%Y-%m-%d'
                     ) -> AnyDataFrame:
    '''
    Checks if <col> is a valid datetime or timestamp column. If not, sets
    it as such as a <datetime.date> object.

    Returns:
        df
    '''
    if not pd.core.dtypes.common.is_datetime64_any_dtype(df[col_date]):
        df[col_date] = pd.to_datetime(df.loc[:, col_date], format=date_format)
    df[col_date] = df[col_date].dt.date
    df[col_date] = pd.to_datetime(df[col_date])  # convert to datetime so pd.merge is possible

    # if isinstance(df[col_date].iloc[0], pd._libs.tslibs.timestamps.Timestamp):
    #     df[col_date] = df[col_date].to_datetime64()
    return df


def join_geom_date(df_left     : AnyDataFrame,
                   df_right    : AnyDataFrame,
                   left_on     : str,
                   right_on    : str,
                   tolerance   : int = 3,
                   delta_label : Optional[str] = None
                  ) -> AnyDataFrame:
    '''
    Merges on geometry, then keeps only closest date.

    The problem here is that pd.merge_asof() requires the data to be sorted
    on the keys to be joined, and because geometry cannot be sorted
    inherently, this will not work for tables/geodataframes that are able
    to support multiple geometries for a given set of primary keys.

    Method:
        0. For any empty geometry, we assume we are missing field_bounds
        and thus do not have Sentinel imagery.
        1. Many to many spatial join between df_left and df_right; this
        will probably result in 10x+ data rows.
        2. Remove all rows whose left geom does not "almost eqaul" the
        right geom (using gpd.geom_almost_equals()).
        3. Remove all rows whose left date is outside the tolerance of the
        right date.
        4. Finally, choose the columns to keep and tidy up df.

    Parameters:
        delta_label (``str``): Used to set the column name of the
            "delta_date" by appending <delta_label> to "delta_date"
            (e.g., "delta_date_mylabel"). Will only be set if <tolerance>
            is greater than zero.
    '''
    subset_left = db_utils.get_primary_keys(df_left)
    subset_right = db_utils.get_primary_keys(df_right)
    # Get names of geometry columns
    geom_l = df_left.geometry.name + '_l'
    geom_r = df_right.geometry.name + '_r'

    # Missing geometry will get filtered out here
    df_merge1 = df_left.merge(df_right, how='inner', on=subset_left, suffixes=['_l', '_r'], validate='many_to_many')
    # Grab geometry for each df; because zonal stats were retrieved based
    # on obs_tissue geom, we can keep observations that geometry is equal.
    gdf_merge1_l = gpd.GeoDataFrame(df_merge1[geom_l], geometry=geom_l)
    gdf_merge1_r = gpd.GeoDataFrame(df_merge1[geom_r], geometry=geom_r)
    # Remove all rows whose left geom does not "almost eqaul" right geom
    df_sjoin2 = df_merge1[gdf_merge1_l.geom_almost_equals(gdf_merge1_r, 8)].copy()
    left_on2 = left_on + '_l'
    right_on2 = right_on + '_r'
    df_sjoin2.rename(columns={left_on:left_on2, right_on:right_on2}, inplace=True)
    # Add "date_delta" column
    # if tolerance == 0:
    #     df_sjoin2.dropna(inplace=True)
    # elif tolerance > 0:
    df_sjoin2, delta_label_out = add_date_delta(
        df_sjoin2, left_on=left_on2, right_on=right_on2,
        delta_label=delta_label)
    df_sjoin2.dropna(inplace=True)
    # idx_delta = df_sjoin2.columns.get_loc(left_on2)
    # df_sjoin2.insert(
    #     idx_delta+1, 'date_delta',
    #     (df_sjoin2[left_on2]-df_sjoin2[right_on2]).astype('timedelta64[D]'))
    # Remove rows whose left date is outside tolerance of the right date.
    df_delta = df_sjoin2[abs(df_sjoin2[delta_label_out]) <= tolerance]

    # Because a left row may have multiple matches with the right <df>,
    # keep only the one that is the closest. First, find duplicate rows.
    subset = subset_left + [left_on2, geom_r]
    df_dup = df_delta[df_delta.duplicated(subset=subset, keep=False)]
    # Next, find row with lowest date_delta; if same, just get first.
    df_keep = pd.DataFrame(data=[], columns=df_dup.columns)  # df for selected entries
    df_keep = df_keep.astype(df_dup.dtypes.to_dict())
    df_unique = df_dup.drop_duplicates(subset=subset, keep='first')[subset]
    for idx, row in df_unique.iterrows():  # Unique subset cols only
        # The magic to get duplicate of a particular unique non-null group
        df_filtered = df_dup[df_dup[row.index].isin(row.values).all(1)]
        # Find index where delta is min and append it to df_keep
        idx_min = abs(df_filtered[delta_label_out]).idxmin(axis=0, skipna=True)
        df_keep = df_keep.append(df_filtered.loc[idx_min, :])
    df_join = df_delta.drop_duplicates(subset=subset, keep=False)
    df_join = df_join.append(df_keep)
    df_join = df_join[pd.notnull(df_join[delta_label_out])]
    # drop right join columns,
    if geom_l in df_join.columns:
        df_join = gpd.GeoDataFrame(df_join, geometry=geom_l)
        df_join.rename(columns={geom_l: 'geom'}, inplace=True)
        df_join.set_geometry(col='geom', inplace=True)
    df_join.rename(columns={left_on2:left_on}, inplace=True)
    cols_drop = [c+side for side in ['_l', '_r'] for c in ['id', 'geom', 'geometry']
                 if c+side in df_join.columns] + [right_on2]
    df_join.drop(columns=cols_drop, inplace=True)
    return df_join


def join_closest_date(df_left     : AnyDataFrame,
                      df_right    : AnyDataFrame,
                      left_on     : str = 'date',
                      right_on    : str = 'date',
                      tolerance   : int = 0,
                      direction   : str ='nearest',
                      delta_label : Optional[str] = None
                     ) -> AnyDataFrame:
    '''
    Joins ``df_left`` to ``df_right`` by the closest date (after first
    joining by the ``by`` columns)
    Parameters:
        df_left (``pd.DataFrame``): The left dataframe to join from.
        df_right (``pd.DataFrame``): The right dataframe to join. If
            <df_left> and <df_right> have geometry, only geometry from
            <df_left> will be kept.
        left_on (``str``): The "date" column name in <df_left> to join
            from.
        right_on (``str``): The "date" column name in <df_right> to join
            to.
        tolerance (``int``): Number of days away to still allow join (if
            date_delta is greater than <tolerance>, the join will not
            occur).
        direction (``str``): Whether to search for prior, subsequent, or
            closest matches. This is only implemented if geometry is not
            present.

    Note:
        Parameter names closely follow the pandas.merge_asof function:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_asof.html
    '''
    subset_left = db_utils.get_primary_keys(df_left)
    subset_right = db_utils.get_primary_keys(df_right)
    msg = ('The primary keys in <df_left> are not the same as those in '
           '<df_right>. <join_closest_date()> requires that both dfs have '
           'the same primary keys.')
    assert subset_left == subset_right, msg

    cols_require_l = subset_left + [left_on]
    cols_require_r = subset_right + [right_on]
    table_util.check_col_names(df_left, cols_require_l)
    table_util.check_col_names(df_right, cols_require_r)
    df_left = dt_or_ts_to_date(df_left, left_on)
    df_right = dt_or_ts_to_date(df_right, right_on)

    # 0. Goal here is to join closest date, but when geometry exists, it
    # gets a bit more complicated. When both are geodataframe, any missing
    # geometry is filtered out, then tables are joined on geometry and only
    # the closest date is kept.

    # The problem is when there is empty geometry, then nothing gets joined.

    # Choices:
    # 1. Fill in any missing geometry before the join_closest_date()
    # function is called.
    # 2. Dynamically fill in missing geometry from df_left to df_right
    # during join_closest_date()
    # 3. Raise a warning if both are geodataframes and there is empty
    # geometry.


    if isinstance(df_left, gpd.GeoDataFrame) and isinstance(df_right, gpd.GeoDataFrame):
        check_empty_geom(df_left)
        check_empty_geom(df_right)

        # 2. Case where some geometries are empty, but some are not
        df_join = join_geom_date(
            df_left, df_right, left_on, right_on, tolerance=tolerance,
            delta_label=delta_label)
    else:
        df_left.sort_values(left_on, inplace=True)
        df_right.sort_values(right_on, inplace=True)
        left_on2 = left_on + '_l'
        right_on2 = right_on + '_r'
        # by = subset_left + [df_left.geometry.name]
        df_join = pd.merge_asof(
            df_left.rename(columns={left_on:left_on2}),
            df_right.rename(columns={right_on:right_on2}),
            left_on=left_on2, right_on=right_on2, by=subset_left,
            tolerance=pd.Timedelta(tolerance, unit='D'),
            direction=direction, suffixes=("_l", "_r"))
        if isinstance(df_left, gpd.GeoDataFrame):
            df_join = gpd.GeoDataFrame(
                df_join, geometry=df_left.geometry.name)
        if isinstance(df_right, gpd.GeoDataFrame):
            df_join = gpd.GeoDataFrame(
                df_join, geometry=df_right.geometry.name)

        # if tolerance == 0:
        #     df_join.dropna(inplace=True)
        # elif tolerance > 0:
        df_join, delta_label_out = add_date_delta(
            df_join, left_on=left_on2, right_on=right_on2,
            delta_label=delta_label)
        # df_join.dropna(inplace=True)
        df_join = df_join.rename(columns={left_on2:left_on})
        cols_drop = [c+side for side in ['_l', '_r'] for c in ['id']
                     if c+side in df_join.columns] + [right_on2]
        df_join.drop(columns=cols_drop, inplace=True)
    return df_join.reset_index(drop=True)


