# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:18:16 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
from copy import deepcopy
from datetime import datetime, date
import os

import geopandas as gpd  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import rasterio as rio  # type: ignore

from .tables.new_columns import dae, dap, rate_ntd

from db import utilities as db_utils
from db import DBHandler
from spatial import utilities as spatial_utils
from spatial import Imagery

from .utils import AnyDataFrame
from .config.config import GroupFeatures


from typing import Dict, Any, List, Optional, Tuple


__allowed_params = ('train', 'estimator', 'feats_x_select', 'date_predict',
                    'primary_keys_pred', 'gdf_pred',
                    'image_search_method', 'refit_X_full', 'dir_out_pred')

def init(**kwargs):
    '''
    '''
    load_tables()

    train = None  # geoml train object
    estimator = None
    feats_x_select = None
    date_predict = datetime.now().date()  # when
    primary_keys_pred = {'owner': None,
                              'farm': None,
                              'field_id': None,
                              'year': None}  # year isn't necessary; overwritten by date_predict.year
    gdf_pred = None  # where
    image_search_method = 'past'
    refit_X_full = False
    dir_out_pred = None

    _set_attributes_pred()
    _set_params_from_kwargs_pred(**kwargs)


def _load_field_bounds(db                : DBHandler,
                       primary_keys_pred : Dict[str, Any],
                       date_predict      : date,
                      ) -> AnyDataFrame:
    '''
    Loads field bounds from DB.
    '''
    primary_keys_pred['year'] = date_predict.year

    gdf = db.get_table_df('field_bounds')
    gdf_pred = gdf.loc[(gdf[list(primary_keys_pred)] ==
                                 pd.Series(primary_keys_pred)
                                 ).all(axis=1)]
    subset = db_utils.get_primary_keys(gdf_pred)
    if len(gdf_pred) == 0:
        print('No field boundaries were loaded. Please be sure '
              '<primary_keys_pred> is properly set and that it '
              'corresponds to an actual field boundary.\n'
              '<primary_keys_pred>: {0}'.format(primary_keys_pred))
    else:
        print('Field boundaries were loaded for the following field(s):')
        for _, r in gdf_pred[subset].iterrows():
            print(r.to_string()+'\n')

    return gdf_pred

def _refit_on_full_X(X_train        : AnyDataFrame,
                     X_test         : AnyDataFrame,
                     labels_x       : List[str],
                     feats_x_select : List[str],
                     X_full_select  : AnyDataFrame,
                     y_full         : AnyDataFrame,
                     y_train        : AnyDataFrame,
                     y_test         : AnyDataFrame,
                     estimator      : Any,
                    ):
    '''
    Refits
    '''
    msg = ('To refit the estimator on the full X matrix, an instance '
           'of the ``Train`` class must be passed as a parameter to '
           '``Predict`` so that ``X_full`` and ``y_full`` can be '
           'created.')
    print('Refitting estimator on full X matrix.')
    X_full = np.concatenate(
        [X_train, X_test], axis=0)

    # get index of labels_x_select in labels_x
    feats_x_select_idx = [i for i in range(len(labels_x))
                          if labels_x[i] in feats_x_select]

    X_full_select = X_full[:,feats_x_select_idx]
    y_full = np.concatenate(
        [y_train, y_test], axis=0)
    estimator.fit(X_full_select, y_full)

def _get_image_dates(tables_s2):
    '''
    Gets the date of each image in the DB.
    '''
    df_date = pd.DataFrame(columns=['raster_name', 'date'])
    for t_name in tables_s2:
        date_img = t_name.split('_')[1]
        df_temp = pd.DataFrame.from_dict({'raster_name': [t_name],
                                          'date': [date_img]})
        df_date = df_date.append(df_temp)
    df_date['date'] =  pd.to_datetime(
        df_date['date'], format='%Y%m%dt%H%M%S')
    return df_date

def _match_date(date_series         : pd.Series,
                date                : date,
                image_search_method : str            = 'nearest'
               ) -> Tuple[date, int]:
    '''
    Finds the closest date from <series> based on the <image_search_method>.

    Parameters:
        image_search_method (``str``): How to search for the "nearest" date. Must be
            one of ['nearest', 'past', 'future']. "past" returns the image
            captured closest in the past, "future" returns the image
            captured closest in the future, and "nearest" returns the image
            captured closest, either in the past or future.
        date_series (``pandas.Series``): A vector containing multiple dates
            to choose from. The returned date will be a date that is
            present in <date_series>.
        date (``datetime``): The date to base the search on.

    Returns:
        date_closest: The closest date.
        delta: The difference (in days) between <date> and <date_closest>.
             A positive delta indicates <date_closest> is in the past, and
             a negative delta indicates <date_closest> is in the future.
    '''
    msg = ("<image_search_method> must be one of ['past', 'future', 'nearest'].")
    assert image_search_method in ['past', 'future', 'nearest'], msg
    if image_search_method == 'nearest':
        date_closest = min(date_series, key=lambda x: abs(x - date))
    elif image_search_method == 'past':
        date_closest = max(i for i in date_series if i <= date)
    elif image_search_method == 'future':
        date_closest = min(i for i in date_series if i >= date)
    delta = (datetime.combine(date, datetime.min.time()) - date_closest).days
    return date_closest, delta

def _find_nearest_raster(db : DBHandler,
                         date_predict : date,
                         primary_key_val : Dict[str, AnyDataFrame],
                         image_search_method : str ='past'
                        ) -> Tuple[Any, int]:
    '''
    Finds the image with the closest date to <Predict.date_predict>.

    Parameters:
        image_search_method (``str``): How to search for the "nearest" date. Must be
            one of ['nearest', 'past', 'future']. "past" returns the image
            captured closest in the past, "future" returns the image
            captured closest in the future, and "nearest" returns the image
            captured closest, either in the past or future.

    Returns:
        raster_name (``str``): The name of the raster that was captured
            closest to <Predict.date_predict>
    '''
    # get a list of all raster images in DB
    tables_s2 = spatial_utils.get_table_names(
        db.engine, db.db_schema, col_filter='rast',
        row_filter=primary_key_val, return_empty=False)
    df_date = _get_image_dates(tables_s2)
    # compare <date> to df_date and match image date based on image_search_method
    date_closest, delta = _match_date(
        df_date['date'], date_predict, image_search_method)
    raster_name = df_date[
        df_date['date'] == date_closest]['raster_name'].item()
    return raster_name, delta

def _get_X_map(db                  : DBHandler,
               date_predict        : date,
               gdf_pred            : AnyDataFrame,
               gdf_pred_s          : AnyDataFrame,
               image_search_method : str,
              ):
    '''
    Retrieves the feature data in the X matrix as a georeferenced map.

    Parameters:
        gdf_pred_s (``pd.Series``): The field_bounds information to create
            the X map for (must be passed as a Series object).
    '''
    subset = db_utils.get_primary_keys(gdf_pred)
    primary_key_val = dict((k, gdf_pred_s[k]) for k in subset)
    raster_name, delta = _find_nearest_raster(db, date_predict, primary_key_val, image_search_method)

    # load raster as array
    array_img, profile, df_metadata = db.get_raster(raster_name, **primary_key_val)
    # array_pred = np.empty(array_img.shape[1:], dtype=float)
    return array_img, profile, df_metadata

def _get_array_img_band_idx(df_metadata, names, col_header='wavelength'):
    '''
    Gets the index of the bands based on feature names from
    <feats_x_select>.

    Parameters:
        df_metadata: The Sentinel metadata dataframe
        names: The names of the Sentinel spectra features from
            <feats_x_select>.
        col_header: Look into why this is needed...

    Returns:
        keys (``dict``): Key/value pairs representing
            <feats_x_select> names/<array_img> band index
    '''
    band_names, wavelengths = db._get_band_wl_from_metadata(
        df_metadata)
    wl_AB_sync = [db.sentinel_AB_sync[b] for b in band_names]
    cols = band_names if col_header == 'bands' else wl_AB_sync
    keys = dict(('wl_{0:.0f}'.format(col), i)
                for i, col in enumerate(cols)
                if 'wl_{0:.0f}'.format(col) in names)
    return keys


def _feats_x_select_data(df_feats       : AnyDataFrame,
                         feats_x_select : List[str],
                         group_feats    : GroupFeatures,
                         df_metadata    : Any,
                         tables         : Dict[str, AnyDataFrame],
                        ):
    '''
    Builds a dataframe with ``feats_x_select`` data.

    Returns:
        df_feats: DataFrame containing each of the feature names as
            column names and the value (if not spatially aware) or
            the array index (if spatially aware; e.g., sentinel image).
    '''
    field_bounds = tables["field_bounds"]
    as_planted = tables.get("as_planted")
    dates_res  = tables.get("dates_res")

    if 'dap' in feats_x_select:
        df_feats = dap(df_feats, field_bounds, as_planted, dates_res)
    if 'dae' in feats_x_select:
        df_feats = dae(df_feats, field_bounds, as_planted, dates_res)

    n_applications = tables.get("n_applications")
    experiments    = tables.get("experiments")
    trt            = tables.get("trt")
    trt_n          = tables.get("trt_n")
    if 'rate_ntd_kgha' in feats_x_select:
        col_rate_n = group_feats['rate_ntd']['col_rate_n']
        col_rate_ntd_out = group_feats['rate_ntd']['col_out']
        df_feats = rate_ntd(df_feats, n_applications, experiments, trt, trt_n, col_rate_n = col_rate_n, col_rate_ntd_out = col_rate_ntd_out)

    if any('wl_' in f for f in feats_x_select):
        # For now, just add null placeholders as a new column
        wl_names = [f for f in feats_x_select if 'wl_' in f]
        keys = _get_array_img_band_idx(df_metadata, wl_names)
        for name in wl_names:
            df_feats[name] = keys[name]
        # TODO: Figure out how to decipher between Sentinel and other sources
    return df_feats


def _fill_array_X(feats_x_select : List[str],
                  array_img      : Any,
                  df_feats : AnyDataFrame
                 ) -> np.ndarray:
    '''
    Populates array_X with all the features in df_feats
    '''
    array_X_shape = (len(feats_x_select),) + array_img.shape[1:]
    array_X = np.empty(array_X_shape, dtype=float)

    for i, feat in enumerate(feats_x_select):
        if 'wl_' in feat:
            array_X[i,:,:] = array_img[df_feats[feat],:,:] / 10000
        else:
            array_X[i,:,:] = df_feats[feat]
    return array_X


def _predict_and_reshape(db        : DBHandler,
                         estimator : Any,
                         array_X   : Any
                        ) -> np.ndarray:
    '''
    Reshapes <array_X> to 2d, predicts, and reshapes back to 3d.
    '''
    array_X_move = np.moveaxis(array_X, 0, -1)  # move first axis to last
    array_X_2d = array_X_move.reshape(-1, array_X_move.shape[2])

    array_pred_1d = estimator.predict(array_X_2d)
    array_pred = array_pred_1d.reshape(array_X_move.shape[:2])
    array_pred = np.expand_dims(array_pred, 0)
    return array_pred


def _mask_by_bounds(db                : DBHandler,
                    array_pred        : np.ndarray,
                    primary_keys_pred : Dict[str, Any],
                    profile           : Any,
                    buffer_dist       : int = 0,
                   ):
    '''
    Masks array by field bounds in ``predict.primary_keys_pred``

    Returns:
        array_pred (``numpy.ndarray``): Input array with masked pixels set
            to 0.
    '''
    profile_out = deepcopy(profile)
    array_pred = array_pred.astype(profile_out['dtype'])
    gdf_bounds = db.get_table_df(  # load field bounds
        'field_bounds', **primary_keys_pred)
    profile_out.update(driver='MEM')
    with rio.io.MemoryFile() as memfile:  # load array_pred as memory object
        with memfile.open(**profile_out) as ds_temp:
            ds_temp.write(array_pred)
            geometry = gdf_bounds[gdf_bounds.geometry.name].to_crs(
                epsg=profile_out['crs'].to_epsg()).buffer(buffer_dist)
            array_pred, _ = rio.mask.mask(ds_temp, geometry, crop=True)
            # adjust profile based on geometry bounds
            # if int(geometry.crs.utm_zone[:-1]) <= 30:
            #     west = geometry.bounds['maxx'].item()
            # else:
            #     west = geometry.bounds['minx'].item()
            profile_out['transform'] = rio.transform.from_origin(
                geometry.bounds['minx'].item(), geometry.bounds['maxy'].item(),
                profile['transform'].a, -profile['transform'].e)
    return array_pred, profile_out

def predict(db                  : DBHandler,
            estimator           : Any,
            primary_keys_pred   : Dict[str, Any],
            group_feats         : GroupFeatures,
            tables              : Dict[str, AnyDataFrame],
            feats_x_select      : List[str],
            date_predict        : date,
            image_search_method : str,
            mask_by_bounds      : bool                   = True,
            gdf_pred_s          : Optional[AnyDataFrame] = None,
            buffer_dist         : int                    = -40,
            clip_min            : int                    = 0,
            clip_max            : Optional[int]          = None,
           ) -> Tuple[np.ndarray, Any]:
    '''
    Makes predictions for a single geometry.

    For spatially aware predictions. Builds a 3d array that contains
    spatial data (x/y) as well as feaure data (z). The 3d array is reshaped
    to 2d (x*y by z shape) and passed to the ``estimator.predict()``
    function. After predictions are made, the 1d array is reshaped to 2d
    and loaded as a rasterio object.

    Parameters:
        gdf_pred_s (``geopandas.GeoSeries``): The geometry to make the
            prediction on.
        mask_by_bounds (``bool``): Whether prediction array should be
            masked by field bounds (``True``) or not (``False``).
        buffer_dist (``int`` or ``float``): The buffer distance to
            apply to mask boundary. Negative ``buffer_dist`` results
            in a smaller array/polygon, whereas a positive
            ``buffer_dist`` results in a larger array/polygon. Units
            should be equal to units used by the coordinate reference
            system of the rasterio profile object from imagery. Ignored
            if ``mask_by_bounds`` is ``False``.
        clip_min, clip_max (``int`` or ``float``): Minimum and maximum
            value. If ``None``, clipping is not performed on the
            corresponding edge. Only one of ``clip_min`` and ``clip_max``
            may be ``None``. Passed to the ``numpy.clip()`` function as
            ``min`` and ``max`` parameters, respectively.
    '''
    gdf_pred = _load_field_bounds(db, primary_keys_pred, date_predict)

    print('Making predictions on new data...')

    if not isinstance(gdf_pred_s, pd.Series):
        # print('Using the first row ')
        gdf_pred_s = gdf_pred.iloc[0]

    subset = db_utils.get_primary_keys(gdf_pred_s)
    array_img, profile, df_metadata = _get_X_map(db, date_predict, gdf_pred, gdf_pred_s, image_search_method)

    # 1. Get features for the model of interest
    cols_feats = subset + ['geom', 'date']  # df must have primary keys

    df_feats = gpd.GeoDataFrame(
        [list(gdf_pred_s[subset]) + [gdf_pred_s.geom, date_predict]],
        columns=cols_feats, geometry='geom', crs=gdf_pred.crs)
    # df_feats = pd.DataFrame(data=[data_feats], columns=cols_feats)


    df_feats = _feats_x_select_data(df_feats, feats_x_select, group_feats, df_metadata, tables)
    weather_derived = tables["weather_derived"]
    # TODO: change when we get individual functions for each wx feature
    if any([f for f in feats_x_select if f in weather_derived.columns]):
        primary_key_val = dict((k, gdf_pred_s[k]) for k in subset)
        primary_key_val['date'] = date_predict
        weather_derived_filter = weather_derived.loc[
            (weather_derived[list(primary_key_val)] ==
             pd.Series(primary_key_val)).all(axis=1)]
        for f in set(feats_x_select).intersection(weather_derived.columns):
            # get its value from weather_derived and add to df_feats
            df_feats[f] = weather_derived_filter[f].values[0]

    array_X = _fill_array_X(feats_x_select, array_img, df_feats)
    mask = array_img[0] == 0

    array_pred = _predict_and_reshape(db, estimator, array_X)
    array_pred[np.expand_dims(mask, 0)] = 0
    if any(v is not None for v in [clip_min, clip_max]):
        array_pred = array_pred.clip(min=clip_min, max=clip_max)
    # array_pred = np.ma.masked_array(data=array_pred, mask=mask_2d)
    profile.update(count=1)
    if mask_by_bounds == True:
        array_pred, profile = _mask_by_bounds(db, array_pred, primary_keys_pred, profile, buffer_dist)
    return array_pred, profile


#def predict_and_save():
#    '''
#    Makes predictions for each geometry in ``gdf_pred`` and saves output
#    as an image.
#
#    This function does 2 tasks in addition to predict(): batch predicts for
#    all geom in ``gdf_pred``, and saves predictions as an image.
#
#    For spatially aware predictions. Builds a 3d array that contains
#    spatial data (x/y) as well as feaure data (z). The 3d array is reshaped
#    to 2d (x*y by z shape) and passed to the ``estimator.predict()``
#    function. After predictions are made, the 1d array is reshaped to 2d
#    and loaded as a rasterio object.
#    '''
#
#    array_preds, df_metadatas = [], []
#    imagery = Imagery()
#    imagery.driver = 'Gtiff'
#    for _, gdf_pred_s in gdf_pred.iterrows():
#        array_pred, ds, df_metadata = predict(gdf_pred_s)
#        array_preds.append(array_pred)
#        df_metadatas.append(df_metadata)
#
#        name_out = 'petno3_ppm_20200713_r20m_css-farms-dalhart_cabrillas_c-06.tif'
#
#        fname_out = os.path.join(dir_out_pred, name_out)
#        imagery._save_image(np.expand_dims(array_pred, axis=0), ds.profile, fname_out, keep_xml=False)
#
#    return array_preds, df_metadatas
