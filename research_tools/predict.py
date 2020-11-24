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
from datetime import datetime
import geopandas as gpd
import numpy as np
import pandas as pd

from db import DBHandler
import db.utilities as db_utils
import spatial.utilities as spatial_utils
from research_tools import Tables
from research_tools import Training


class Predict(Tables):
    '''
    ``Predict`` inherits from an instance of ``Training``, and consists of
    variables and functions to carry out predictions on new data with the
    previously trained models.

    Some things to be determined:
        1.  Should this class predict a single primary keyset at at time (e.g.,
            a single owner/farm/field_id/year/date) for a given geometry? Or
            should it look for all fields in the field_bounds table for this
            year and try to make predictions on each one?
            To start, just take a single field (it can be switched out for new
            predictions pretty easily).
        2.  Should we inherit from training, or just have the ability to pass
            a ``Training`` instance to ``Predict``? Not having the dependence
            on ``Training`` would be valuable so ``Predict`` could be used as
            a standalone module where the only thing passed is the
            tuned/trained estimator. With this, if ``Training`` is passed,
            there might be deeper functionality not available if an estimator
            is simply passed.
        3.  Right now, predict() inherits from train, but we don't really want
            the predict class to depend on train. For example, if we already
            have a trained estimator and its corresponding params, we would like
            to be able to run predict using that estimator. A way to implent
            this is to have an optional parameter "train"
    '''
    __allowed_params = ('train', 'estimator', 'feats_x_select', 'date_predict',
                        'primary_keys_pred', 'gdf_pred',
                        'image_search_method', 'refit_X_full')

    def __init__(self, **kwargs):
        '''
        '''
        super(Predict, self).__init__(**kwargs)
        self.load_tables()
        # super(Predict, self).__init__(**kwargs)
        # self.train(**kwargs)

        self.train = None  # research_tools train object
        self.loc_df_test = None
        self.estimator = None
        self.feats_x_select = None
        self.date_predict = datetime.now().date()  # when
        self.primary_keys_pred = {'owner': None,
                                  'farm': None,
                                  'field_id': None,
                                  'year': None}  # year isn't necessary; overwritten by date_predict.year
        self.gdf_pred = None  # where
        self.image_search_method = 'past'
        self.refit_X_full = False

        self._set_params_from_kwargs_pred(**kwargs)
        self._set_attributes_pred()

    def _set_params_from_dict_pred(self, config_dict):
        '''
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        '''
        if config_dict is not None and 'Predict' in config_dict:
            params_fd = config_dict['Predict']
        elif config_dict is not None and 'Predict' not in config_dict:
            params_fd = config_dict
        else:  # config_dict is None
            return
        for k, v in params_fd.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, v)

    def _set_params_from_kwargs_pred(self, **kwargs):
        '''
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'config_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``Prediction`` more explicitly.
        '''
        if 'config_dict' in kwargs:
            self._set_params_from_dict_pred(kwargs.get('config_dict'))
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)
        if self.date_predict is None:
            self.date_predict = datetime.now().date()
        elif isinstance(self.date_predict, str):
            try:
                self.date_predict = datetime.strptime(self.date_predict,
                                                      '%Y-%m-%d')
            except ValueError as e:
                raise ValueError('{0}.\nPlease either pass a datetime object, '
                                 'or a string in the "YYYY-mm-dd" format.'
                                 ''.format(e))

        if isinstance(self.train, Training) and self.loc_df_test:
            if self.estimator and self.feats_x_select:
                print('<predict.estimator> and <predict.feats_x_select> are '
                      'being overwritten by <predict.train.df_test.loc[loc_df_test]>')
            self.estimator = self.train.df_test.loc[self.loc_df_test,'regressor']
            self.feats_x_select = self.train.df_test.loc[self.loc_df_test,'feats_x_select']

        if not self.db and isinstance(self.train, Training):
            if self.train.db:
                self.db = self.train.db
        if not isinstance(self.gdf_pred, gpd.GeoDataFrame):
            self._load_field_bounds()

    def _set_attributes_pred(self):
        '''
        Sets any class attribute to ``None`` that will be created in one of the
        user functions from the ``feature_selection`` class
        '''
        print('add variable here.')

    def _load_field_bounds(self):
        '''
        Loads field bounds from DB.
        '''
        msg = ('<primary_keys_pred> must be set to load field bounds from DB.')
        assert self.primary_keys_pred != None, msg
        self.primary_keys_pred['year'] = self.date_predict.year

        gdf = self.db.get_table_df('field_bounds')
        gdf_pred = gdf.loc[(gdf[list(self.primary_keys_pred)] ==
                                     pd.Series(self.primary_keys_pred)
                                     ).all(axis=1)]
        subset = db_utils.get_primary_keys(gdf_pred)
        if len(gdf_pred) == 0:
            print('No field boundaries were loaded. Please be sure '
                  '<primary_keys_pred> is properly set and that it '
                  'corresponds to an actual field boundary.\n'
                  '<primary_keys_pred>: {0}'.format(self.primary_keys_pred))
        else:
            print('Field boundaries were loaded for the following field(s):')
            for _, r in gdf_pred[subset].iterrows():
                print(r.to_string()+'\n')
        self.gdf_pred = gdf_pred

    def _get_image_dates(self, tables_s2):
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

    def _match_date(self, date_series, date=None, image_search_method='nearest'):
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
        if date is None:
            date = self.date
        if image_search_method == 'nearest':
            date_closest = min(date_series, key=lambda x: abs(x - date))
        elif image_search_method == 'past':
            date_closest = max(i for i in date_series if i <= date)
        elif image_search_method == 'future':
            date_closest = min(i for i in date_series if i >= date)
        delta = (date - date_closest).days
        return date_closest, delta

    def _find_nearest_raster(self, primary_key_val, image_search_method='past'):
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
            self.db.engine, self.db.db_schema, col_filter='rast',
            row_filter=primary_key_val, return_empty=False)
        df_date = self._get_image_dates(tables_s2)
        # compare <date> to df_date and match image date based on image_search_method
        date_closest, delta = self._match_date(
            df_date['date'], self.date_predict, image_search_method)
        raster_name = df_date[
            df_date['date'] == date_closest]['raster_name'].item()
        return raster_name, delta

    def _get_X_map(self, gdf_pred_s):
        '''
        Retrieves the feature data in the X matrix as a georeferenced map.

        Parameters:
            gdf_pred_s (``pd.Series``): The field_bounds information to create
                the X map for (must be passed as a Series object).
        '''
        subset = db_utils.get_primary_keys(self.gdf_pred)
        primary_key_val = dict((k, gdf_pred_s[k]) for k in subset)
        raster_name, delta = self._find_nearest_raster(
            primary_key_val, self.image_search_method)

        # load raster as array
        ds, df_metadata = self.db.get_raster(raster_name, **primary_key_val)
        array_img = ds.read()
        # array_pred = np.empty(array_img.shape[1:], dtype=float)
        return array_img, df_metadata

    def _get_array_img_band_idx(self, df_metadata, names, col_header='wavelength'):
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
        band_names, wavelengths = self.db._get_band_wl_from_metadata(
            df_metadata)
        wl_AB_sync = [self.db.sentinel_AB_sync[b] for b in band_names]
        cols = band_names if col_header == 'bands' else wl_AB_sync
        keys = dict(('wl_{0:.0f}'.format(col), i)
                    for i, col in enumerate(cols)
                    if 'wl_{0:.0f}'.format(col) in names)
        return keys

    def predict(self, **kwargs):
        '''
        Makes predictions.
        '''
        print('Making predictions on new data...')
        self._set_params_from_kwargs_pred(**kwargs)

        array_preds, array_imgs, df_metadatas = [], [], []
        for _, gdf_pred_s in self.gdf_pred.iterrows():
            subset = db_utils.get_primary_keys(gdf_pred_s)
            array_img, df_metadata = self._get_X_map(gdf_pred_s)
            array_imgs.append(array_img)
            df_metadatas.append(df_metadata)
            # 2. Enter the estimator "features" (columns in the ML X matrix) as
            #    the 3rd dimension in a 3D array (D1 and D2 are the spatial
            #    dimensions from <array_img>).

            # 1. Get features for the model of interest
            # How should we decide the model of interest?

            cols_feats = subset + ['date']  # df must have primary keys
            data_feats = list(gdf_pred_s[subset]) + [self.date_predict]
            df_feats = pd.DataFrame(data=[data_feats], columns=cols_feats)

            feats_x_select = self.feats_x_select

            if 'dap' in feats_x_select:
                df_feats = self.dap(df_feats)
            if 'dae' in feats_x_select:
                df_feats = self.dae(df_feats)
            if 'rate_ntd_kgha' in feats_x_select:
                df_feats = self.rate_ntd(df_feats)
            if any('wl_' in f for f in feats_x_select):
                # For now, just add null placeholders as a new column
                wl_names = [f for f in feats_x_select if 'wl_' in f]
                keys = self._get_array_img_band_idx(df_metadata, wl_names)
                for name in wl_names:
                    df_feats[name] = keys[name]
                # TODO: Figure out how to decipher between Sentinel and other sources
                # grab bands from array_img according to df_metadata
            # TODO: change when we get individual functions for each wx feature
            if any([f for f in feats_x_select if f in self.weather_derived.columns]):
                primary_key_val = dict((k, gdf_pred_s[k]) for k in subset)
                primary_key_val['date'] = self.date_predict
                weather_derived_filter = self.weather_derived.loc[
                    (self.weather_derived[list(primary_key_val)] ==
                     pd.Series(primary_key_val)).all(axis=1)]
                for f in set(feats_x_select).intersection(self.weather_derived.columns):
                    # get its value from weather_derived and add to df_feats
                    df_feats[f] = weather_derived_filter[f].values[0]

            # Now that we have all features in df_feats, populate array_feat
            array_X_shape = (len(feats_x_select),) + array_img.shape[1:]
            array_X = np.empty(array_X_shape, dtype=float)

            for i, feat in enumerate(feats_x_select):
                if 'wl_' in feat:
                    array_X[i,:,:] = array_img[df_feats[feat],:,:] / 10000
                else:
                    array_X[i,:,:] = df_feats[feat]

            # Reshape to 2d, predict, and reshape back to 3d
            self.array_X = array_X
            # print(self.array_X.shape)
            # import numpy as np
            array_X = self.array_X

            array_X_move = np.moveaxis(array_X, 0, -1)  # move first axis to last
            array_X_2d = array_X_move.reshape(-1, array_X_move.shape[2])

            # a = array_X_2d[287].reshape(1, -1)
            # b = predict.estimator.predict(array_X_2d)
            # arr_pred = b.reshape(array_X_move.shape[:2])

            # array_X_2d = array_X.reshape(0, array_X.shape[0])

            array_pred_1d = self.estimator.predict(array_X_2d)
            array_pred = array_pred_1d.reshape(array_X_move.shape[:2])

            # Finally mask out invalid pixels (from geometry?)
            array_preds.append(array_pred)
        return array_preds, array_imgs, df_metadatas

# plt.imshow(arr_pred, interpolation='nearest')
# plt.show()

def extra():
    # Steps
    # 0. [DONE] In the Training class, pass the image_name and the field_bounds for a single
    #    geometry.
    # 1. [DONE] Find the most recent image that has valid pixels for that geometry.
    # 2. [DONE] Load the image in, and use it to build the X matrix "map".
    # 3. Make predictions for each "pixel" and output to a georeferenced raster to
    #    be visualized.


    import numpy
    arr = numpy.random.rand(50,100,25)
    arr_2d = arr.reshape(-1, arr.shape[-1])
    print(arr_2d.shape)
    arr_3d = arr_2d.reshape(arr.shape)
    print(arr_2d.shape)
    np.all(arr == arr_3d)

    # to go from 1d pred to 2d array. arr_pred is what is expected from estimator.predict()
    arr_pred = arr_2d[:,0]
    arr_pred_2d = arr_pred.reshape(arr.shape[:2])

    # In[Build X matrix for spatially-aware predictions]
    # 1. [DONE] Get an empty raster with the shape of the RS data on that date (can be
    #    resampled if desired).
    import numpy as np

    raster_name = 's2a_20180521t172901_msi_r20m_css_farms_dalhart'
    field_id = 'c-08'
    ds, df_metadata = handler.get_raster(raster_name, field_id=field_id)
    array_img = ds.read()
    array_pred = np.empty(array_img.shape[1:], dtype=float)

    # 2. Processing one field_id at a time, treat the columns in the ML X matrix as
    #    the 3rd dimension in a 3D array (where D1 and D2 are spatial dimensions)
    #    that can be made into an image/geotiff.
    col_header = 'wavelength'
    band_names, wavelengths = handler._get_band_wl_from_pg_raster(raster_name, rid=1)
    cols = band_names if col_header == 'bands' else wavelengths
    wl_AB_sync = [handler.sentinel_AB_sync[b] for b in band_names]
    cols = band_names if col_header == 'bands' else wl_AB_sync
    keys = ['wl_{0:.0f}'.format(col) for col in cols]



    feats_x_select = ('dap', 'rate_ntd_kgha', 'wl_665', 'wl_1612', 'wl_864',
                      'gdd_cumsum_plant_to_date')
    array_X_shape = array_img.shape[1:]
    array_X_shape = (len(feats_x_select),) + array_img.shape[1:]
    band_shape = (1,) + array_img.shape[1:]
    array_X = np.empty(array_X_shape, dtype=float)
    # array_X = None

    for i, feat in enumerate(feats_x_select):
        print(i, feat)
        if 'wl_' in feat:  # grab reflectance band from ds.read(band_n)
            idx = keys.index(feat)
            array_band = np.expand_dims(array_img[idx,:,:], axis=0)
        elif feat == 'dap':
            array_band = np.empty(band_shape)
        elif feat == 'dae':
            array_band = np.empty(band_shape)
        elif feat == 'gdd_cumsum_plant_to_date':
            array_band = np.empty(band_shape)
        else:
            array_band = np.empty(band_shape)
        array_X[i,:,:] = array_band
    array_X.mean(axis=(1,2))

    # 3. Pass the reshaped 3D array (now a 2D array where 3rd dimension is cols) to
    #    the trained scikit-learn model to make new predictions.
    # 4. Reshape back into the 3D array, making sure it's just the inverse of the
    #    first reshape so spatial integrity is maintained.
    # 5. Save the resulting prediction array as a geotiff (and optionally save the
    #    input "Xarray" image).
    # 6. Consider storing another image providing an estimate of +/- of predicted
    #    value (based on where it lies in the "predicted" axis of the
    #    measured/predicted cross-validated test set.)