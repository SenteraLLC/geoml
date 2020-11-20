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

from research_tools import Training
import spatial.utilities as spatial_utils


class Predict(Training):
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
        2.
    '''
    __allowed_params = ()

    def __init__(self, **kwargs):
        '''
        '''
        super(Predict, self).__init__(**kwargs)
        self.train(**kwargs)

        self.field_bounds_pred = None
        self.date_predict = datetime.now().date()

        self._set_params_from_kwargs_pred(**kwargs)
        self._set_attributes_train()

    def _set_params_from_dict_pred(self, config_dict):
        '''
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        '''
        if config_dict is not None and 'Prediction' in config_dict:
            params_fd = config_dict['Prediction']
        elif config_dict is not None and 'Prediction' not in config_dict:
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
        if not isinstance(self.field_bounds_pred, gpd.GeoDataFrame):
            df = self.db.get_table_df('field_bounds')
            self.field_bounds_pred = df[df['year'] == self.date_predict.year]

    def _set_attributes_pred(self):
        '''
        Sets any class attribute to ``None`` that will be created in one of the
        user functions from the ``feature_selection`` class
        '''
        print('add variable here.')

    def _get_image_dates(self):
        '''
        Gets the date of each image in the DB.
        '''
        df_date = pd.DataFrame(columns=['raster_name', 'date'])
        for t_name in tables_s3:
            date_img = t_name.split('_')[1]
            df_temp = pd.DataFrame.from_dict({'raster_name': [t_name],
                                              'date': [date_img]})
            df_date = df_date.append(df_temp)
        df_date['date'] =  pd.to_datetime(
            df_date['date'], format='%Y%m%dt%H%M%S')
        return df_date

    def _match_date(self, date_series, date=None, method='nearest'):
        '''
        Finds the closest date from <series> based on the <method>.

        Parameters:
            method (``str``): How to search for the "nearest" date. Must be
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
        msg = ("<method> must be one of ['past', 'future', 'nearest'].")
        assert method in ['past', 'future', 'nearest'], msg
        if date is None:
            date = self.date
        if method == 'nearest':
            date_closest = min(date_series, key=lambda x: abs(x - date))
        elif method == 'past':
            date_closest = max(i for i in date_series if i <= date)
        elif method == 'future':
            date_closest = min(i for i in date_series if i >= date)
        delta = (date - date_closest).days
        return date_closest, delta

    def _find_nearest_date(self, method='past'):
        '''
        Finds the image with the closest date to <Predict.date>.

        Parameters:
            method (``str``): How to search for the "nearest" date. Must be
                one of ['nearest', 'past', 'future']. "past" returns the image
                captured closest in the past, "future" returns the image
                captured closest in the future, and "nearest" returns the image
                captured closest, either in the past or future.

        field_id = 'c-06'
        from datetime import datetime
        date = datetime(2020, 7, 13)
        '''
        # get a list of all raster images in DB
        table_names = self.engine.table_names(schema=self.db.db_schema)
        # tables_s2 = get_table_names(handler.engine, handler.db_schema, col_filter='rast', row_filter={'field_id': field_id}, return_empty=False)
        tables_s2 = spatial_utils.get_table_names(
            self.db.engine, self.db.db_schema, col_filter='rast',
            row_filter={'field_id': self.field_id}, return_empty=False)
        df_date = self._get_image_dates()
        # compare <date> to df_date and match image date based on method
        date_closest, delta = self._match_date(df_date['date'], self.date,
                                               method)
        raster_name = df_date[
            df_date['date'] == date_closest]['raster_name'].item()
        return raster_name


    def _get_X_map(self, method='past'):
        '''
        Retrieves the feature data in the X matrix as a georeferenced map.

        field_id = 'c-08'
        '''
        # get most recent raster
        raster_name = self.find_nearest_date(method)

        # load raster as array
        ds, df_metadata = self.db.get_raster(raster_name, field_id=self.field_id)
        array_img = ds.read()
        array_pred = np.empty(array_img.shape[1:], dtype=float)


# Steps
# 0. In the Training class, pass the image_name and the field_bounds for a single
#    geometry.
# 1. Find the most recent image that has valid pixels for that geometry.
# 2. Load the image in, and use it to build the X matrix "map".
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
# 1. Get an empty raster with the shape of the RS data on that date (can be
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