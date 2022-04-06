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
import os
import pandas as pd
import rasterio as rio
from rasterio.mask import mask as rio_mask

from db import DBHandler

# from spatial import Imagery
import db.utilities as db_utils
from db.sql.sql_constructors import closest_date_sql

# import spatial.utilities as spatial_utils
from geoml import Tables
from geoml import Training


class Predict(Tables):
    """
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
    """

    __allowed_params = (
        "train",
        "estimator",
        "feats_x_select",
        "date_predict",
        "primary_keys_pred",
        "gdf_pred",
        "image_search_method",
        "refit_X_full",
        "dir_out_pred",
    )

    def __init__(self, **kwargs):
        """ """
        super(Predict, self).__init__(**kwargs)
        self.load_tables()
        # super(Predict, self).__init__(**kwargs)
        # self.train(**kwargs)

        self.train = None  # geoml train object
        self.loc_df_test = None
        self.estimator = None
        self.feats_x_select = None
        self.date_predict = datetime.now().date()  # when
        self.primary_keys_pred = {
            "owner": None,
            "farm": None,
            "field_id": None,
            "year": None,
        }  # year isn't necessary; overwritten by date_predict.year
        self.gdf_pred = None  # where
        self.image_search_method = "past"
        self.refit_X_full = False
        self.dir_out_pred = None

        self._set_attributes_pred()
        self._set_params_from_kwargs_pred(**kwargs)

    def _set_params_from_dict_pred(self, config_dict):
        """
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        """
        if config_dict is not None and "Predict" in config_dict:
            params_p = config_dict["Predict"]
        elif config_dict is not None and "Predict" not in config_dict:
            params_p = config_dict
        else:  # config_dict is None
            return
        for k, v in params_p.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, v)

    def _set_params_from_kwargs_pred(self, **kwargs):
        """
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'config_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``Prediction`` more explicitly.
        """
        if "config_dict" in kwargs:
            config_dict = kwargs.get("config_dict")
            self._set_params_from_dict_pred(config_dict)
        else:
            config_dict = None

        if config_dict is not None and "Predict" in config_dict:
            params_p = config_dict["Predict"]
        elif config_dict is not None and "Predict" not in config_dict:
            params_p = config_dict
        else:
            params_p = deepcopy(kwargs)

        if len(kwargs) > 0:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)

        if self.date_predict is None:
            self.date_predict = datetime.now().date()
        elif isinstance(self.date_predict, str):
            try:
                self.date_predict = datetime.strptime(self.date_predict, "%Y-%m-%d")
            except ValueError as e:
                raise ValueError(
                    "{0}.\nPlease either pass a datetime object, or a string "
                    'in the "YYYY-mm-dd" format.'.format(e)
                )

        if isinstance(self.train, Training) and self.loc_df_test:
            if self.estimator and self.feats_x_select:
                print(
                    "<predict.estimator> and <predict.feats_x_select> are "
                    "being overwritten by <predict.train.df_test.loc[loc_df_test]>"
                )
            self.estimator = self.train.df_test.loc[self.loc_df_test, "regressor"]
            self.feats_x_select = self.train.df_test.loc[
                self.loc_df_test, "feats_x_select"
            ]

        if not self.db and isinstance(self.train, Training):
            if self.train.db:
                self.db = self.train.db
        if (
            not isinstance(self.gdf_pred, gpd.GeoDataFrame)
            or "primary_keys_pred" in kwargs
            or "primary_keys_pred" in params_p
        ):
            self._load_field_bounds()

        if self.refit_X_full is True:
            if "estimator" in kwargs or "estimator" in params_p:
                self._refit_on_full_X()

    def _set_attributes_pred(self):
        """
        Sets any class attribute to ``None`` that will be created in one of the
        user functions from the ``feature_selection`` class
        """
        self.X_full = None
        self.y_full = None

    def _load_field_bounds(self):
        """
        Loads field bounds from DB.
        """
        msg = "<primary_keys_pred> must be set to load field bounds from DB."
        assert self.primary_keys_pred != None, msg
        self.primary_keys_pred["year"] = self.date_predict.year

        gdf_pred = self.db.get_table_df("field_bounds", **self.primary_keys_pred)
        subset = db_utils.get_primary_keys(gdf_pred)
        if len(gdf_pred) == 0:
            print(
                "No field boundaries were loaded. Please be sure "
                "<primary_keys_pred> is properly set and that it "
                "corresponds to an actual field boundary.\n"
                "<primary_keys_pred>: {0}".format(self.primary_keys_pred)
            )
        else:
            print("Field boundaries were loaded for the following field(s):")
            for _, r in gdf_pred[subset].iterrows():
                print(r.to_string() + "\n")
        self.gdf_pred = gdf_pred

    def _refit_on_full_X(self):
        """
        Refits
        """
        msg = (
            "To refit the estimator on the full X matrix, an instance "
            "of the ``Train`` class must be passed as a parameter to "
            "``Predict`` so that ``X_full`` and ``y_full`` can be "
            "created."
        )
        assert self.train is not None, msg
        print("Refitting estimator on full X matrix.")
        X_full = np.concatenate([self.train.X_train, self.train.X_test], axis=0)

        # get index of labels_x_select in labels_x
        feats_x_select_idx = [
            i
            for i in range(len(self.train.labels_x))
            if self.train.labels_x[i] in self.feats_x_select
        ]

        self.X_full_select = X_full[:, feats_x_select_idx]
        self.y_full = np.concatenate([self.train.y_train, self.train.y_test], axis=0)
        self.estimator.fit(self.X_full_select, self.y_full)

    def _get_X_map(self, gdf_pred_s):
        """
        Retrieves the feature data in the X matrix as a georeferenced map.

        Parameters:
            gdf_pred_s (``pd.Series``): The field_bounds information to create
                the X map for (must be passed as a Series object).
        """
        subset = db_utils.get_primary_keys(self.gdf_pred)
        primary_key_val = dict((k, gdf_pred_s[k]) for k in subset)

        sql_closest_date = closest_date_sql(
            db_schema=self.db.db_schema,
            pkey=primary_key_val,
            date_str=self.date_predict.strftime(format="%Y-%m-%d"),
            tolerance=None,
            direction=self.image_search_method,
            limit=1,
        )
        df_reflectance = pd.read_sql(sql_closest_date, con=self.db.engine)
        if len(df_reflectance) != 1:
            raise ValueError(
                f"There aren't any rasters for <{primary_key_val}>. Please add images "
                "or change <date_predict> to be a later date."
            )

        array_img, profile, df_metadata = self.db.get_raster(
            table_name="reflectance", rid=df_reflectance.iloc[0]["rid"]
        )
        return array_img, profile, df_metadata

    def _get_array_img_band_idx(
        self, df_metadata, names, col_header="wavelength", tol=10
    ):
        """
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
        """
        band_names, wavelengths = self.db._get_band_wl_from_metadata(df_metadata)
        wl_AB_sync = [self.db.sentinel_AB_sync[b] for b in band_names]
        cols = band_names if col_header == "bands" else wl_AB_sync
        keys = dict(
            (n, cols.index(min(cols, key=lambda x: abs(x - int(n.split("_")[-1])))))
            for n in names
            if abs(
                min(cols, key=lambda x: abs(x - int(n.split("_")[-1])))
                - int(n.split("_")[-1])
            )
            < tol
        )
        return keys

    def _feats_x_select_data(self, df_feats, df_metadata):
        """
        Builds a dataframe with ``feats_x_select`` data.

        Returns:
            df_feats: DataFrame containing each of the feature names as
                column names and the value (if not spatially aware) or
                the array index (if spatially aware; e.g., sentinel image).
        """
        if "dap" in self.feats_x_select:
            df_feats = self.dap(df_feats)
        if "dae" in self.feats_x_select:
            df_feats = self.dae(df_feats)
        if "rate_ntd_kgha" in self.feats_x_select:
            df_feats = self.rate_ntd(df_feats)
        if any("wl_" in f for f in self.feats_x_select):
            # For now, just add null placeholders as a new column
            wl_names = [f for f in self.feats_x_select if "wl_" in f]
            keys = self._get_array_img_band_idx(df_metadata, wl_names, tol=10)
            for name in wl_names:
                df_feats[name] = keys[name]
            # TODO: Figure out how to decipher between Sentinel and other sources
        return df_feats

    def _fill_array_X(self, array_img, df_feats):
        """
        Populates array_X with all the features in df_feats
        """
        array_X_shape = (len(self.feats_x_select),) + array_img.shape[1:]
        array_X = np.empty(array_X_shape, dtype=float)

        for i, feat in enumerate(self.feats_x_select):
            if "wl_" in feat:
                array_X[i, :, :] = array_img[df_feats[feat], :, :] / 10000
            else:
                array_X[i, :, :] = df_feats[feat]
        return array_X

    def _predict_and_reshape(self, array_X):
        """
        Reshapes <array_X> to 2d, predicts, and reshapes back to 3d.
        """
        array_X_move = np.moveaxis(array_X, 0, -1)  # move first axis to last
        array_X_2d = array_X_move.reshape(-1, array_X_move.shape[2])

        array_pred_1d = self.estimator.predict(array_X_2d)
        array_pred = array_pred_1d.reshape(array_X_move.shape[:2])
        array_pred = np.expand_dims(array_pred, 0)
        return array_pred

    def _mask_by_bounds(self, array_pred, profile, buffer_dist=0):
        """
        Masks array by field bounds in ``predict.primary_keys_pred``

        Returns:
            array_pred (``numpy.ndarray``): Input array with masked pixels set
                to 0.
        """
        profile_out = deepcopy(profile)
        array_pred = array_pred.astype(profile_out["dtype"])
        gdf_bounds = self.db.get_table_df(  # load field bounds
            "field_bounds", **self.primary_keys_pred
        )
        profile_out.update(driver="MEM")
        with rio.io.MemoryFile() as memfile:  # load array_pred as memory object
            with memfile.open(**profile_out) as ds_temp:
                ds_temp.write(array_pred)

                # convert from 4326 to utm
                srid_utm = self.db._get_utm_epsg(gdf_bounds, how="centroid")

                geometry = (
                    gdf_bounds[gdf_bounds.geometry.name]
                    .to_crs(epsg=srid_utm)
                    .buffer(buffer_dist)
                    .to_crs(epsg=profile_out["crs"].to_epsg())
                )
                array_pred, _ = rio_mask(ds_temp, geometry, crop=True)
                profile_out["transform"] = rio.transform.from_origin(
                    geometry.bounds["minx"].item() - (profile["transform"].a / 2),
                    geometry.bounds["maxy"].item() - (profile["transform"].e / 2),
                    profile["transform"].a,
                    -profile["transform"].e,
                )
        return array_pred, profile_out

    def _update_profile(self, profile):
        from geoml import __version__

        p = deepcopy(profile)
        if self.train is None:
            date_train = None
            response = None
            band_names = ["Sentera GeoML Prediction"]
        else:
            date_train = self.train.date_train.strftime("%Y-%m-%d")
            response = self.train.response_data
            band_names = [
                self.train.response_data["tissue"]
                + "-"
                + self.train.response_data["measure"]
            ]

        p.update(
            count=1,
            bands=1,
            description={
                "date_train": date_train,
                "response": response,
                "date_predict": self.date_predict.strftime("%Y-%m-%d"),
                "feats_x_select": self.feats_x_select,
                "estimator": str(self.estimator).replace(" ", "").replace("\n", ""),
                "parameters": self.estimator.get_params(),
                "geoml_version": __version__,
            },
        )
        p["band names"] = band_names
        del (
            p["wavelength"],
            p["fwhm"],
            p["reflectance scale factor"],
            p["sensor type"],
            p["wavelength units"],
        )
        return p

    def predict(
        self,
        gdf_pred_s=None,
        mask_by_bounds=True,
        buffer_dist=-40,
        clip_min=0,
        clip_max=None,
        **kwargs,
    ):
        """
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
        """
        print("Making predictions on new data...")
        self._set_params_from_kwargs_pred(**kwargs)

        if not isinstance(gdf_pred_s, pd.Series):
            # print('Using the first row ')
            gdf_pred_s = self.gdf_pred.iloc[0]

        subset = db_utils.get_primary_keys(self.gdf_pred)
        array_img, profile, df_metadata = self._get_X_map(gdf_pred_s)

        # 1. Get features for the model of interest
        cols_feats = subset + ["geom", "date"]  # df must have primary keys
        # data_feats = list(gdf_pred_s[subset]) + [self.date_predict]

        df_feats = gpd.GeoDataFrame(
            [list(gdf_pred_s[subset]) + [gdf_pred_s.geom, self.date_predict]],
            columns=cols_feats,
            geometry="geom",
            crs=self.gdf_pred.crs,
        )
        # df_feats = pd.DataFrame(data=[data_feats], columns=cols_feats)

        df_feats = self._feats_x_select_data(df_feats, df_metadata)
        # TODO: change when we get individual functions for each wx feature
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if any([f for f in self.feats_x_select if f in self.weather_derived.columns]):
        #     primary_key_val = dict((k, gdf_pred_s[k]) for k in subset)
        #     primary_key_val["date"] = self.date_predict
        #     weather_derived_filter = self.weather_derived.loc[
        #         (
        #             self.weather_derived[list(primary_key_val)]
        #             == pd.Series(primary_key_val)
        #         ).all(axis=1)
        #     ]
        #     for f in set(self.feats_x_select).intersection(
        #         self.weather_derived.columns
        #     ):
        #         # get its value from weather_derived and add to df_feats
        #         df_feats[f] = weather_derived_filter[f].values[0]

        array_X = self._fill_array_X(array_img, df_feats)
        mask = array_img[0] == 0

        array_pred = self._predict_and_reshape(array_X)
        array_pred[np.expand_dims(mask, 0)] = 0
        if any(v is not None for v in [clip_min, clip_max]):
            array_pred = array_pred.clip(min=clip_min, max=clip_max)
        profile = self._update_profile(profile)
        if mask_by_bounds == True:
            array_pred, profile = self._mask_by_bounds(array_pred, profile, buffer_dist)
        return array_pred, profile

    def predict_and_save(self, **kwargs):
        """
        Makes predictions for each geometry in ``gdf_pred`` and saves output
        as an image.

        This function does 2 tasks in addition to predict(): batch predicts for
        all geom in ``gdf_pred``, and saves predictions as an image.

        For spatially aware predictions. Builds a 3d array that contains
        spatial data (x/y) as well as feaure data (z). The 3d array is reshaped
        to 2d (x*y by z shape) and passed to the ``estimator.predict()``
        function. After predictions are made, the 1d array is reshaped to 2d
        and loaded as a rasterio object.
        """
        print("Making predictions on new data...")
        self._set_params_from_kwargs_pred(**kwargs)

        array_preds, df_metadatas = [], []
        imagery = Imagery()
        imagery.driver = "Gtiff"
        for _, gdf_pred_s in self.gdf_pred.iterrows():
            array_pred, ds, df_metadata = self.predict(gdf_pred_s)
            array_preds.append(array_pred)
            df_metadatas.append(df_metadata)

            name_out = "petno3_ppm_20200713_r20m_css-farms-dalhart_cabrillas_c-06.tif"

            fname_out = os.path.join(self.dir_out_pred, name_out)
            imagery._save_image(
                np.expand_dims(array_pred, axis=0),
                ds.profile,
                fname_out,
                keep_xml=False,
            )

        return array_preds, df_metadatas

    # 5. Save the resulting prediction array as a geotiff (and optionally save the
    #    input "Xarray" image).
    # 6. Consider storing another image providing an estimate of +/- of predicted
    #    value (based on where it lies in the "predicted" axis of the
    #    measured/predicted cross-validated test set.)


#     def save_preds(self, fname_out):
#         '''
#         Saves a
#         '''

# fname_out = r'G:\Shared drives\Data\client_data\CSS Farms\preds_prototype\petiole-no3_20200714T172859_CSS-Farms-Dalhart_Cabrillas_C-18.tif'
# my_imagery = Imagery()
# my_imagery.driver = 'Gtiff'
# rast = rio.open(fname_img)
# metadata = rast.meta
# my_imagery._save_image(array_pred, metadata, fname_out, keep_xml=False)

# fname = r'G:\Shared drives\Data\client_data\CSS Farms\preds_prototype\petiole-no3_20200714T172859_CSS-Farms-Dalhart_Cabrillas_C-18.tif'
# with rasterio.open(fname) as src:
#     with rasterio.Env():
#         profile = ds.profile
#         profile = {'driver': 'GTiff',
#                    'dtype': 'uint16',
#                    'nodata': 0.0,
#                    'width': 48,
#                    'height': 24,
#                    'count': 1,
#                    'crs': CRS.from_epsg(32613),
#                    'transform': Affine(20.0, 0.0, 695800.0, 0.0, -20.0, 3983280.0),
#                    'tiled': False,
#                    'interleave': 'band'}

#     # And then change the band count to 1, set the
#     # dtype to uint8, and specify LZW compression.
#     profile.update(
#         dtype=rasterio.uint8,
#         count=1,
#         compress='lzw')

#     with rasterio.open('example.tif', 'w', **profile) as dst:
#         dst.write(array.astype(rasterio.uint8), 1)
