import inspect
import logging
import os
import warnings
from ast import literal_eval
from copy import deepcopy
from datetime import datetime

import db.utilities as db_utils
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

# from geoml import feature_groups
from geoml import Tables


class FeatureData(Tables):
    """
    Class that provides file management functionality specifically for research
    data that are used for the basic training of supervised regression models.
    This class assists in the loading, filtering, and preparation of research
    data for its use in a supervised regression model.
    """

    __allowed_params = (
        "fname_obs_tissue",
        "fname_cropscan",
        "fname_sentinel",
        "random_seed",
        "dir_results",
        "group_feats",
        "response_data",
        # "ground_truth_tissue",
        # "ground_truth_measure",
        "date_train",
        "date_tolerance",
        "cv_method",
        "cv_method_kwargs",
        "cv_split_kwargs",
        "impute_method",
        "cv_method_tune",
        "cv_method_tune_kwargs",
        "cv_split_tune_kwargs",
        # 'kfold_stratify', 'n_splits', 'n_repeats',
        "train_test",
        "print_out_fd",
        "print_splitter_info",
    )

    def __init__(self, **kwargs):
        super(FeatureData, self).__init__(**kwargs)
        self.load_tables()

        # FeatureData defaults
        self.random_seed = None
        # self.fname_obs_tissue = 'obs_tissue.csv'
        # self.fname_cropscan = 'rs_cropscan.csv'
        # self.fname_sentinel = 'rs_sentinel.csv'
        # self.fname_wx = 'calc_weather.csv'
        self.dir_results = None
        self.group_feats = {
            "dae": "dae",
            "rate_ntd": {"col_rate_n": "rate_n_kgha", "col_out": "rate_ntd_kgha"},
            "cropscan_wl_range1": [400, 900],
        }
        self.response_data = {
            "table_name": "obs_tissue",
            "value_col": "value",
            "owner": "css-farms-dalhart",
            "tissue": "petiole",
            "measure": "no3_ppm",
        }
        # self.ground_truth = 'vine_n_pct'
        # self.ground_truth_tissue = "vine"
        # self.ground_truth_measure = "n_pct"
        self.date_train = datetime.now().date()
        self.date_tolerance = 3
        self.cv_method = train_test_split
        self.cv_method_kwargs = {
            "arrays": "df",
            "test_size": "0.4",
            "stratify": 'df[["owner", "year"]]',
        }
        self.cv_split_kwargs = None
        self.impute_method = "iterative"
        # self.kfold_stratify = ['owner', 'year']
        # self.n_splits = 2
        # self.n_repeats = 3
        self.train_test = "train"
        self.cv_method_tune = RepeatedStratifiedKFold
        self.cv_method_tune_kwargs = {"n_splits": 4, "n_repeats": 3}
        self.cv_split_tune_kwargs = None
        self.print_out_fd = False
        self.print_splitter_info = False
        # self.test_f_self(**kwargs)

        self._set_params_from_kwargs_fd(**kwargs)
        self._set_attributes_fd()
        self._load_df_response()

        if self.dir_results is not None:
            os.makedirs(self.dir_results, exist_ok=True)
        self._get_random_seed()
        # self.tables = Tables(base_dir_data=self.base_dir_data)

    def _set_params_from_dict_fd(self, config_dict):
        """
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        """
        if config_dict is not None and "FeatureData" in config_dict:
            params_fd = config_dict["FeatureData"]
        elif config_dict is not None and "FeatureData" not in config_dict:
            params_fd = deepcopy(config_dict)
        else:  # config_dict is None
            return
        for k, v in params_fd.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, deepcopy(v))

    def _set_params_from_kwargs_fd(self, **kwargs):
        """
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'config_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``FeatureData`` more explicitly.
        """
        if "config_dict" in kwargs:
            self._set_params_from_dict_fd(kwargs.get("config_dict"))
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, deepcopy(v))
                    # if k == 'ground_truth_tissue' or k == 'ground_truth_measure':
                    if k == "response_data":
                        self._load_df_response()

    def _set_attributes_fd(self):
        """
        Sets any class attribute to ``None`` that will be created in one of the
        user functions
        """
        self.df_X_y = None
        self.df_X = None
        self.df_y = None

        # "labels" vars indicate the df columns in the X matrix and y vector
        self.labels_id = None
        self.labels_x = None
        self.labels_y_id = None
        self.label_y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.stratify_train = None
        self.stratify_test = None

        # self.df_obs_tissue = None
        # self.df_tuber_biomdry_Mgha = None
        # self.df_vine_biomdry_Mgha = None
        # self.df_wholeplant_biomdry_Mgha = None
        # self.df_tuber_biomfresh_Mgha = None
        # self.df_canopy_cover_pct = None
        # self.df_tuber_n_kgha = None
        # self.df_vine_n_kgha = None
        # self.df_wholeplant_n_kgha = None
        # self.df_tuber_n_pct = None
        # self.df_vine_n_pct = None
        # self.df_wholeplant_n_pct = None
        # self.df_petiole_no3_ppm = None
        # self.df_cs = None
        # self.df_wx = None

    def _handle_wl_cols(self, c, wl_range, labels_x, prefix="wl_"):
        """
        Checks for reflectance column validity with a prefix present.

        Args:
            c (``str``): The column label to evaluate. If it is numeric and
                resembles an integer, it will be added to labels_x.
            labels_x (``list``): The list that holds the x labels to use/keep.
            prefix (``str``): The prefix to disregard when evaluating <c> for its
                resemblance of an integer.

        Note:
            If prefix is set to '' or ``None``, <c> will be appended to <labels_x>.
        """
        if not isinstance(c, str):
            c = str(c)
        col = c.replace(prefix, "") if prefix in c else c

        if col.isnumeric() and int(col) >= wl_range[0] and int(col) <= wl_range[1]:
            labels_x.append(c)
        return labels_x

    def _get_labels_x(self, group_feats, cols=None):
        """
        Parses ``group_feats`` and returns a list of column headings so that
        a df can be subset to build the X matrix with only the appropriate
        features indicated by ``group_feats``.
        """
        labels_x = []
        for key in group_feats:
            logging.info("Loading <group_feats> key: {0}".format(key))
            if "wl_range" in key:
                wl_range = group_feats[key]
                assert cols is not None, "``cols`` must be passed."
                for c in cols:
                    labels_x = self._handle_wl_cols(c, wl_range, labels_x, prefix="wl_")
            elif (
                "bands" in key
                or "weather_derived" in key
                or "weather_derived_res" in key
            ):
                labels_x.extend(group_feats[key])
            elif "applications" in key:
                apps_kwargs = group_feats[key]["rate_kwargs"]
                select_extra = (
                    apps_kwargs["select_extra"]
                    if "select_extra" in apps_kwargs.keys()
                    else []
                )
                feats_apps = [
                    f.split(" as ")[-1] for f in group_feats[key]["features"]
                ] + select_extra
                labels_x.extend(feats_apps)
            elif "planting" in key:
                plant_kwargs = group_feats[key]["date_origin_kwargs"]
                select_extra = (
                    plant_kwargs["select_extra"]
                    if "select_extra" in plant_kwargs.keys()
                    else []
                )
                feats_plant = [
                    f.split(" as ")[-1] for f in group_feats[key]["features"]
                ] + select_extra
                labels_x.extend(feats_plant)
            elif "weather" in key:
                weather_kwargs = group_feats[key]["date_origin_kwargs"]
                select_extra = (
                    weather_kwargs["select_extra"]
                    if "select_extra" in weather_kwargs.keys()
                    else []
                )
                feats_weather = [
                    literal_eval(f.split(" as ")[-1])
                    for f in group_feats[key]["features"]
                ] + select_extra
                labels_x.extend(feats_weather)
            else:
                labels_x.append(group_feats[key])
        self.labels_x = labels_x
        return self.labels_x

    def _check_empty_geom(self, df):
        if isinstance(df, gpd.GeoDataFrame):
            if all(df[~df[df.geometry.name].is_empty]):
                df = pd.DataFrame(df.drop(columns=[df.geometry.name]))
        return df

    # group_feats = {
    #     "dae": "dae",
    #     "rate_ntd": {"col_rate_n": "rate_n_kgha", "col_out": "rate_ntd_kgha"},
    #     "cropscan_wl_range1": [400, 900],
    #     "weather": {
    #         "date_origin_kwargs": {"table": "as_planted", "column": "date_plant"},
    #         "features": [
    #             'sum(w."precip_24h:mm") as "precip_csp:mm"',
    #             'sum(w."evapotranspiration_24h:mm") as "evapotranspiration_csp:mm"',
    #             'sum(w."global_rad_24h:MJ") as "global_rad_csp:MJ"',
    #             'sum(w."gdd_10C30C_2m_24h:C") as "gdd_10C30C_2m_csp:C"',
    #         ]
    #     }
    # }

    def _clean_zonal_stats(self, gdf_stats, wl_min, wl_max):
        # Assumes all rasters in db.reflectance have same bands and band order
        rast_metadata = pd.read_sql(
            "select rast_metadata from reflectance order by rid limit 1",
            con=self.db.engine,
        )["rast_metadata"].item()
        bands = ["b{0}".format(i + 1) for i in range(rast_metadata["count"])]
        cols = ["wl_{0:.0f}".format(round(wl)) for wl in rast_metadata["wavelength"]]
        gdf_stats.rename(columns=dict(zip(bands, cols)), inplace=True)
        gdf_stats.rename(columns={"geom": "geom_rast"}, inplace=True)
        wl_drop, wl_keep = [], []
        for c in gdf_stats.columns:
            try:
                if int(c.split("_")[-1]) < wl_min or int(c.split("_")[-1]) > wl_max:
                    wl_drop.append(c)
                else:
                    wl_keep.append(c)
            except:
                pass
        gdf_stats.drop(columns=wl_drop, inplace=True)
        (
            table_name_response,
            value_col,
            filter_kwargs,
        ) = db_utils.pop_table_val_from_response_data(self.response_data)
        return gdf_stats, wl_keep

    def _join_group_feats(self, df, group_feats, date_tolerance):
        """
        Joins predictors to ``df`` based on the contents of group_feats
        """
        subset = db_utils.get_primary_keys(self.df_response)

        for key in group_feats:
            logging.info(
                "Adding features to feature matrix:\n{0}\n" "".format(group_feats[key])
            )
            if "applications" in key:
                rate_kwargs = group_feats[key]["rate_kwargs"]
                select_extra = (
                    rate_kwargs["select_extra"]
                    if "select_extra" in rate_kwargs.keys()
                    else []
                )
                feats_apps = [
                    f.split(" as ")[-1] for f in group_feats[key]["features"]
                ] + select_extra

                gdf_app = self.db.get_application_summary_gis(
                    response_data=self.response_data,
                    rate_kwargs=rate_kwargs,
                    feature_list=group_feats[key]["features"],
                    filter_last_x_days=group_feats[key]["filter_last_x_days"],
                    predict=False,
                )
                df = pd.merge(
                    df,
                    gdf_app[["id"] + subset + feats_apps],
                    how="left",
                    on=subset + ["id"],
                )
            if "planting" in key:
                plant_kwargs = group_feats[key]["date_origin_kwargs"]
                select_extra = (
                    plant_kwargs["select_extra"]
                    if "select_extra" in plant_kwargs.keys()
                    else []
                )
                feats_plant = [
                    f.split(" as ")[-1] for f in group_feats[key]["features"]
                ] + select_extra

                gdf_plant = self.db.get_planting_summary_gis(
                    response_data=self.response_data,
                    date_origin_kwargs=plant_kwargs,
                    feature_list=group_feats[key]["features"],
                    predict=False,
                )
                df = pd.merge(
                    df,
                    gdf_plant[["id"] + subset + feats_plant],
                    how="left",
                    on=subset + ["id"],
                )
            if "sentinel" in key:
                gdf_stats = self.db.get_zonal_stats(
                    response_data=self.response_data,
                    tolerance=date_tolerance,  # days
                    direction="nearest",  # ["nearest", "past", "future"]
                    buffer=-20,
                    stat="mean",
                    units_expression="/10000",
                    wide=True,
                )
                wl_min, wl_max = group_feats[key]
                gdf_stats, wl_keep = self._clean_zonal_stats(gdf_stats, wl_min, wl_max)
                df = pd.merge(
                    df,
                    gdf_stats[["id"] + subset + wl_keep],
                    how="left",
                    on=subset + ["id"],
                )
            if "weather" in key:
                weather_kwargs = group_feats[key]["date_origin_kwargs"]
                select_extra = (
                    weather_kwargs["select_extra"]
                    if "select_extra" in weather_kwargs.keys()
                    else []
                )
                feature_list_sql = db_utils.swap_single_for_double_quotes(
                    group_feats[key]["features"]
                )
                feats_weather = [
                    literal_eval(f.split(" as ")[-1]) for f in feature_list_sql
                ] + select_extra

                gdf_weather = self.db.get_weather_summary_gis(
                    response_data=self.response_data,
                    date_origin_kwargs=group_feats[key]["date_origin_kwargs"],
                    feature_list=feature_list_sql,
                )
                df = pd.merge(
                    df,
                    gdf_weather[["id"] + subset + feats_weather],
                    how="left",
                    on=subset + ["id"],
                )
        if "rate_ntd" in group_feats:
            col_rate_n = group_feats["rate_ntd"]["col_rate_n"]
            col_rate_ntd_out = group_feats["rate_ntd"]["col_out"]
            # unit_str = value.rsplit('_', 1)[1]
            df = self.rate_ntd(
                df, col_rate_n=col_rate_n, col_rate_ntd_out=col_rate_ntd_out
            )
        if "dae" in group_feats:
            df = self.dae(df)  # add DAE
        if "dap" in group_feats:
            df = self.dap(df)  # add DAP
        if "weather_derived" in group_feats:
            self.weather_derived = self._check_empty_geom(self.weather_derived)
            df = self.join_closest_date(  # join weather by closest date
                df,
                self.weather_derived,
                left_on="date",
                right_on="date",
                tolerance=0,
                delta_label=None,
            )
        if "weather_derived_res" in group_feats:
            df = self.join_closest_date(  # join weather by closest date
                df,
                self.weather_derived_res,
                left_on="date",
                right_on="date",
                tolerance=0,
                delta_label=None,
            )
        return df

    def _load_df_response(self):
        """
        Loads the response DataFrame based on ``self.response_data``.
        Checks that target variable is of dtype[float64] in the Pandas
        DataFrame.
        """
        response_data = deepcopy(self.response_data)
        table_name = response_data.pop("table_name")
        value_col = response_data.pop("value_col")
        logging.info(
            "Loading response dataframe:\n\tTable: {0}\n\tkwargs: {1}\n"
            "".format(table_name, response_data)
        )
        df_response = self.db.get_table_df(table_name, **response_data)
        if len(df_response) < 1:
            raise ValueError(
                "No response data is present. Please add data to database, or adjust "
                "<response_data> config parameters.\ntable_name: {0}"
                "".format(self.response_data)
            )  # check to ensure there is data available

        target_type = type(df_response[value_col].dtypes).__name__
        msg2 = (
            f"The target variable is of {target_type}.",
            "GeoML can currently only handle regression problems and",
            "requires that the target variable be of dtype[float64].",
            "Please reframe your ML problem.",
        )
        assert target_type == "dtype[float64]", " ".join(
            msg2
        )  # check target variable type

        self.labels_y_id = list(response_data.keys())
        self.label_y = value_col

        df_response = df_response[
            pd.notnull(df_response[value_col])
        ]  # remove NULL values
        df_response = self._add_empty_geom(df_response)
        self.df_response = df_response

    def _write_to_readme(self, msg, msi_run_id=None, row=None):
        """
        Writes ``msg`` to the README.txt file
        """
        # Note if I get here to modify foler_name or use msi_run_id:
        # Try to keep msi-run_id out of this class; instead, make all folder
        # names, etc. be reflected in the self.dir_results variable (?)
        # if msi_run_id is not None and row is not None:
        #     folder_name = 'msi_' + str(msi_run_id) + '_' + str(row.name).zfill(3)
        #     dir_out = os.path.join(self.dir_results, folder_name)
        # with open(os.path.join(self.dir_results, folder_name + '_README.txt'), 'a') as f:
        if self.dir_results is None:
            logging.info("<dir_results> must be set to create README file.")
            return
        else:
            with open(os.path.join(self.dir_results, "README.txt"), "a") as f:
                f.write(str(msg) + "\n")

    def _get_random_seed(self):
        """
        Assign the random seed
        """
        if self.random_seed is None:
            self.random_seed = np.random.randint(0, 1e6)
        else:
            self.random_seed = int(self.random_seed)
        self._write_to_readme("Random seed: {0}".format(self.random_seed))

    def _add_empty_geom(self, gdf):
        """Adds field_bounds geometry to all Empty gdf geometries"""
        subset = db_utils.get_primary_keys(gdf)
        field_bounds = self.db.get_table_df("field_bounds")

        # Split gdf into geom/no-geom
        gdf_geom = gdf[~gdf[gdf.geometry.name].is_empty]
        gdf_nogeom = gdf[gdf[gdf.geometry.name].is_empty]

        # Get field bounds where owner, farm, field_id, and year match obs_tissue_no_geom

        gdf_nogeom.drop(columns=[gdf.geometry.name], inplace=True)
        field_bounds.drop(columns=["id"], inplace=True)
        gdf_out = pd.concat(
            [gdf_geom, gdf_nogeom.merge(field_bounds, on=subset)], axis=0
        )
        return gdf_out

    # def _get_response_df(
    #     self, tissue, measure, tissue_col="tissue", measure_col="measure"
    # ):
    #     # ground_truth='vine_n_pct'):
    #     """
    #     Gets the relevant response dataframe

    #     Args:
    #         ground_truth_tissue (``str``): The tissue to use for the response
    #             variable. Must be in "obs_tissue.csv", and dictates which table
    #             to access to retrieve the relevant training data.
    #         ground_truth_measure (``str``): The measure to use for the response
    #             variable. Must be in "obs_tissue.csv"
    #         tissue_col (``str``): The column name from "obs_tissue.csv" to look
    #             for ``tissue``.
    #         measure_col (``str``): The column name from "obs_tissue.csv" to
    #             look for ``measure``.
    #     """
    #     tissue_list = (
    #         self.df_response.groupby(by=[measure_col, tissue_col], as_index=False)
    #         .first()[tissue_col]
    #         .tolist()
    #     )
    #     measure_list = (
    #         self.df_response.groupby(by=[measure_col, tissue_col], as_index=False)
    #         .first()[measure_col]
    #         .tolist()
    #     )
    #     avail_list = ["_".join(map(str, i)) for i in zip(tissue_list, measure_list)]
    #     # avail_list = ["vine_n_pct", "pet_no3_ppm", "tuber_n_pct",
    #     #               "biomass_kgha"]
    #     msg = (
    #         "``tissue``  and ``measure`` must be "
    #         'one of:\n{0}.\nPlease see "obs_tissue" table to be sure your '
    #         "intended data are available."
    #         "".format(list(zip(tissue_list, measure_list)))
    #     )
    #     assert "_".join((tissue, measure)) in avail_list, msg

    #     df = self.df_response[
    #         (self.df_response[measure_col] == measure)
    #         & (self.df_response[tissue_col] == tissue)
    #     ]
    #     df = self._add_empty_geom(df)
    #     return df

    def _stratify_set(
        self, stratify_cols=["owner", "farm", "year"], train_test=None, df=None
    ):
        """
        Creates a 1-D array of the stratification IDs (to be used by k-fold)
        for both the train and test sets: <stratify_train> and <stratify_test>

        Returns:
            groups (``numpy.ndarray): Array that asssigns each observation to
                a stratification group.
        """
        if df is None:
            df = self.df_y.copy()
        msg1 = "All <stratify> strings must be columns in <df_y>"
        for c in stratify_cols:
            assert c in df.columns, msg1
        if train_test is None:
            groups = df.groupby(stratify_cols).ngroup().values
        else:
            groups = (
                df[df["train_test"] == train_test]
                .groupby(stratify_cols)
                .ngroup()
                .values
            )

        unique, counts = np.unique(groups, return_counts=True)
        logging.info("\nStratification groups: {0}".format(stratify_cols))
        logging.info("Number of stratification groups:  {0}".format(len(unique)))
        logging.info("Minimum number of splits allowed: {0}".format(min(counts)))
        return groups

    def _check_sklearn_splitter(
        self, cv_method, cv_method_kwargs, cv_split_kwargs=None, raise_error=False
    ):
        """
        Checks <cv_method>, <cv_method_kwargs>, and <cv_split_kwargs> for
        continuity.

        Displays a UserWarning or raises ValueError if an invalid parameter
        keyword is provided.

        Args:
            raise_error (``bool``): If ``True``, raises a ``ValueError`` if
                parameters do not appear to be available. Otherwise, simply
                issues a warning, and will try to move forward anyways. This
                exists because <inspect.getfullargspec(self.cv_method)[0]> is
                used to get the arguments, but certain scikit-learn functions/
                methods do not expose their arguments to be screened by
                <inspect.getfullargspec>. Thus, the only way to use certain
                splitter functions is to bypass this check.

        Note:
            Does not check for the validity of the keyword argument(s). Also,
            the warnings does not work as fully intended because when
            <inspect.getfullargspec(self.cv_method)[0]> returns an empty list,
            there is no either a warning or ValueError can be raised.
        """
        if cv_split_kwargs is None:
            cv_split_kwargs = {}

        # import inspect
        # from sklearn.model_selection import RepeatedStratifiedKFold
        # cv_method = RepeatedStratifiedKFold
        # cv_method_kwargs = {'n_splits': 4, 'n_repeats': 3}
        # cv_split_kwargs = None

        cv_method_args = inspect.getfullargspec(cv_method)[0]
        cv_split_args = inspect.getfullargspec(cv_method.split)[0]
        if "self" in cv_method_args:
            cv_method_args.remove("self")
        if "self" in cv_split_args:
            cv_split_args.remove("self")
        return cv_split_kwargs

        msg1 = (
            "Some <cv_method_kwargs> parameters do not appear to be "
            "available with the <{0}> function.\nAllowed parameters: {1}\n"
            "Passed to <cv_method_kwargs>: {2}\n\nPlease adjust "
            "<cv_method> and <cv_method_kwargs> so they follow the "
            'requirements of one of the many scikit-learn "splitter '
            'classes". Documentation available at '
            "https://scikit-learn.org/stable/modules/classes.html#splitter-classes."
            "".format(cv_method.__name__, cv_method_args, list(cv_method_kwargs.keys()))
        )
        msg2 = (
            "Some <cv_split_kwargs> parameters are not available with "
            "the <{0}.split()> method.\nAllowed parameters: {1}\nPassed "
            "to <cv_split_kwargs>: {2}\n\nPlease adjust <cv_method>, "
            "<cv_method_kwargs>, and/or <cv_split_kwargs> so they follow "
            'the requirements of one of the many scikit-learn "splitter '
            'classes". Documentation available at '
            "https://scikit-learn.org/stable/modules/classes.html#splitter-classes."
            "".format(cv_method.__name__, cv_split_args, list(cv_split_kwargs.keys()))
        )
        if (
            any(
                [
                    i not in inspect.getfullargspec(cv_method)[0]
                    for i in cv_method_kwargs
                ]
            )
            == True
        ):
            if raise_error:
                raise ValueError(msg1)
            else:
                warnings.warn(msg1, UserWarning)

        if (
            any(
                [
                    i not in inspect.getfullargspec(cv_method.split)[0]
                    for i in cv_split_kwargs
                ]
            )
            == True
        ):
            if raise_error:
                raise ValueError(msg2)
            else:
                warnings.warn(msg2, UserWarning)

    def _cv_method_check_random_seed(self, cv_method, cv_method_kwargs):
        """
        If 'random_state' is a valid parameter in <cv_method>, sets from
        <random_seed>.
        """
        # cv_method_args = inspect.getfullargspec(cv_method)[0]
        # TODO: Add tests for all supported <cv_method>s
        cv_method_args = inspect.signature(cv_method).parameters
        if "random_state" in cv_method_args:  # ensure random_seed is set correctly
            cv_method_kwargs[
                "random_state"
            ] = (
                self.random_seed
            )  # if this will get passed to eval(), should be fine since it gets passed to str() first
        return cv_method_kwargs

    def _splitter_eval(self, cv_split_kwargs, df=None):
        """
        Preps the CV split keyword arguments (evaluates them to variables).
        """
        if cv_split_kwargs is None:
            cv_split_kwargs = {}
        if "X" not in cv_split_kwargs and df is not None:  # sets X to <df>
            cv_split_kwargs["X"] = "df"
        scope = locals()

        if df is None and "df" in [i for i in [a for a in cv_split_kwargs.values()]]:
            raise ValueError(
                "<df> is None, but is present in <cv_split_kwargs>. Please "
                "pass <df> or ajust <cv_split_kwargs>"
            )
        # evaluate any str; keep anything else as is
        cv_split_kwargs_eval = dict(
            (k, eval(str(cv_split_kwargs[k]), scope))
            if isinstance(cv_split_kwargs[k], str)
            else (k, cv_split_kwargs[k])
            for k in cv_split_kwargs
        )
        return cv_split_kwargs_eval

    def _train_test_split_df(self, df):
        """
        Splits <df> into train and test sets.

        Any of the many scikit-learn "splitter classes" should be supported
        (documentation available at: https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        All parameters used by the ``cv_method`` function or the ``cv_method.split``
        method should be set via ``cv_method_kwargs`` and ``cv_split_kwargs``.

        Note:
            If <n_splits> is set for any <SplitterClass>, it is generally
            ignored. That is, if there are multiple splitting iterations
            (<n_splits> greater than 1), only the first iteration is used to
            split between train and test sets.

        Args:
            df (``DataFrame``): The df to split between train and test sets.
            cv_method (``sklearn.model_selection.SplitterClass``): The scikit-learn
            method to use to split into training and test groups. In addition to
            ``SplitterClass``(es), ``cv_method`` can be
            ``sklearn.model_selection.train_test_split``, in which case
            ``cv_split_kwargs`` is ignored and ``cv_method_kwargs`` should be used to
            pass ``cv_method`` parameters to be evaluated via the eval() function.
            cv_method_kwargs (``dict``): Kwargs to be passed to ``cv_method()``.
            cv_split_kwargs (``dict``): Kwargs to be passed to ``cv_method.split()``.
            Note that the ``X`` kwarg defaults to ``df`` if not set.
        """
        cv_method = deepcopy(self.cv_method)
        cv_method_kwargs = deepcopy(self.cv_method_kwargs)
        cv_split_kwargs = deepcopy(self.cv_split_kwargs)
        cv_method_kwargs = self._cv_method_check_random_seed(
            cv_method, cv_method_kwargs
        )
        if cv_method.__name__ == "train_test_split":
            # Because train_test_split has **kwargs for options, random_state is not caught, so it should be set explicitly
            cv_method_kwargs["random_state"] = self.random_seed
            if "arrays" in cv_method_kwargs:  # I think can only be <df>?
                df = eval(cv_method_kwargs.pop("arrays", None))
            scope = locals()  # So it understands what <df> is inside func scope
            cv_method_kwargs_eval = dict(
                (k, eval(str(cv_method_kwargs[k]), scope)) for k in cv_method_kwargs
            )
            # return
            df_train, df_test = cv_method(df, **cv_method_kwargs_eval)
        else:
            cv_split_kwargs = self._check_sklearn_splitter(
                cv_method, cv_method_kwargs, cv_split_kwargs, raise_error=False
            )
            cv = cv_method(**cv_method_kwargs)
            for key in ["y", "groups"]:
                if key in cv_split_kwargs:
                    if isinstance(cv_split_kwargs[key], list):
                        # assume these are columns to group by and adjust kwargs
                        cv_split_kwargs[key] = self._stratify_set(
                            stratify_cols=cv_split_kwargs[key], train_test=None, df=df
                        )

            # Now cv_split_kwargs should be ready to be evaluated
            cv_split_kwargs_eval = self._splitter_eval(cv_split_kwargs, df=df)

            if "X" not in cv_split_kwargs_eval:  # sets X
                cv_split_kwargs_eval["X"] = df

            train_idx, test_idx = next(cv.split(**cv_split_kwargs_eval))
            df_train, df_test = df.loc[train_idx], df.loc[test_idx]

        train_pct = (len(df_train) / (len(df_train) + len(df_test))) * 100
        test_pct = (len(df_test) / (len(df_train) + len(df_test))) * 100
        logging.info(
            '\nNumber of observations in the "training" set: {0} ({1:.1f}%)'.format(
                len(df_train), train_pct
            )
        )
        logging.info(
            'Number of observations in the "test" set: {0} ({1:.1f}%)\n'.format(
                len(df_test), test_pct
            )
        )

        df_train.insert(0, "train_test", "train")
        df_test.insert(0, "train_test", "test")
        df_out = df_train.copy()
        df_out = pd.concat([df_out, df_test], axis=0).reset_index(drop=True)
        return df_out

    def _impute_missing_data(self, X, method="iterative"):
        """
        Imputes missing data in X - sk-learn models will not work with null data

        Args:
            method (``str``): should be one of "iterative" (takes more time)
                or "knn" (default: "iterative").
        """
        if pd.isnull(X).any() is False:
            return X

        if method == "iterative":
            imp = IterativeImputer(max_iter=10, random_state=self.random_seed)
        elif method == "knn":
            imp = KNNImputer(n_neighbors=2, weights="uniform")
        elif method == None:
            return X
        X_out = imp.fit_transform(X)
        return X_out
        # if X.shape == X_out.shape:
        #     return X_out  # does not impute if all nan columns (helps debug)
        # else:
        #     return X

    def _get_X_and_y(self, df):
        """
        Gets the X and y from df for both the train and test datasets;
        y is determined by the ``y_label`` column.
        This function depends on having the following variables already
        set:
            1. self.label_y
            2. self.group_feats

        Args:
            df (pandas.DataFrame): full data frame which contains features and response variable and
            has already been marked for splitting with the `train_test` column

        """

        df_train = df[df["train_test"] == "train"]
        df_test = df[df["train_test"] == "test"]

        X_train = df_train[self.labels_x].values
        X_test = df_test[self.labels_x].values
        y_train = df_train[self.label_y].values
        y_test = df_test[self.label_y].values

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        subset = db_utils.get_primary_keys(df)
        labels_id = subset + ["date", "train_test"]
        self.df_X = df[labels_id + self.labels_x]
        self.df_y = df[labels_id + self.labels_y_id + [self.label_y]]
        self.labels_id = labels_id

    def _save_df_X_y(self):
        """
        Saves both ``FeatureData.df_X`` and ``FeatureData.df_y`` to
        ``FeatureData.dir_results``.
        """
        dir_out = os.path.join(self.dir_results, self.label_y)
        os.makedirs(dir_out, exist_ok=True)

        fname_out_X = os.path.join(dir_out, "data_X_" + self.label_y + ".csv")
        fname_out_y = os.path.join(dir_out, "data_y_" + self.label_y + ".csv")
        self.df_X.to_csv(fname_out_X, index=False)
        self.df_y.to_csv(fname_out_y, index=False)

    def _splitter_print(self, splitter, train_test="train"):
        """
        Checks the proportions of the stratifications in the dataset and prints
        the number of observations in each stratified group. The keys are based
        on the stratified IDs in <stratify_train> or <stratify_test>
        """
        df_X = self.df_X[self.df_X["self_test"] == train_test]
        if train_test == "train":
            stratify_vector = self.stratify_train
        elif train_test == "test":
            stratify_vector = self.stratify_test
        logging.info(
            "The number of observations in each cross-validation dataset "
            "are listed below.\nThe key represents the <stratify_{0}> ID, "
            "and the value represents the number of observations used from "
            "that stratify ID".format(train_test)
        )
        logging.info("Total number of observations: {0}".format(len(stratify_vector)))
        train_list = []
        val_list = []
        for train_index, val_index in splitter:
            X_meta_train_fold = stratify_vector[train_index]
            X_meta_val_fold = stratify_vector[val_index]
            X_train_dataset_id = X_meta_train_fold[:]
            train = {}
            val = {}
            for uid in np.unique(X_train_dataset_id):
                n1 = len(np.where(X_meta_train_fold[:] == uid)[0])
                n2 = len(np.where(X_meta_val_fold[:] == uid)[0])
                train[uid] = n1
                val[uid] = n2
            train_list.append(train)
            val_list.append(val)
        logging.info("\nK-fold train set:")
        logging.info("Number of observations: {0}".format(len(train_index)))
        logging.info(*train_list, sep="\n")
        logging.info("\nK-fold validation set:")
        logging.info("Number of observations: {0}".format(len(val_index)))
        logging.info(*val_list, sep="\n")

    def _manage_missing_feat_data(self, df):
        """
        Manages missing data from the training feature matrix.

        Note:
            - If there isn't missing data, ``df`` is returned unchanged.
            - If ``self.impute_method`` is ``None``, then all rows with NA values are dropped.
            - If ``self.impute_method`` is not ``None``, the missing data values are
              imputed following ``self.impute_method``.
            - Prior to imputation, all fully NA columns are removed.

        Args:
            df (``GeoDataFrame``): full feature training matrix (X and y)
        """
        msg = '``impute_method`` must be one of: ["iterative", "knn", None]'
        assert self.impute_method in ["iterative", "knn", None], msg

        if self.impute_method is None and df.isna().any(axis=None):
            df = df.dropna()
            labels_x = self._get_labels_x(self.group_feats, cols=df.columns)
        else:
            # check to make sure there are no columns with all NA values prior to imputing
            cols_nan = df.columns[df.isna().all(0)]
            if len(cols_nan) > 0:
                df.drop(list(cols_nan), axis="columns", inplace=True)

            # which columns remain after NA drops?
            labels_x = self._get_labels_x(self.group_feats, cols=df.columns)

            # perform imputation
            X_df = df[labels_x].values
            X_df = self._impute_missing_data(X_df, method=self.impute_method)
            df.iloc[:, [df.columns.get_loc(c) for c in labels_x if c in df]] = X_df
        return df

    def get_feat_group_X_y(self, **kwargs):
        """
        Constructs the training feature matrix and response vector based on config settings.

        Note:
            - Step 1: Checks for and drops NULL observations.
            - Step 2: Checks for and drops observations on or after ``date_train``.
            - Step 3: Creates training features for all observations based on ``group_feats``.
            - Step 4: Handles missing feature data based on ``impute_method``.
            - Step 5: Splits data into training and test sets based on ``cv_method``.

        Example:
            >>> from geoml import FeatureData
            >>> from geoml.tests import config

            >>> fd = FeatureData(config_dict=config.config_dict)
            >>> fd.get_feat_group_X_y(test_size=0.1)
            >>> print('Shape of training matrix "X": {0}'.format(fd.X_train.shape))
            >>> print('Shape of training vector "y": {0}'.format(fd.y_train.shape))
            >>> print('Shape of testing matrix "X":  {0}'.format(fd.X_test.shape))
            >>> print('Shape of testing vector "y":  {0}'.format(fd.y_test.shape))
            Shape of training matrix "X": (579, 14)
            Shape of training vector "y": (579,)
            Shape of testing matrix "X":  (65, 14)
            Shape of testing vector "y":  (65,)
        """

        logging.info("Constructing the training feature matrix.")
        self._set_params_from_kwargs_fd(**kwargs)

        df1_nn_obs = self.df_response[
            pd.notnull(self.df_response[self.label_y])
        ]  # check to see if there are any NULL observations and remove them so we don't extract data for them

        df2_valid_date = df1_nn_obs[
            df1_nn_obs["date"] < self.date_train
        ].reset_index()  # remove all observations on or after date_train so we don't extract data for them
        msg1 = (
            "After removing null observations and limiting df to before {0}, there "
            "are no obseravtions left for training. Check that there are a sufficient "
            "number of observations for this customer and time period.".format(
                self.date_train
            )
        )
        assert (
            len(df2_valid_date) > 0
        ), msg1  # CHECKPOINT: make sure there is data left after all of the removals

        df3_X_y = self._join_group_feats(
            df2_valid_date,
            group_feats=self.group_feats,
            date_tolerance=self.date_tolerance,
        )  # extract features from `group_feats` for all observations in `df2_valid_date`
        self.df_X_y = df3_X_y.copy()

        df4_impute = self._manage_missing_feat_data(
            df3_X_y
        )  # handle missing feature data using impute method
        msg2 = (
            "After removing all observations which have NA values for the desired "
            "features, no observations remain. Re-consider features to be included or "
            "date_tolerance of {0}.".format(self.date_tolerance)
        )
        assert (
            len(df4_impute) > 0
        ), msg2  # CHECKPOINT: make sure there is data left with the specified date_tolerance

        df5_cv_split = self._train_test_split_df(
            df4_impute
        )  # now that we have the complete training feature matrix, split data
        self._get_X_and_y(
            df5_cv_split
        )  # organize X and y data across train and test sets

        if self.dir_results is not None:
            self._save_df_X_y()

    def get_tuning_splitter(self, **kwargs):
        self._set_params_from_kwargs_fd(**kwargs)

        cv_method = deepcopy(self.cv_method_tune)
        cv_method_kwargs = deepcopy(self.cv_method_tune_kwargs)
        cv_split_kwargs = deepcopy(self.cv_split_tune_kwargs)
        cv_method_kwargs = self._cv_method_check_random_seed(
            cv_method, cv_method_kwargs
        )

        if cv_method.__name__ == "train_test_split":
            # Because train_test_split has **kwargs for options, random_state is not caught, so it should be set explicitly
            cv_method_kwargs["random_state"] = self.random_seed
            if "arrays" in cv_method_kwargs:  # I think can only be <df>?
                df = eval(cv_method_kwargs.pop("arrays", None))
            scope = locals()  # So it understands what <df> is inside func scope
            cv_method_kwargs_eval = dict(
                (k, eval(str(cv_method_kwargs[k]), scope)) for k in cv_method_kwargs
            )
            return cv_method(df, **cv_method_kwargs_eval)
        else:
            cv_split_kwargs = self._check_sklearn_splitter(
                cv_method, cv_method_kwargs, cv_split_kwargs, raise_error=False
            )
            self.cv_split_tune_kwargs = cv_split_kwargs
            cv = cv_method(**cv_method_kwargs)
            for key in ["y", "groups"]:
                if key in cv_split_kwargs:
                    if isinstance(cv_split_kwargs[key], list):
                        # assume these are columns to group by and adjust kwargs
                        cv_split_kwargs[key] = self._stratify_set(
                            stratify_cols=cv_split_kwargs[key], train_test="train"
                        )

            # Now cv_split_kwargs should be ready to be evaluated
            df_X_train = self.df_X[self.df_X["train_test"] == "train"]
            cv_split_kwargs_eval = self._splitter_eval(cv_split_kwargs, df=df_X_train)

            if "X" not in cv_split_kwargs_eval:  # sets X
                cv_split_kwargs_eval["X"] = df_X_train

        if self.print_splitter_info == True:
            n_train = []
            n_val = []
            for idx_train, idx_val in cv.split(**cv_split_kwargs_eval):
                n_train.append(len(idx_train))
                n_val.append(len(idx_val))
            logging.info(
                "Tuning splitter: number of cross-validation splits: {0}".format(
                    cv.get_n_splits(**cv_split_kwargs_eval)
                )
            )
            train_pct = (np.mean(n_train) / (np.mean(n_train) + np.mean(n_val))) * 100
            val_pct = (np.mean(n_val) / (np.mean(n_train) + np.mean(n_val))) * 100
            logging.info(
                "Number of observations in the (tuning) train set (avg): {0:.1f} ({1:.1f}%)".format(
                    np.mean(n_train), train_pct
                )
            )
            logging.info(
                "Number of observations in the (tuning) validation set (avg): {0:.1f} ({1:.1f}%)\n".format(
                    np.mean(n_val), val_pct
                )
            )

        return cv.split(**cv_split_kwargs_eval)
