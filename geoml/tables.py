import os
import pandas as pd
import geopandas as gpd
from sqlalchemy import inspect
import warnings

from db import DBHandler
import db.utilities as db_utils


class Tables(object):
    """
    Class for accessing tables that contain training data. In addition to
    accessing the data (via either local files or connecting to a database),
    this class makes the appropriate joins and has functions available to add
    new columns to the table(s) that may be desireable regression features.
    """

    __allowed_params = (
        "db_name",
        "db_host",
        "db_user",
        "password",
        "db_schema",
        "db_port",
        "db",
        "base_dir_data",
        "table_names",
    )

    def __init__(self, **kwargs):
        """
        Parameters:
            base_dir_data (``str``): The base directory containing all the
                tables available to be joined. Ignored if there is a valid
                connection to a database.
            table_names (``dict`` of ``str``): The filenames of the various
                tables; these files must be in ``base_dir_data``. Ignored if
                there is a valid connection to a database.

        Note:
            To set the DB password in via keyring, adapt the following:
            >>> db_name = 'test_db_pw'
            >>> db_host = 'localhost'
            >>> db_user = 'postgres'
            >>> password = 'my_db_pw2!'
            >>> service_name = '@'.join((db_name, db_host))  # 'test_db_pw@localhost'
            >>> keyring.set_password(service_name, db_user, password)
        """
        self.db_name = None
        self.db_host = None
        self.db_user = None
        self.password = (
            None  # password does not have to be passsed if stored in local keyring
        )
        self.db_schema = None
        self.db_port = None
        self.db = None
        self.base_dir_data = None
        self.table_names = {
            "experiments": "experiments.geojson",
            "dates_res": "dates_res.csv",
            "trt": "trt.csv",
            "trt_n": "trt_n.csv",
            "trt_n_crf": "trt_n_crf.csv",
            "obs_tissue_res": "obs_tissue_res.geojson",
            "obs_soil_res": "obs_soil_res.geojson",
            "rs_cropscan_res": "rs_cropscan.csv",
            "rs_micasense_res": "rs_micasense.csv",
            "rs_spad_res": "rs_spad.csv",
            "field_bounds": "field_bounds.geojson",
            "dates": "dates.csv",
            "as_planted": "as_planted.geojson",
            "n_applications": "n_applications.geojson",
            "obs_tissue": "obs_tissue.geojson",
            "obs_soil": "obs_soil.geojson",
            "rs_sentinel": "rs_sentinel.geojson",
            "reflectance": "reflectance.geojson",
            "weather": "weather.csv",
            "weather_derived": "calc_weather.csv",
            "weather_derived_res": "calc_weather_res.csv",
        }

        self._set_params_from_kwargs_t(**kwargs)
        self._set_attributes_t(**kwargs)

        # The following are "temporary" tables, in that they need to find a
        # home in the DB (stored), or we have to come up with a solution to
        # derive them on demand (derived).
        # self.weather_derived = pd.read_csv(os.path.join(
        #     self.base_dir_data, self.table_names['weather_derived']))

    def _set_params_from_dict_t(self, config_dict):
        """
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        """
        if config_dict is not None and "Tables" in config_dict:
            params_jt = config_dict["Tables"]
        elif config_dict is not None and "Tables" not in config_dict:
            params_jt = config_dict
        else:  # config_dict is None
            return

        for k, v in params_jt.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, v)

        if params_jt["db"] is not None:  # db takes precedence over all db settings
            setattr(self, "db_name", params_jt["db"].db_name)
            setattr(self, "db_host", params_jt["db"].db_host)
            setattr(self, "db_user", params_jt["db"].db_user)
            setattr(self, "db_schema", params_jt["db"].db_schema)
            setattr(self, "db_port", params_jt["db"].db_port)
        # else:
        #     self._connect_to_db()

    def _set_params_from_kwargs_t(self, **kwargs):
        """
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'config_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``FeatureData`` more explicitly.
        """
        if "config_dict" in kwargs:
            self._set_params_from_dict_t(kwargs.get("config_dict"))
            # self._connect_to_db()

        # db_creds_old = [self.db_name, self.db_host, self.db_user,
        #                 self.db_schema, self.db_port]
        if len(kwargs) > 0:  # this evaluates to False if empty dict ({})
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)
            if "db" in kwargs:
                setattr(self, "db_name", kwargs["db"].db_name)
                setattr(self, "db_host", kwargs["db"].db_host)
                setattr(self, "db_user", kwargs["db"].db_user)
                setattr(self, "db_schema", kwargs["db"].db_schema)
                setattr(self, "db_port", kwargs["db"].db_port)
            # else:
            #     db_creds_new = [self.db_name, self.db_host, self.db_user,
            #                     self.db_schema, self.db_port]
            #     if db_creds_new != db_creds_old:  # Only connects/reconnects if something changed
            #         self._connect_to_db()
        if self.db is None:
            self._connect_to_db()

    def _connect_to_db(self):
        """
        Using DBHandler, tries to make a connection to ``tables.db_name``.
        """
        msg = (
            "To connect to the DB, all of the following variables must "
            "be passed to ``Tables`` (either directly or via the config "
            "file): [db_name, db_host, db_user, db_schema, db_port]"
        )
        if (
            any(
                v is None
                for v in [
                    self.db_name,
                    self.db_host,
                    self.db_user,
                    self.db_schema,
                    self.db_port,
                ]
            )
            == True
        ):
            print(msg)
            return

        self.db = DBHandler(
            database=self.db_name,
            host=self.db_host,
            user=self.db_user,
            password=self.password,
            port=self.db_port,
            schema=self.db_schema,
        )

        if not isinstance(self.db, DBHandler):
            print(
                "Failed to connect to database via DBHanlder. Please check "
                "DB credentials."
            )

    def _set_attributes_t(self, **kwargs):
        """
        Sets any class attribute to ``None`` that will be created in one of the
        user functions
        """
        self.experiments = None
        self.dates_res = None
        self.trt = None
        self.trt_n = None
        self.trt_n_crf = None
        self.obs_tissue_res = None
        self.obs_soil_res = None
        self.rs_cropscan_res = None
        self.rs_micasense_res = None
        self.rs_spad_res = None
        self.field_bounds = None
        self.dates = None
        self.as_planted = None
        self.n_applications = None
        self.obs_tissue = None
        self.obs_soil = None
        self.rs_sentinel = None
        self.weather = None

        self.weather_derived = (
            None  # Not sure if this will be a derived or stored table
        )
        self.weather_derived_res = None

    def _cr_rate_ntd(self, df):
        """
        join_tables.rate_ntd() must sum all the N rates before a particular
        date within each study/year/plot_id combination. Therefore, there can
        NOT be duplicate rows of the metadata when using
        join_tables.rate_ntd(). This function ensures that there is not
        duplicate metadata information in df.
        """
        msg = (
            "There can NOT be duplicate rows of the metadata in the ``df`` "
            "passed to ``join_tables.rate_ntd()``. Please filter ``df`` so "
            "there are not duplicate metadata rows.\n\nHint: is ``df`` "
            "in a long format with multiple types of data (e.g., vine N "
            "and tuber N)?\n..or does ``df`` contain subsamples?"
        )
        # cols = ['owner', 'study', 'year', 'plot_id', 'date']
        subset = db_utils.get_primary_keys(df)
        # if df.groupby(cols).size()[0].max() > 1:
        if df.groupby(subset + ["date"]).size()[0].max() > 1:
            raise AttributeError(msg)

    # def _check_requirements(self, df, table_or_feat, date_format='%Y-%m-%d'):
    #     '''
    #     Checks that ``df`` has all of the correct columns and that they contain
    #     the correct data types

    #     Parameters:
    #         df (``pandas.DataFrame``): the input DataFrame
    #         table_name (``str``): the function calling the _check_requirements()
    #             function. This is used to access join_tables.msg_require, which
    #             contains all the messages to be raised if the correct columns
    #             are not in ``df``. If ``None``, just assumes ``df`` should
    #             contain ["study", "year", and "plot_id"]
    #     '''
    #     self.cols_require = {
    #         'df_dates': ['owner', 'study', 'year', 'date_plant', 'date_emerge'],
    #         # 'df_exp': ['owner', 'study', 'year', 'plot_id', 'rep', 'trt_id'],
    #         'df_exp': ['owner', 'study', 'year', 'plot_id', 'trt_id'],
    #         'df_trt': ['owner', 'study', 'year', 'trt_id', 'trt_n', 'trt_var', 'trt_irr'],
    #         'df_n_apps': ['owner', 'study', 'year', 'trt_n', 'date_applied', 'source_n', 'rate_n_kgha'],
    #         'df_n_crf': ['owner', 'study', 'year', 'date_applied', 'source_n', 'b0', 'b1', 'b2', 'b2'],
    #         'df_wx': ['owner', 'study', 'year', 'date'],
    #         }

    # def _check_requirements_custom(
    #         self, df, date_cols=['date'], by=['owner', 'study', 'year', 'plot_id'],
    #         date_format='%Y-%m-%d'):
    #     '''
    #     Checks that ``df`` has all of the correct columns and that they contain
    #     the correct data types

    #     Parameters:
    #         df (``pandas.DataFrame``): the input DataFrame
    #         f (``str``): the function calling the _check_requirements()
    #             function. This is used to access join_tables.msg_require, which
    #             contains all the messages to be raised if the correct columns
    #             are not in ``df``. If ``None``, just assumes ``df`` should
    #             contain ["study", "year", and "plot_id"]
    #     '''
    #     if not isinstance(date_cols, list):
    #         date_cols = [date_cols]
    #     cols_require = by.copy()
    #     cols_require.extend(date_cols)

    #     msg = ('The following columns are required in ``df``: {0}. Please '
    #            'check that each of these column names are in "df.columns".'
    #            ''.format(cols_require))
    #     if not all(i in df.columns for i in cols_require):
    #         raise AttributeError(msg)

    #     n_dt = len(df.select_dtypes(include=[np.datetime64]).columns)
    #     if n_dt < len(date_cols):
    #         for d in date_cols:
    #             df[d] = pd.to_datetime(df.loc[:, d], format=date_format)
    #     return df

    # TODO: check each of the column data types if they must be particular (e.g., datetime)

    # def _read_dfs(self, date_format='%Y-%m-%d'):
    #     '''
    #     Read in all to dataframe tables and convert date columns to datetime
    #     '''
    #     df_dates = pd.read_csv(self.fnames['dates'])
    #     df_dates = self._check_requirements(df_dates, f='df_dates', date_format=date_format)
    #     self.df_dates = df_dates

    #     df_exp = pd.read_csv(self.fnames['experiments'])
    #     df_exp = self._check_requirements(df_exp, f='df_exp', date_format=date_format)
    #     self.df_exp = df_exp

    #     df_trt = pd.read_csv(self.fnames['treatments'])
    #     df_trt = self._check_requirements(df_trt, f='df_trt', date_format=date_format)
    #     self.df_trt = df_trt

    #     df_n_apps = pd.read_csv(self.fnames['n_apps'])
    #     df_n_apps = self._check_requirements(df_n_apps, f='df_n_apps', date_format=date_format)
    #     self.df_n_apps = df_n_apps

    #     df_n_crf = pd.read_csv(self.fnames['n_crf'])
    #     df_n_crf = self._check_requirements(df_n_crf, f='df_n_crf', date_format=date_format)
    #     self.df_n_crf = df_n_crf

    #     df_wx = pd.read_csv(self.fnames['wx'])
    #     df_wx = self._check_requirements(df_wx, f='df_wx', date_format=date_format)
    #     self.df_wx = df_wx

    def _load_tables(
        self, tissue_col="tissue", measure_col="measure", value_col="value"
    ):
        """
        Loads the appropriate table based on the value passed for ``tissue``,
        then filters observations according to
        """
        print("loading tables....")
        fname_obs_tissue = os.path.join(self.base_dir_data, self.fname_obs_tissue)
        df_obs_tissue = self._read_csv_geojson(fname_obs_tissue)
        self.labels_y_id = [tissue_col, measure_col]
        self.label_y = value_col
        self.df_obs_tissue = df_obs_tissue[pd.notnull(df_obs_tissue[value_col])]

        # get all unique combinations of tissue and measure cols
        tissue = (
            df_obs_tissue.groupby(by=[measure_col, tissue_col], as_index=False)
            .first()[tissue_col]
            .tolist()
        )
        measure = (
            df_obs_tissue.groupby(by=[measure_col, tissue_col], as_index=False)
            .first()[measure_col]
            .tolist()
        )
        for tissue, measure in zip(tissue, measure):
            df = self.df_obs_tissue[
                (self.df_obs_tissue[measure_col] == measure)
                & (self.df_obs_tissue[tissue_col] == tissue)
            ]
            if tissue == "tuber" and measure == "biomdry_Mgha":
                self.df_tuber_biomdry_Mgha = df.copy()
            elif tissue == "vine" and measure == "biomdry_Mgha":
                self.df_vine_biomdry_Mgha = df.copy()
            elif tissue == "wholeplant" and measure == "biomdry_Mgha":
                self.df_wholeplant_biomdry_Mgha = df.copy()
            elif tissue == "tuber" and measure == "biomfresh_Mgha":
                self.df_tuber_biomfresh_Mgha = df.copy()
            elif tissue == "canopy" and measure == "cover_pct":
                self.df_canopy_cover_pct = df.copy()
            elif tissue == "tuber" and measure == "n_kgha":
                self.df_tuber_n_kgha = df.copy()
            elif tissue == "vine" and measure == "n_kgha":
                self.df_vine_n_kgha = df.copy()
            elif tissue == "wholeplant" and measure == "n_kgha":
                self.df_wholeplant_n_kgha = df.copy()
            elif tissue == "tuber" and measure == "n_pct":
                self.df_tuber_n_pct = df.copy()
            elif tissue == "vine" and measure == "n_pct":
                self.df_vine_n_pct = df.copy()
            elif tissue == "wholeplant" and measure == "n_pct":
                self.df_wholeplant_n_pct = df.copy()
            elif tissue == "petiole" and measure == "no3_ppm":
                self.df_petiole_no3_ppm = df.copy()

        fname_cropscan = os.path.join(self.base_dir_data, self.fname_cropscan)
        if os.path.isfile(fname_cropscan):
            df_cs = self._read_csv_geojson(fname_cropscan)
            subset = db_utils.get_primary_keys(df_cs)
            self.df_cs = df_cs.groupby(subset + ["date"]).mean().reset_index()
        fname_sentinel = os.path.join(self.base_dir_data, self.fname_sentinel)
        if os.path.isfile(fname_sentinel):
            df_sentinel = self._read_csv_geojson(fname_sentinel)
            df_sentinel.rename(columns={"acquisition_time": "date"}, inplace=True)
            subset = db_utils.get_primary_keys(df_sentinel)
            self.df_sentinel = (
                df_sentinel.groupby(subset + ["date"]).mean().reset_index()
            )
        fname_wx = os.path.join(self.base_dir_data, self.fname_wx)
        if os.path.isfile(fname_wx):
            df_wx = self._read_csv_geojson(fname_wx)
            subset = db_utils.get_primary_keys(df_sentinel)
            subset = [i for i in subset if i not in ["field_id", "plot_id"]]
            self.df_wx = df_wx.groupby(subset + ["date"]).mean().reset_index()
        # TODO: Function to filter cropscan data (e.g., low irradiance, etc.)

    def _load_table_from_db(self, table_name):
        """
        Loads <table_name> from database via <Table.db>
        """
        inspector = inspect(self.db.engine)
        if table_name in inspector.get_table_names(schema=self.db.db_schema):
            df = self.db.get_table_df(table_name)
            if len(df) > 0:
                return df

    def _read_csv_geojson(self, fname):
        """
        Depending on file extension, will read from either pd or gpd
        """
        if os.path.splitext(fname)[-1] == ".csv":
            df = pd.read_csv(fname)
        elif os.path.splitext(fname)[-1] == ".geojson":
            df = gpd.read_file(fname)
        else:
            raise TypeError("<fname_sentinel> must be either a .csv or " ".geojson...")
        return df

    def _load_table_from_file(self, table_name, fname):
        """
        Loads <table_name> from file.
        """
        fname_full = os.path.join(self.base_dir_data, fname)
        if os.path.isfile(fname_full):
            df = self._read_csv_geojson(fname_full)
            if len(df) > 0:
                return df

    def _get_table_from_self(self, table_name):
        """
        Gets table as df from self based on table_name
        """
        if table_name == "experiments":
            return self.experiments
        if table_name == "dates_res":
            return self.dates_res
        if table_name == "trt":
            return self.trt
        if table_name == "trt_n":
            return self.trt_n
        if table_name == "trt_n_crf":
            return self.trt_n_crf
        if table_name == "obs_tissue_res":
            return self.obs_tissue_res
        if table_name == "obs_soil_res":
            return self.obs_soil_res
        if table_name == "rs_cropscan_res":
            return self.rs_cropscan_res
        if table_name == "rs_micasense_res":
            return self.rs_micasense_res
        if table_name == "rs_spad_res":
            return self.rs_spad_res
        if table_name == "field_bounds":
            return self.field_bounds
        if table_name == "dates":
            return self.dates
        if table_name == "as_planted":
            return self.as_planted
        if table_name == "n_applications":
            return self.n_applications
        if table_name == "obs_tissue":
            return self.obs_tissue
        if table_name == "obs_soil":
            return self.obs_soil
        if table_name == "rs_sentinel":
            return self.rs_sentinel
        if table_name == "weather":
            return self.weather
        if table_name == "weather_derived":
            return self.weather_derived
        if table_name == "weather_derived_res":
            return self.weather_derived_res

    def _set_table_to_self(self, table_name, df):
        """
        Sets df to appropriate variable based on table_name
        """
        msg = 'The following columns are required in "{0}". Missing columns: ' '"{1}".'
        # print(table_name)
        if self.db is not None:
            engine = self.db.engine
            db_schema = self.db.db_schema
        else:
            engine, db_schema = None, None
        cols_require, _ = db_utils.get_cols_nullable(
            table_name, engine=engine, db_schema=db_schema
        )
        cols_require = [item for item in cols_require if item != "id"]
        if not all(i in df.columns for i in cols_require):
            cols_missing = list(sorted(set(cols_require) - set(df.columns)))
            raise AttributeError(msg.format(table_name, cols_missing))

        # for c in cols_require:
        #     if 'date' in c or 'time' in c:
        # df.loc[:, c] = pd.to_datetime(df.loc[:, c], format=date_format)

        if table_name == "experiments":
            self.experiments = df
        if table_name == "dates_res":
            self.dates_res = df
        if table_name == "trt":
            self.trt = df
        if table_name == "trt_n":
            self.trt_n = df
        if table_name == "trt_n_crf":
            self.trt_n_crf = df
        if table_name == "obs_tissue_res":
            self.obs_tissue_res = df
        if table_name == "obs_soil_res":
            self.obs_soil_res = df
        if table_name == "rs_cropscan_res":
            self.rs_cropscan_res = df
        if table_name == "rs_micasense_res":
            self.rs_micasense_res = df
        if table_name == "rs_spad_res":
            self.rs_spad_res = df
        if table_name == "field_bounds":
            self.field_bounds = df
        if table_name == "as_planted":
            self.as_planted = df
        if table_name == "dates":
            self.dates = df
        if table_name == "n_applications":
            self.n_applications = df
        if table_name == "obs_tissue":
            self.obs_tissue = df
        if table_name == "obs_soil":
            self.obs_soil = df
        if table_name == "rs_sentinel":
            self.rs_sentinel = df
        if table_name == "weather":
            self.weather = df
        if table_name == "weather_derived":
            self.weather_derived = df
        if table_name == "weather_derived_res":
            self.weather_derived_res = df

    def _check_col_names(self, df, cols_require):
        """
        Checks to be sure all of the required columns are in <df>.

        Raises:
            AttributeError if <df> is missing any of the required columns.
        """
        if not all(i in df.columns for i in cols_require):
            cols_missing = list(sorted(set(cols_require) - set(df.columns)))
            raise AttributeError(
                "<df> is missing the following required "
                "columns: {0}.".format(cols_missing)
            )

    def _dt_or_ts_to_date(self, df, col_date, date_format="%Y-%m-%d"):
        """
        Checks if <col> is a valid datetime or timestamp column. If not, sets
        it as such as a <datetime.date> object.

        Returns:
            df
        """
        if not pd.core.dtypes.common.is_datetime64_any_dtype(df[col_date]):
            df[col_date] = pd.to_datetime(df.loc[:, col_date], format=date_format)
        df[col_date] = df[col_date].dt.date
        df[col_date] = pd.to_datetime(
            df[col_date]
        )  # convert to datetime so pd.merge is possible

        # if isinstance(df[col_date].iloc[0], pd._libs.tslibs.timestamps.Timestamp):
        #     df[col_date] = df[col_date].to_datetime64()
        return df

    def _add_date_delta(self, df_join, left_on, right_on, delta_label=None):
        """
        Adds "date_delta" column to <df_join>
        """
        idx_delta = df_join.columns.get_loc(right_on)
        if delta_label is None:
            date_delta_str = "date_delta"
        else:
            date_delta_str = "date_delta_{0}".format(delta_label)
        df_join.insert(idx_delta + 1, date_delta_str, None)
        df_join[date_delta_str] = (df_join[left_on] - df_join[right_on]).astype(
            "timedelta64[D]"
        )
        df_join = df_join[pd.notnull(df_join[date_delta_str])]
        return df_join, date_delta_str

    def _check_empty_geom(self, df):
        """Issues a warning if there is empty geometry in <df>."""
        msg1 = (
            "Some geometries are empty and will not result in joins on those "
            "rows. Either pass as a normal dataframe with no geometry, or add "
            "the appropriate geometry for all rows."
        )
        msg2 = (
            "A geodataframe was passed to ``join_closest_date()``, but all "
            "geometries are empty and thus no rows will be joined. Please "
            "either pass as a normal dataframe with no geometry, or add in "
            "the appropriate geometry."
        )
        if all(df[df.geometry.name].is_empty):
            raise ValueError(msg2)
        if any(df[df.geometry.name].is_empty):
            warnings.warn(msg1, category=RuntimeWarning, stacklevel=1)

    def _join_geom_date(
        self, df_left, df_right, left_on, right_on, tolerance=3, delta_label=None
    ):
        """
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
        """
        subset_left = db_utils.get_primary_keys(df_left)
        subset_right = db_utils.get_primary_keys(df_right)
        # Get names of geometry columns
        geom_l = df_left.geometry.name + "_l"
        geom_r = df_right.geometry.name + "_r"

        # Missing geometry will get filtered out here
        df_merge1 = df_left.merge(
            df_right,
            how="inner",
            on=subset_left,
            suffixes=["_l", "_r"],
            validate="many_to_many",
        )
        # Grab geometry for each df; because zonal stats were retrieved based
        # on obs_tissue geom, we can keep observations that geometry is equal.
        gdf_merge1_l = gpd.GeoDataFrame(df_merge1[geom_l], geometry=geom_l)
        gdf_merge1_r = gpd.GeoDataFrame(df_merge1[geom_r], geometry=geom_r)
        # Remove all rows whose left geom does not "almost eqaul" right geom
        df_sjoin2 = df_merge1[gdf_merge1_l.geom_almost_equals(gdf_merge1_r, 8)].copy()
        left_on2 = left_on + "_l"
        right_on2 = right_on + "_r"
        df_sjoin2.rename(columns={left_on: left_on2, right_on: right_on2}, inplace=True)
        # Add "date_delta" column
        # if tolerance == 0:
        #     df_sjoin2.dropna(inplace=True)
        # elif tolerance > 0:
        df_sjoin2, delta_label_out = self._add_date_delta(
            df_sjoin2, left_on=left_on2, right_on=right_on2, delta_label=delta_label
        )
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
        df_keep = pd.DataFrame(
            data=[], columns=df_dup.columns
        )  # df for selected entries
        df_keep = df_keep.astype(df_dup.dtypes.to_dict())
        df_unique = df_dup.drop_duplicates(subset=subset, keep="first")[subset]
        for idx, row in df_unique.iterrows():  # Unique subset cols only
            # The magic to get duplicate of a particular unique non-null group
            df_filtered = df_dup[df_dup[row.index].isin(row.values).all(1)]
            # Find index where delta is min and append it to df_keep
            idx_min = abs(df_filtered[delta_label_out]).idxmin(axis=0, skipna=True)
            # df_keep = df_keep.append(df_filtered.loc[idx_min, :])
            df_keep = pd.concat([df_keep, df_filtered.loc[idx_min, :]], axis=0)
        df_join = df_delta.drop_duplicates(subset=subset, keep=False)
        # df_join = df_join.append(df_keep)
        df_join = pd.concat([df_join, df_keep], axis=0)
        df_join = df_join[pd.notnull(df_join[delta_label_out])]
        # drop right join columns,
        if geom_l in df_join.columns:
            df_join = gpd.GeoDataFrame(df_join, geometry=geom_l)
            df_join.rename(columns={geom_l: "geom"}, inplace=True)
            df_join.set_geometry(col="geom", inplace=True)
        df_join.rename(columns={left_on2: left_on}, inplace=True)
        cols_drop = [
            c + side
            for side in ["_l", "_r"]
            for c in ["id", "geom", "geometry"]
            if c + side in df_join.columns
        ] + [right_on2]
        df_join.drop(columns=cols_drop, inplace=True)
        return df_join

    def _add_id_by_subset(self, df, subset, col_id="id"):
        """
        Adds a new column named 'id' to <df> based on unique subset values.
        """
        df_unique_id = df.drop_duplicates(subset=subset, keep="first").sort_values(
            by=subset
        )[subset]
        df_unique_id.insert(0, "id", list(range(len(df_unique_id))))
        df_id = df.merge(df_unique_id, on=subset)
        df_id = df_id[["id"] + [c for c in df_id.columns if c != "id"]]
        return df_id

    def _find_unique_geom(self, df):
        """
        Finds the unique geometries in <df> and adds a new column as UID.

        GeoML utilities?
        """
        df = df.reset_index(drop=True)
        df_copy = df.copy()
        df_copy.loc[:, "unique_geom"] = None

        uid = 0
        for i, r in df.iterrows():
            # break
            if df_copy.loc[i, "unique_geom"] is not None:
                continue  # skip if already found
            else:
                df_copy.loc[
                    df_copy.geom_almost_equals(r.geom, align=False), "unique_geom"
                ] = uid
                uid += 1
        return df_copy

    def _calc_ntd_with_geom(
        self, df, df_join, col_rate_n, col_rate_ntd_out, col_id="id"
    ):
        """
        Calculates nitrogen applied to date for a geodataframe potentially
        having multiple geometries in the same "subset".

        The issue is that with multiple geometries in the same field_id, the
        N applications are being added across geometries all towards the same
        field_id and it is mis-calculated as being much higher N rate than it
        actually was. By separating, we can add up N applied thus far for
        fields with a single geometry, then handle fields with mutliple
        geometries separate.
            # 0. Separate between primary key rows that are unique (those with
            # only a single geometry), and those that aren't (those with
            # multiple geometries).
        Note:
            <df> should already have the join with the <n_applications> table.
        """
        subset = db_utils.get_primary_keys(df)
        cols_require = subset + ["date"]  # Unique subset for which to calculate NTD

        if col_id not in df_join.columns:  # required for multiple geometries
            df_join = self._add_id_by_subset(
                df_join, subset=cols_require, col_id=col_id
            )

        # 1. Get all unique geometries for each year
        # subgeom = subset + [df_join.geometry.name]
        # df_unique = df_join[~df_join.duplicated(
        #     subset=subgeom, keep='first')][subgeom]

        # # 2. Separate fields with only a single geometry from those with
        # # multiple geometries by checking <df_unique> subset.
        # df_subset_1_geom = df_unique[~df_unique.duplicated(
        #     subset=subset, keep=False)][subgeom]
        # df_subset_mult_geom = df_unique[df_unique.duplicated(
        #     subset=subset, keep=False)][subgeom]

        # # 3. Combine single and multiple geometries, and get N app info
        # df_single_mult = pd.concat([df_subset_1_geom, df_subset_mult_geom],
        #                            ignore_index=True)
        # df_join_single_mult = df_join.merge(  # get N app info for all dates
        #     df_single_mult, on=subgeom)

        # 4. Calculate N applied to date, and merge with original <df>
        # df_ntd = df_join_single_mult.groupby(
        #     cols_require + [col_id])[col_rate_n].sum().reset_index()

        # TODO: merge geom and sum 'rate_n_kgha'
        df_ntd = (
            df_join.groupby(cols_require + [col_id])[col_rate_n].sum().reset_index()
        )
        df_ntd = df_ntd.sort_values(by=col_id).reset_index(drop=True)
        df_ntd.rename(columns={col_rate_n: col_rate_ntd_out}, inplace=True)
        df_out = df.merge(
            df_ntd, on=cols_require + [col_id], validate="many_to_one"
        )  # col_id required because it is unique for each observation
        return df_out

    def _fill_empty_geom(self, df):
        """Fills/replaces empty geometry with field_bounds geometry"""
        if self.field_bounds is not None:
            subset = db_utils.get_primary_keys(self.as_planted)

            df_geom = df[~df[df.geometry.name].is_empty]
            df_empty = df[df[df.geometry.name].is_empty]
            df_empty.drop(columns=[df_empty.geometry.name], inplace=True)

            # df_fill = df_geom.append(
            df_fill = pd.concat(
                [
                    df_geom,
                    df_empty.merge(
                        self.field_bounds[subset + [self.field_bounds.geometry.name]],
                        on=subset,
                    ),
                ],
                axis=0,
            )
            return df_fill

    def _spatial_join_clean_keys(self, df_left, df_right):
        """
        Performs spatial join, then only keeps matched primary keys.

        Deletes any rows whose primary keys in the joined dataframes do not
        match, then renames primary keys with standard naming.
        """

        df_join_s = gpd.tools.sjoin(
            df_left, df_right.to_crs(df_left.crs.to_epsg()), how="inner"
        )

        subset = db_utils.get_primary_keys(df_left)
        if "field_id" in subset:
            keys = [
                ["owner_left", "owner_right"],
                ["farm_left", "farm_right"],
                ["field_id_left", "field_id_right"],
                ["year_left", "year_right"],
            ]
        else:
            keys = [
                ["owner_left", "owner_right"],
                ["study_left", "study_right"],
                ["plot_id_left", "plot_id_right"],
                ["year_left", "year_right"],
            ]
        # keys = [['{0}_left'.format(k), '{0}_right'.format(k)] for k in subset]

        df_join = df_join_s.loc[
            (df_join_s[keys[0][0]] == df_join_s[keys[0][1]])
            & (df_join_s[keys[1][0]] == df_join_s[keys[1][1]])
            & (df_join_s[keys[2][0]] == df_join_s[keys[2][1]])
            & (df_join_s[keys[3][0]] == df_join_s[keys[3][1]])
        ]

        # df_join = df_join_s.loc[(df_join_s['owner_left'] == df_join_s['owner_right']) &
        #                         (df_join_s['farm_left'] == df_join_s['farm_right']) &
        #                         (df_join_s['field_id_left'] == df_join_s['field_id_right']) &
        #                         (df_join_s['year_left'] == df_join_s['year_right'])]
        df_join.rename(
            columns={
                keys[0][0]: keys[0][0].split("_left")[0],
                keys[1][0]: keys[1][0].split("_left")[0],
                keys[2][0]: keys[2][0].split("_left")[0],
                keys[3][0]: keys[3][0].split("_left")[0],
            },
            inplace=True,
        )
        df_join.drop(
            columns={
                keys[0][1]: keys[0][1].split("_right")[0],
                keys[1][1]: keys[1][1].split("_right")[0],
                keys[2][1]: keys[2][1].split("_right")[0],
                keys[3][1]: keys[3][1].split("_right")[0],
            },
            # 'owner_right': 'owner', 'farm_right': 'farm',
            # 'field_id_right': 'field_id', 'year_right': 'year'},
            inplace=True,
        )
        return df_join

    def _get_geom_from_primary_keys(self, df):
        """Adds field_bounds geom to <df> based on primary keys"""
        for g in ["goem", "geometry"]:
            if g in df.columns:
                df.drop(columns=[g], inplace=True)
        subset = db_utils.get_primary_keys(df)
        gdf = gpd.GeoDataFrame(
            df.merge(
                self.field_bounds[subset + [self.field_bounds.geometry.name]], on=subset
            ),
            geometry="geom",
        )
        return gdf

    # ##############
    #         # 1. Get all unique geometries
    #         subgeom = subset + [df_join.geometry.name]
    #         df_unique = df_join[~df_join.duplicated(  # don't want id because this is unique geom not obs
    #             subset=subgeom, keep='first')][subgeom]

    #         # 2. Separate fields with only a single geometry from those with
    #         # multiple geometries by checking <df_unique> subset.
    #         df_subset_1_geom = df_unique[~df_unique.duplicated(
    #             subset=subset, keep=False)][subgeom]
    #         df_subset_mult_geom = df_unique[df_unique.duplicated(
    #             subset=subset, keep=False)][subgeom]

    #         # 3. For fields with 1 geom, business as usual
    #         df_join_single = df_join.merge(  # get attributes from <df_join>
    #             df_subset_1_geom, on=subgeom)
    #         df_ntd_single = df_join_single.groupby(cols_require + ['id'])[col_rate_n].sum().reset_index()

    #         # 4. For fields with multiple geom, calculate NTD for each separately
    #         df_join_mult = df_join.merge(
    #             df_subset_mult_geom, on=subgeom)
    #         df_ntd_mult = df_join_mult.groupby(cols_require + ['id'])[col_rate_n].sum().reset_index()  # have to use id because 'geom' won't work

    #         # Combine at point of renaming col
    #         df_single_mult = pd.concat([df_ntd_single, df_ntd_mult], ignore_index=True).sort_values(by='id')
    #         df_single_mult.rename(columns={col_rate_n: col_rate_ntd_out}, inplace=True)

    #         df_out = df.merge(df_single_mult, on=cols_require + ['id'], validate='many_to_one')
    #         # at this point, I think 'id' can be dropped?

    # ##################
    # df_ntd_single.rename(columns={col_rate_n: col_rate_ntd_out}, inplace=True)
    # df_out_single = df.merge(df_ntd_single, on=cols_require)

    # # df_ntd_mult.drop(columns='id', inplace=True)  # have to drop id because it will influence final merge with df
    # df_ntd_mult.rename(columns={col_rate_n: col_rate_ntd_out}, inplace=True)

    # df_out_mult = df.merge(df_ntd_mult, on=cols_require + ['id'], validate='many_to_one')
    # df_out_mult = df_out_mult.drop_duplicates(subset=cols_require + ['id', col_rate_ntd_out])
    # df_out = pd.concat([df_out_single, df_out_mult], ignore_index=True).sort_values(by='id')
    # # at this point, I think 'id' can be dropped?
    # return df_out

    def load_tables(self, **kwargs):
        """
        Loads all tables in ``Tables.table_names``.

        If there is a valid DB connection, it will attempt to load tables from
        the DB first based on ``Tables.table_names.keys()``. If there are any
        tables that were not loaded or are empty, it will attempt to load those
        tables from ``Tables.base_dir_data``/``Tables.table_names.values()``.

        Whether there is a valid DB connection or not, any table whose value is
        ``None`` in ``Tables.table_names.values()`` will not be loaded.

        Parameters:
            base_dir_data (``str``): The base directory containing all the tables
                available to be joined.
        """
        self._set_params_from_kwargs_t(**kwargs)

        if isinstance(self.db, DBHandler):
            print("\nLoading tables from database...")
            for table_name in self.table_names.keys():
                df = self._load_table_from_db(table_name)
                if df is not None:
                    # print(table_name)
                    self._set_table_to_self(table_name, df)
        else:
            msg = (
                "There is no connection to a database, and "
                "<Tables.base_dir_data> was not passed. Thus, there is no "
                "way to load any tables. Please either connect to DB or "
                "pass <base_dir_data>."
            )
            assert self.base_dir_data is not None, msg

        if self.base_dir_data is not None:
            print("\nLoading tables from <base_dir_data>...")
            for table_name, fname in self.table_names.items():
                if self._get_table_from_self(table_name) is None and fname is not None:
                    df = self._load_table_from_file(table_name, fname)
                    if df is not None:
                        # print(table_name)
                        df = db_utils.cols_to_datetime(df)
                        self._set_table_to_self(table_name, df)

        # Now that tables are loaded, fill in empty geometry for tables that
        # we assume take on field_bounds geom if empty: as-planted, n_applications, obs_tissue
        for t in ["as_planted", "n_applications", "obs_tissue"]:
            df = self._get_table_from_self(t)
            df_fill = self._fill_empty_geom(df)
            self._set_table_to_self(t, df_fill)

        # TODO: Function to filter cropscan data (e.g., low irradiance, etc.)
        # TODO: Do any groupby() function for cropscan (and all data for that
        # matter) before putting into the DB/tables - it must line up with the
        # obs_tissue observations or whatever the response variable is.
        if self.rs_cropscan_res is not None:
            subset = db_utils.get_primary_keys(self.rs_cropscan_res)
            df_cs = self.rs_cropscan_res.groupby(subset + ["date"]).mean().reset_index()

            # TODO: the groupby() function removes geometry - add geometry
            # column back in by copying relevant info from rs_cropscan_res

            # in the meantime, chnage to regular DataFrame if there isn't valid geometry
            if isinstance(df_cs, gpd.GeoDataFrame):
                gdf_cs = df_cs.copy()
                try:
                    _ = any(gdf_cs.geom_type)
                except AttributeError:  # if geometry has not been set, AttributeError will be raised.
                    df_cs = pd.DataFrame(gdf_cs)  # change to regular pandas
            self._set_table_to_self("rs_cropscan_res", df_cs)

    def join_closest_date(
        self,
        df_left,
        df_right,
        left_on="date",
        right_on="date",
        tolerance=0,
        direction="nearest",
        delta_label=None,
    ):
        """
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
        """
        subset_left = db_utils.get_primary_keys(df_left)
        subset_right = db_utils.get_primary_keys(df_right)
        msg = (
            "The primary keys in <df_left> are not the same as those in "
            "<df_right>. <join_closest_date()> requires that both dfs have "
            "the same primary keys."
        )
        assert subset_left == subset_right, msg

        cols_require_l = subset_left + [left_on]
        cols_require_r = subset_right + [right_on]
        self._check_col_names(df_left, cols_require_l)
        self._check_col_names(df_right, cols_require_r)
        df_left = self._dt_or_ts_to_date(df_left, left_on)
        df_right = self._dt_or_ts_to_date(df_right, right_on)

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

        if isinstance(df_left, gpd.GeoDataFrame) and isinstance(
            df_right, gpd.GeoDataFrame
        ):
            self._check_empty_geom(df_left)
            self._check_empty_geom(df_right)

            # 2. Case where some geometries are empty, but some are not
            df_join = self._join_geom_date(
                df_left,
                df_right,
                left_on,
                right_on,
                tolerance=tolerance,
                delta_label=delta_label,
            )
        else:
            df_left.sort_values(left_on, inplace=True)
            df_right.sort_values(right_on, inplace=True)
            left_on2 = left_on + "_l"
            right_on2 = right_on + "_r"
            # by = subset_left + [df_left.geometry.name]
            df_join = pd.merge_asof(
                df_left.rename(columns={left_on: left_on2}),
                df_right.rename(columns={right_on: right_on2}),
                left_on=left_on2,
                right_on=right_on2,
                by=subset_left,
                tolerance=pd.Timedelta(tolerance, unit="D"),
                direction=direction,
                suffixes=("_l", "_r"),
            )
            if isinstance(df_left, gpd.GeoDataFrame):
                df_join = gpd.GeoDataFrame(df_join, geometry=df_left.geometry.name)
            if isinstance(df_right, gpd.GeoDataFrame):
                df_join = gpd.GeoDataFrame(df_join, geometry=df_right.geometry.name)

            # if tolerance == 0:
            #     df_join.dropna(inplace=True)
            # elif tolerance > 0:
            df_join, delta_label_out = self._add_date_delta(
                df_join, left_on=left_on2, right_on=right_on2, delta_label=delta_label
            )
            # df_join.dropna(inplace=True)
            df_join = df_join.rename(columns={left_on2: left_on})
            cols_drop = [
                c + side
                for side in ["_l", "_r"]
                for c in ["id"]
                if c + side in df_join.columns
            ] + [right_on2]
            df_join.drop(columns=cols_drop, inplace=True)
        return df_join.reset_index(drop=True)

    def dae(self, df):
        """
        Adds a days after emergence (DAE) column to df.

        Retrieves emergence dates from as_planted table, so <df> should be a
        GeoDataFrame to properly account for multiple as_planted rows for the
        same set of primary keys.

        Parameters:
            df (``pandas.DataFrame``): The input dataframe to add DAE column
                to. Must have the following column to perform the appropriate
                joins: "study", "year", "plot_id", and "date" (must be able to
                be converted to datetime).

        Example:
            >>> import os
            >>> import pandas as pd
            >>> from geoml import Tables

            >>> base_dir_data = os.path.join(os.getcwd(), 'geoml', 'tests', 'testdata')
            >>> fname_petiole = os.path.join(base_dir_data, 'tissue_petiole_NO3_ppm.csv')
            >>> my_join = Tables(base_dir_data=base_dir_data)
            >>> df_pet_no3 = pd.read_csv(fname_petiole)
            >>> df_pet_no3.head(3)
              study  year  plot_id        date   tissue  measure         value
            0   NNI  2019      101  2019-06-25  Petiole  NO3_ppm  18142.397813
            1   NNI  2019      101  2019-07-09  Petiole  NO3_ppm   2728.023000
            2   NNI  2019      101  2019-07-23  Petiole  NO3_ppm   1588.190000

            >>> df_pet_no3 = my_join.dae(df_pet_no3)
            >>> df_pet_no3.head(3)
              study  year  plot_id       date   tissue  measure         value  dae
            0   NNI  2019      101 2019-06-25  Petiole  NO3_ppm  18142.397813   33
            1   NNI  2019      101 2019-07-09  Petiole  NO3_ppm   2728.023000   47
            2   NNI  2019      101 2019-07-23  Petiole  NO3_ppm   1588.190000   61
        """
        subset = db_utils.get_primary_keys(df)
        cols_require = subset + ["date"]
        self._check_col_names(df, cols_require)
        df.loc[:, "date"] = df.loc[:, "date"].apply(pd.to_datetime, errors="coerce")

        if "id" not in df.columns:  # required for multiple geometries
            df = self._add_id_by_subset(df, subset=cols_require)

        if not isinstance(df, gpd.GeoDataFrame):
            df = self._get_geom_from_primary_keys(df)  # Returns GeoDataFrame
        else:
            df = self._fill_empty_geom(df)

        if "field_id" in subset:
            subset_plant = [
                "date_plant",
                "date_emerge",
                "crop",
                "variety",
                "population_nha",
                "description",
            ]
            df_join = None
            for y in sorted(df["year"].unique()):
                df_y = df[df["year"] == y]
                plant_apps_y = self.as_planted[self.as_planted["year"] == y]
                if df_join is None:
                    df_join = gpd.overlay(
                        df_y[["id"] + cols_require + [df.geometry.name]],
                        plant_apps_y[subset_plant + [self.as_planted.geometry.name]],
                        how="intersection",
                    )
                else:
                    df_join = pd.concat(
                        [
                            df_join,
                            gpd.overlay(
                                df_y[["id"] + cols_require + ["geom"]],
                                plant_apps_y[subset_plant + ["geom"]],
                                how="intersection",
                            ),
                        ],
                        axis=0,
                    )
            df_join.rename(columns={df_join.geometry.name: "geom"}, inplace=True)
        elif "plot_id" in subset:
            on = [i for i in subset if i != "plot_id"]
            df_join = df.merge(self.dates_res, on=on, validate="many_to_one")
        if len(df_join) == 0:
            raise RuntimeError(
                'Unable to calculate DAE. Check that "date_emerge" is set in '
                "<as_planted> table for primary keys:\n{0}".format(df[subset])
            )

        df_join.loc[:, "date_emerge"] = df_join.loc[:, "date_emerge"].apply(
            pd.to_datetime, errors="coerce"
        )
        df_join["dae"] = (df_join["date"] - df_join["date_emerge"]).dt.days
        df_out = df.drop(columns="geom").merge(
            df_join[cols_require + ["geom", "dae"]], on=cols_require
        )
        df_out = df_out[list(df.drop(columns="id").columns) + ["dae"]]
        return df_out.drop_duplicates().reset_index(drop=True)

    def dap(self, df):
        """
        Adds a days after planting (DAP) column to df

        Retrieves planting dates from as_planted table, so <df> should be a
        GeoDataFrame to properly account for multiple as_planted rows for the
        same set of primary keys.

        Parameters:
            df (``pandas.DataFrame``): The input dataframe to add DAP column
                to. Must have the following column to perform the appropriate
                joins: "study", "year", "plot_id", and "date" (must be able to
                be converted to datetime).

        Example:
            >>> import os
            >>> import pandas as pd
            >>> from geoml import Tables

            >>> base_dir_data = os.path.join(os.getcwd(), 'geoml', 'tests', 'testdata')
            >>> fname_petiole = os.path.join(base_dir_data, 'tissue_petiole_NO3_ppm.csv')
            >>> my_join = Tables(base_dir_data=base_dir_data)
            >>> df_pet_no3 = pd.read_csv(fname_petiole)
            >>> df_pet_no3.head(3)
              study  year  plot_id        date   tissue  measure         value
            0   NNI  2019      101  2019-06-25  Petiole  NO3_ppm  18142.397813
            1   NNI  2019      101  2019-07-09  Petiole  NO3_ppm   2728.023000
            2   NNI  2019      101  2019-07-23  Petiole  NO3_ppm   1588.190000

            >>> df_pet_no3 = my_join.dap(df_pet_no3)
            >>> df_pet_no3.head(3)
              study  year  plot_id       date   tissue  measure         value  dap
            0   NNI  2019      101 2019-06-25  Petiole  NO3_ppm  18142.397813   54
            1   NNI  2019      101 2019-07-09  Petiole  NO3_ppm   2728.023000   68
            2   NNI  2019      101 2019-07-23  Petiole  NO3_ppm   1588.190000   82
        """
        subset = db_utils.get_primary_keys(df)
        cols_require = subset + ["date"]
        self._check_col_names(df, cols_require)
        df.loc[:, "date"] = df.loc[:, "date"].apply(pd.to_datetime, errors="coerce")

        if "id" not in df.columns:  # required for multiple geometries
            df = self._add_id_by_subset(df, subset=cols_require)

        if not isinstance(df, gpd.GeoDataFrame):
            df = self._get_geom_from_primary_keys(df)  # Returns GeoDataFrame
        else:
            df = self._fill_empty_geom(df)

        if "field_id" in subset:
            subset_plant = [
                "date_plant",
                "date_emerge",
                "crop",
                "variety",
                "population_nha",
                "description",
            ]
            df_join = None
            for y in sorted(df["year"].unique()):
                df_y = df[df["year"] == y]
                plant_apps_y = self.as_planted[self.as_planted["year"] == y]
                if df_join is None:
                    df_join = gpd.overlay(
                        df_y[["id"] + cols_require + [df.geometry.name]],
                        plant_apps_y[subset_plant + [self.as_planted.geometry.name]],
                        how="intersection",
                    )
                else:
                    df_join = pd.concat(
                        [
                            df_join,
                            gpd.overlay(
                                df_y[["id"] + cols_require + ["geom"]],
                                plant_apps_y[subset_plant + ["geom"]],
                                how="intersection",
                            ),
                        ],
                        axis=0,
                    )
            df_join.rename(columns={df_join.geometry.name: "geom"}, inplace=True)
        elif "plot_id" in subset:
            on = [i for i in subset if i != "plot_id"]
            df_join = df.merge(self.dates_res, on=on, validate="many_to_one")
        if len(df_join) == 0:
            raise RuntimeError(
                'Unable to calculate DAP. Check that "date_plant" is set in '
                "<as_planted> table for primary keys:\n{0}".format(df[subset])
            )

        df_join.loc[:, "date_plant"] = df_join.loc[:, "date_plant"].apply(
            pd.to_datetime, errors="coerce"
        )
        df_join["dap"] = (df_join["date"] - df_join["date_plant"]).dt.days
        df_out = df.drop(columns="geom").merge(
            df_join[cols_require + ["geom", "dap"]], on=cols_require
        )
        df_out = df_out[list(df.drop(columns="id").columns) + ["dap"]]

        return df_out.drop_duplicates().reset_index(drop=True)

    def rate_ntd(self, df, col_rate_n="rate_n_kgha", col_rate_ntd_out="rate_ntd_kgha"):
        """
        Adds a column "rate_ntd" indicating the amount of N applied up to the
        date in the df['date'] column (not inclusive).

        First joins df_exp to get unique treatment ids for each plot
        Second joins df_trt to get "breakout" treatment ids for "trt_n"
        Third, joins df_n_apps to get date_applied and rate information
        Fourth, calculates the rate N applied to date (sum of N applied before the
        "date" column; "date" must be a column in df_left)

        Parameters:
            df (``pandas.DataFrame``): The input dataframe to add DAP column
                to. Must have the following column to perform the appropriate
                joins: "study", "year", "plot_id", and "date" (must be able to
                be converted to datetime).
            col_rate_n (``str``): the column name in ``df`` that contains N
                rates
            # unit_str (``str``): A string to be appended to the new column name.
            #     If "kgha", the column name will be "rate_ntd_kgha"

        Example:
            >>> import os
            >>> import pandas as pd
            >>> from geoml import Tables

            >>> base_dir_data = os.path.join(os.getcwd(), 'geoml', 'tests', 'testdata')
            >>> fname_petiole = os.path.join(base_dir_data, 'tissue_petiole_NO3_ppm.csv')
            >>> my_join = Tables(base_dir_data=base_dir_data)
            >>> df_pet_no3 = pd.read_csv(fname_petiole)
            >>> df_pet_no3.head(3)
              study  year  plot_id        date   tissue  measure         value
            0   NNI  2019      101  2019-06-25  Petiole  NO3_ppm  18142.397813
            1   NNI  2019      101  2019-07-09  Petiole  NO3_ppm   2728.023000
            2   NNI  2019      101  2019-07-23  Petiole  NO3_ppm   1588.190000

            >>> df_pet_no3 = my_join.rate_ntd(df_pet_no3)
            >>> df_pet_no3.head(3)
              study  year  plot_id        date   tissue  measure         value rate_ntd_kgha
            0   NNI  2019      101  2019-06-25  Petiole  NO3_ppm  18142.397813       156.919
            1   NNI  2019      101  2019-07-09  Petiole  NO3_ppm   2728.023000       156.919
            2   NNI  2019      101  2019-07-23  Petiole  NO3_ppm   1588.190000       179.336
        """
        subset = db_utils.get_primary_keys(df)
        cols_require = subset + ["date"]
        self._check_col_names(df, cols_require)
        self._cr_rate_ntd(df)  # raises an error if data aren't suitable

        if "id" not in df.columns:  # required for multiple geometries
            df = self._add_id_by_subset(df, subset=cols_require)

        if "field_id" in subset:
            # We need spatial join, but then have to remove excess columns and rename
            subset_n_apps = ["date_applied", "source_n", "rate_n_kgha"]
            df_join = None
            for y in sorted(df["year"].unique()):
                df_y = df[df["year"] == y]
                n_apps_y = self.n_applications[self.n_applications["year"] == y]
                if df_join is None:
                    # df_join = gpd.tools.sjoin(
                    #     n_apps_y[subset_n_apps + ['geom']], df_y[cols_require + ['geom']], how='inner')
                    # df_join = gpd.tools.sjoin(
                    #     df_y[['id'] + cols_require + ['geom']], n_apps_y[subset_n_apps + ['geom']], how='inner')
                    df_join = gpd.overlay(
                        df_y[["id"] + cols_require + ["geom"]],
                        n_apps_y[subset_n_apps + ["geom"]],
                        how="intersection",
                    )
                else:
                    # df_join = df_join.append(gpd.tools.sjoin(
                    #     n_apps_y[subset_n_apps + ['geom']], df_y[cols_require + ['geom']], how='inner'))
                    df_join = pd.concat(
                        [
                            df_join,
                            gpd.overlay(
                                df_y[["id"] + cols_require + ["geom"]],
                                n_apps_y[subset_n_apps + ["geom"]],
                                how="intersection",
                            ),
                        ],
                        axis=0,
                    )

            # Don't drop index_right? Becasue it defines the duplicates from the right df
            if "index_right" in df_join.columns:
                df_join.drop(columns="index_right", inplace=True)
            # df_join_s = gpd.tools.sjoin(fd.n_applications[subset_n_apps + ['geom']], df[cols_require + ['geom']], how='inner')

            # df_join = df_join_s.drop_duplicates(
            #     subset=cols_require + subset_n_apps + ['geom'], keep='first')

            # remove all rows where date_applied is after date
            df_join = df_join[df_join["date_applied"] <= df_join["date"]]

            # df_join = df_join_s.loc[(df_join_s['owner_left'] == df_join_s['owner_right']) &
            #                         (df_join_s['farm_left'] == df_join_s['farm_right']) &
            #                         (df_join_s['field_id_left'] == df_join_s['field_id_right']) &
            #                         (df_join_s['year_left'] == df_join_s['year_right'])]
            # df_join = df_join.rename(columns={
            #     'owner_left': 'owner', 'farm_left': 'farm',
            #     'field_id_left': 'field_id', 'year_left': 'year'})
            # cols_drop = ['index_right', 'owner_right', 'farm_right', 'field_id_right', 'year_right']
            # # if 'id_right' in df_join.columns:
            # #     cols_drop = cols_drop + ['id_right']
            # if 'id_left' in df_join.columns:
            #     cols_drop = cols_drop + ['id_left']
            # df_join.drop(columns=cols_drop, inplace=True)

            df_join.loc[
                ~df_join.geometry.is_valid, df_join.geometry.name
            ] = df_join.loc[~df_join.geometry.is_valid].buffer(0)

            # df_unique = fd._find_unique_geom(df_join)

            df_dissolve = df_join.dissolve(
                by=["id"] + cols_require, as_index=False, aggfunc="sum"
            )

            df_out = df.merge(
                df_dissolve[cols_require + ["rate_n_kgha"]],
                on=cols_require,
                how="left",
                indicator=True,
            )
            df_out = (
                df_out[df_out["_merge"] == "both"]
                .drop(columns="_merge")
                .drop_duplicates()
            )
            df_out.rename(columns={col_rate_n: col_rate_ntd_out}, inplace=True)

            # df_out = fd._calc_ntd_with_geom(df, df_join, col_rate_n,
            #                                   col_rate_ntd_out, col_id='id')

        # The following adds extra N apps due to multiple n_app geometries within a field
        # if 'field_id' in subset:
        #     df_join = df.merge(self.field_bounds[subset], on=subset)
        #     df_join = df_join.merge(
        #         self.n_applications[subset + ['date_applied', col_rate_n]],
        #         on=subset, validate='many_to_many')
        elif "plot_id" in subset:
            df_join = (
                df.merge(self.experiments, on=subset)
                .merge(
                    self.trt,
                    on=["owner", "study", "year", "trt_id"],
                    validate="many_to_many",
                )
                .merge(
                    self.trt_n,
                    on=["owner", "study", "year", "trt_n"],
                    validate="many_to_many",
                )
            )

            # remove all rows where date_applied is after date
            df_join = df_join[df_join["date_applied"] <= df_join["date"]]
            # if isinstance(df, gpd.GeoDataFrame):
            # df_single_geom = df[~df.duplicated(subset=cols_require, keep=False)][cols_require]
            # if len(df_single_geom) != len(df):  # assume there are multiple geometries in some fields
            #     df_out = train._calc_ntd_with_geom(df, df_join, col_rate_n,
            #                                       col_rate_ntd_out, col_id='id')
            # else:
            df_sum = df_join.groupby(cols_require)[col_rate_n].sum().reset_index()
            df_sum.rename(columns={col_rate_n: col_rate_ntd_out}, inplace=True)
            df_out = df.merge(df_sum, on=cols_require)
        return df_out.reset_index(drop=True)
