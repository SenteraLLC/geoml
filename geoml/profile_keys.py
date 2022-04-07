PROFILE_REQUIRED_KEYS = set(
    ["driver", "dtype", "nodata", "width", "height", "count", "crs", "transform"]
)

PROFILE_GTIFF_KEYS = set(["AREA_OR_POINT", "COMPRESS", "INTERLEAVE"])

PROFILE_ENVI_BASIC_KEYS = set(
    [
        "sensor_type",
        "acquisition_time",
        "samples",
        "lines",
        "bands",
        "data_ignore_value",
        "description",
        "header_offset",
        "map_info",
        "coordinate_system_string",
        "cloud_cover",
        "pixel_size",
        "band_names",
    ]
)

PROFILE_ENVI_REFLECTANCE_KEYS = set(
    ["wavelength_units", "wavelength", "fwhm", "reflectance_scale_factor"]
)

PROFILE_PREDICTION_KEYS1 = (
    PROFILE_REQUIRED_KEYS | PROFILE_GTIFF_KEYS | PROFILE_ENVI_BASIC_KEYS
)

PROFILE_GEOML_KEYS = set(
    [
        "pkeys",
        "date_train",
        "response",
        "response_units",
        "date_predict",
        "feats_x_select",
        "estimator",
        "estimator_parameters",
        "geoml_version",
    ]
)

PROFILE_SENTINEL2_KEYS = set(
    [
        "url",
    ]
)
