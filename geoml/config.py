from typing import TypedDict, Dict, Union, Set, Tuple, List, Optional, Literal, Any


class DBConfig(TypedDict):
    name   : str
    host   : str
    user   : str
    schema : str
    port   : int


class TableConfig(TypedDict):
    database    : DBConfig
    table_names : Dict[str, str]


class GroupFeatures(TypedDict):
    dae                 : str
    dap                 : str
    rate_ntd            : Dict[str, str]
    sentinel_wl_range   : Tuple[int, int]
    weather_derived     : List[str]
    weather_derived_res : List[str]
    bands               : List[str]


class FeatureDataConfig(TypedDict):
    random_seed          : int
    dir_results          : Optional[str]
    group_feats          : GroupFeatures
    ground_truth_tissue  : str
    ground_truth_measure : str
    date_tolerance       : int
    # TODO: These kwarg arguments can be much more flexible
    cv_method_kwargs      : Dict[str, int]
    cv_split_kwargs       : Dict[str, str]
    impute_method         : str
    train_test            : str
    cv_method_tune_kwargs : Dict[str, float]
    cv_split_tune_kwargs  : Dict[str, List[str]]
    print_out_fd          : bool
    print_spliiter_info   : bool


class FeatureSelectionConfig(TypedDict):
    model_fs                   : Any
    model_fs_params_set        : Dict[str, Any]
    model_fs_params_adjust_min : Dict[str, Any]
    model_fs_params_adjust_max : Dict[str, Any]
    n_feats                    : int
    n_linspace                 : int


class TrainingConfig(TypedDict):
    regressor : Any
    regressor_params : Dict[str, Any]
    param_grid       : Dict[str, Any]
    n_jobs_tune      : int
    scoring          : List[str]
    refit            : str
    rank_scoring     : str
    print_out_train  : bool


class Config(TypedDict):
    table             : TableConfig
    feature_data      : FeatureDataConfig
    feature_selection : FeatureSelectionConfig
    training          : TrainingConfig



