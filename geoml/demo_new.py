import json
import os
from datetime import datetime

import numpy as np

from typing import Optional, Dict, Any

from .utils import AnyDataFrame, write_to_readme

from .config.config import Config, TableConfig, FeatureDataConfig
from .config.parse import parseConfig


def main(base_dir_data : str,
         config_path   : str,
         password      : Optional[str],
        ) -> None:

    with open(config_path) as f:
        to_parse : Dict[str, Any] = json.load(f)
        config = parseConfig(to_parse)

    random_seed = 999

    # TODO: Refactor into separate functions
    #       Should probably be done once intermediate types are solidified

    # Tables

    from .tables.load import connect_to_db, load_tables

    table_config : TableConfig = config["table"]

    db = connect_to_db(table_config["database"], password=password)
    tables = load_tables(db, table_config, base_dir_data)


    # Feature Data

    from .feature_data.test_and_train import get_test_and_train
    from .feature_data.feat_group import get_feat_group_X_y

    from .feature_data.load import load_df_response, get_X_and_y, get_tuning_splitter

    feature_data_config : FeatureDataConfig = config["feature_data"]

    dir_results = feature_data_config["dir_results"]
    write_to_readme('Random seed: {0}'.format(random_seed), dir_results)

    ground_truth_tissue  = feature_data_config["ground_truth_tissue"]
    ground_truth_measure = feature_data_config["ground_truth_measure"]

    # TODO: Is this run every time?
    df_response, labels_y_id, label_y = load_df_response(tables,
                                                         ground_truth_tissue,
                                                         ground_truth_measure,
                                                        )

    if dir_results is not None:
        os.makedirs(dir_results, exist_ok=True)

    group_feats      = feature_data_config["group_feats"]
    date_tolerance   = feature_data_config["date_tolerance"]
    cv_method        = feature_data_config["cv_method"]
    cv_method_tune   = feature_data_config["cv_method_tune"]
    cv_method_kwargs = feature_data_config["cv_method_kwargs"]
    cv_split_kwargs  = feature_data_config["cv_split_kwargs"]
    impute_method    = feature_data_config["impute_method"]
    date_train       = feature_data_config["date_train"]

    df = get_feat_group_X_y(df_response,
                            tables,
                            group_feats,
                            ground_truth_tissue,
                            ground_truth_measure,
                            date_tolerance,
                            date_train,
                            random_seed,
                            cv_method,
                            cv_method_kwargs,
                            cv_split_kwargs,
                           )


    X_train, X_test, y_train, y_test, df, labels_x, = get_test_and_train(df,
                                                                         label_y,
                                                                         group_feats,
                                                                         random_seed,
                                                                         impute_method
                                                                        )

    df_X, df_y = get_X_and_y(df,
                             labels_x,
                             label_y,
                             labels_y_id,
                             dir_results,
                            )

    cv_method_tune_kwargs = feature_data_config["cv_method_tune_kwargs"]
    cv_split_tune_kwargs = feature_data_config["cv_split_tune_kwargs"]

    tuning_splitter = get_tuning_splitter(df, df_X, cv_method_tune, cv_method_tune_kwargs, cv_split_tune_kwargs, random_seed)


    # Feature Selection #

    from .feature_selection.select import set_model_fs, fs_find_params

    feature_selection_config   = config["feature_selection"]

    model_fs                   = feature_selection_config["model_fs"]
    print("Model fs.")
    model_fs_params_set        = feature_selection_config["model_fs_params_set"]
    n_feats                    = feature_selection_config["n_feats"]
    n_linspace                 = feature_selection_config["n_linspace"]
    model_fs_params_adjust_min = feature_selection_config["model_fs_params_adjust_min"]

    model_fs, model_fs_name = set_model_fs(model_fs, model_fs_params_set, random_seed)

    df_fs_params = fs_find_params(X_train, y_train, model_fs, model_fs_name, labels_x, n_feats, n_linspace, model_fs_params_adjust_min)


    # Training #
    from .training.fit import fit

    training_config = config["training"]

    regressor        = training_config["regressor"]
    param_grid       = training_config["param_grid"]
    regressor_params = training_config["regressor_params"]
    n_jobs_tune      = training_config["n_jobs_tune"]
    scoring          = training_config["scoring"]
    refit            = training_config["refit"]
    rank_scoring     = training_config["rank_scoring"]

    df_tune, df_test_full, df_pred, df_pred_full, df_test = fit(df_y, df_fs_params, X_train, y_train, X_test, y_test, tuning_splitter, n_jobs_tune, regressor, regressor_params, refit, rank_scoring, model_fs_name, param_grid, scoring, random_seed)


    # Predict #
    feats_x_select = df_test.loc[
        df_test[df_test['feat_n'] == n_feats].index,
        'feats_x_select'].item()
    estimator = df_test.loc[
        df_test[df_test['feat_n'] == n_feats].index,
        'regressor'].item()

    from .predict import predict

    predict_config = config["predict"]

    date_predict = predict_config["date_predict"]
    primary_keys_pred = predict_config["primary_keys_pred"]
    image_search_method = predict_config["image_search_method"]

    if db is not None:
        array_pred, profile = predict(db, estimator, primary_keys_pred, group_feats, tables, feats_x_select, date_predict, image_search_method)
        print("Array pred: ",array_pred)
        print("Profile: ",profile)

    import sys
    sys.exit(1)






import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a field and all of its input data as JSON')
    parser.add_argument('--base_dir_data', type=str, help='Directory with data', required=False)
    parser.add_argument('--config', type=str, help='Table config', required=True)
    parser.add_argument('--password', type=str, help='Database password')

    args = parser.parse_args()

    main(args.base_dir_data, args.config, args.password)


