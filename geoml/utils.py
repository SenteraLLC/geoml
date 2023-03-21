"""Miscellaneous ML functions that were pulled from `mosaic-modeling` and will be re-integrated into GeoML."""
from os.path import join
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array, concatenate
from numpy import linspace as np_linspace
from numpy import mean
from numpy import nan as np_nan
from numpy import sqrt as np_sqrt
from pandas import DataFrame, Series
from pandas import concat as pd_concat
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def sort_features_by_ranking(feature_list, rank_list):
    """Return `feature_list` in sorted order based on `rank_list`."""
    sorted_list = [""] * len(feature_list)
    for i in range(1, len(feature_list) + 1):
        sorted_list[i - 1] = feature_list[rank_list.index(i)]
    return sorted_list


def impute_missing_vals(x_train, x_test):
    """Is a simple imputer from `abi_modeling`.

    This function has not been tested since ABI project.
    """
    imp = SimpleImputer(missing_values=np_nan, strategy="mean")
    if len(x_train) > 0:
        x_train = imp.fit(x_train).transform(x_train)
    if len(x_test) > 0:
        x_test = imp.fit(x_train).transform(
            x_test
        )  # ATTN: Do fit and transform separately so data doesn't leak
    return x_train, x_test


def evaluate_error(y_test, y_pred, y_mean):
    """Estimates accuracy metrics for predicted `y_pred` based on observed `y_test`.

    Accuracy metrics are returned in order as: MAE, relative MAE, RMSE, relative RMSE, R squared.

    Args:
        y_test (numpy.array): Observed values
        y_pred (numpy.array): Predicted values to be compared to `y_test`.
        y_mean (float): Average observed value to be used to calculate relative error metrics.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mae_rel = (mae / y_mean) * 100
    rmse = np_sqrt(mean_squared_error(y_test, y_pred))
    rmse_rel = (rmse / y_mean) * 100

    if y_pred.shape[0] > 1:
        r2 = r2_score(y_test, y_pred)
    else:
        r2 = np_nan
    return mae, mae_rel, rmse, rmse_rel, r2


def train_and_test(
    pipeline,
    search,
    x_train,
    y_train,
    x_test,
    y_test,
    response,
):
    """Fit a tuned model (`search`) with training set and predict on the test set.

    This function has not been tested since ABI project.
    """
    y_mean = mean(concatenate((y_test, y_train)))

    y_pred_test = (
        pipeline.set_params(**search.best_params_).fit(x_train, y_train).predict(x_test)
    )
    mae, mae_rel, rmse, rmse_rel, r2 = evaluate_error(
        y_test, y_pred=y_pred_test, y_mean=y_mean
    )

    model_name = type(pipeline.get_params()["model"]).__name__
    plot_title = f"{model_name} Validation (test set)"

    ax = make_1_to_1_plot(
        y_pred=y_pred_test,
        y_obs=y_test,
        y_obs_mean=y_mean,
        plot_title=plot_title,
        response=response,
    )

    return mae, mae_rel, rmse, rmse_rel, r2, ax


def make_1_to_1_plot(
    y_pred: array,
    y_obs: array,
    y_obs_mean: float,
    plot_title: str,
    response: str,
    plot_save: bool = False,
    model_dir: str = None,
    model_name: str = None,
    hue: List[Any] = None,
):
    """Create 1:1 plot that compares observations with the model predictions.

    Args:
        y_pred (numpy.array): Array of predicted values
        y_obs (numpy.array): Array of observed values (same order as `y_pred`)

        y_obs_mean (float): Mean value of all observations to calculate relative
            accuracy metrics.

        plot_title (str): Text for plot title.
        response (str): Text for variable plotted to be used in axis title.

        plot_save (bool): If True, plot will be saved to `model_dir` as "{model_name}_1_to_1.png"
            and nothing will be returned. Otherwise, the figure will be returned.

        model_dir (str): If `plot_save`, file directory to save 1:1 plot.
        model_name (str): If `plot_save`, model name to be used in 1:1 plot file name

        hue (list): If provided, this list of same length as `y_pred` should be used to determine
            color groupings for plotted points in the 1:1 plot.
    """
    # get accuracy metrics to include in the plot
    mae, mae_rel, rmse, rmse_rel, r2 = evaluate_error(
        y_obs, y_pred=y_pred, y_mean=y_obs_mean
    )

    # use information from the data to determine what the plot bounds should be
    y_pred_stats = DataFrame(y_pred).describe()[0]
    y_obs_stats = DataFrame(y_obs).describe()[0]

    obs_std = y_obs_stats["std"]
    pred_std = y_pred_stats["std"] if y_pred_stats["std"] > 0 else obs_std

    xlim = [
        y_pred_stats["min"] - (pred_std / 2),
        y_pred_stats["max"] + (pred_std / 2),
    ]

    ylim = [
        y_obs_stats["min"] - (obs_std / 2),
        y_obs_stats["max"] + (obs_std / 2),
    ]

    # create plot (include `hue` if it is available)
    df_plot = DataFrame(data={"y_pred": y_pred, "y_obs": y_obs})
    if hue is not None:
        df_plot["color"] = hue
        sns.lmplot(data=df_plot, x="y_pred", y="y_obs", hue="color", fit_reg=False)
        ax = sns.regplot(
            data=df_plot, x="y_pred", y="y_obs", color="black", scatter=False
        )
    else:
        ax = sns.regplot(data=df_plot, x="y_pred", y="y_obs", color="black")

    ax.set(
        title=plot_title,
        xlabel=f"Predicted {response}",
        ylabel=f"Measured {response}",
        xlim=xlim,
        ylim=ylim,
    )
    ax.plot(
        np_linspace(xlim[0], xlim[1], 2),
        np_linspace(ylim[0], ylim[1], 2),
        color="k",
        linestyle="--",
        linewidth=1,
    )
    ax.text(
        xlim[0] + (pred_std / 4),
        ylim[1] - (obs_std / 4),
        f"n = {len(y_obs)}\nR2: {r2:.2f}\nRMSE: {rmse:.2f} ({rmse_rel:.1f} %)\nMAE: {mae:.2f} ({mae_rel:.1f} %)",
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
    )

    # save if desired
    if plot_save:
        fname = join(model_dir, f"{model_name}_1_to_1.png")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(fname)
        plt.close()
        return None
    else:
        return ax


# TODO: What kind of splitting functions already exist? How does our way of defining `cv_kwargs` relate to the way in which `GridSearchCV` understands its `cv` argument?
def train_test_split_custom_func(
    df: DataFrame, cv_kwargs: dict = {"method": "no CV"}, print_output: bool = True
) -> DataFrame:
    """Split `df` into train and test datasets based on passed CV scheme.

    Currently, this function can perform three types of CV: simple random, leave one group out, and
    no CV.

    - If cv_kwargs["method"] is "simple_random", "test_size" (proportion of data to be used as test data)
        and "random_state" must also be passed as keys in `cv_kwargs`.
    - If cv_kwargs["method"] is "leave_group_out", "group_col" (column to consider for CV grouping) and
        "test_col_value" (value of "group_col" to be used as test group) must also be passed as keys in
        `cv_kwargs`.
    - If cv_kwargs["method"] is "no CV", all data is used for training.

    Args:
        df (pandas.DataFrame): Dataset to be split into train/test datasets.
        cv_kwargs (dict): Parameters to set up cross-validation scheme. Defaults to no CV.
        print_output (bool): Indicates whether the test/train sample sizes should be printed to console.
    """
    if cv_kwargs["method"] == "simple_random":
        df_train, df_test = train_test_split(
            df,
            test_size=cv_kwargs["test_size"],
            random_state=cv_kwargs["random_state"],
        )

    elif cv_kwargs["method"] == "leave_group_out":
        assert (
            cv_kwargs["group_col"] in df.columns.values
        ), "Grouping column is not in `df`."

        train_groups = df[cv_kwargs["group_col"]] != cv_kwargs["test_col_value"]
        df_train, df_test = df.loc[train_groups], df.loc[~train_groups]

    else:
        df_train = df.copy()
        df_test = DataFrame(data=None, columns=df.columns)

    if print_output:
        print(f"Number of observations in the training set: {len(df_train)}")
        print(f"Number of observations in the test set:     {len(df_test)}")

    df_train = df_train.assign(train_test="train")
    df_test = df_test.assign(train_test="test")
    df_train_test = pd_concat([df_train, df_test], axis=0, ignore_index=True)

    return df_train_test


def split_x_y_arrays(response: str, df_train_test: DataFrame) -> List:
    """Split `DataFrame` from `train_test_split_custom_func()` into the train/test X/y arrays based on `train_test` column.

    These arrays can then be used with sklearn model objects for ML workflow.
    """
    df_train = df_train_test.loc[df_train_test["train_test"] == "train"].drop(
        columns=["train_test"]
    )
    df_test = df_train_test.loc[df_train_test["train_test"] == "test"].drop(
        columns=["train_test"]
    )

    x_train = df_train.drop([response], axis=1).to_numpy()
    y_train = df_train[response].to_numpy()
    x_test = df_test.drop([response], axis=1).to_numpy()
    y_test = df_test[response].to_numpy()

    return x_train, y_train, x_test, y_test


CV_DF_COLUMNS = [
    "model_name",
    "model_type",
    "n_features",
    "feature_list",
    "response",
    "n",
    "cv_strategy",
    "n_train",
    "n_test",
    "r2_train",
    "r2_test",
    "rmse_train",
    "rmse_test",
    "rmse_rel_test",
    "mae_train",
    "mae_test",
    "mae_rel_test",
]


# TODO: Getting a high level function like cross_validation() that accepts a parameter like mlflow (bool) that dictates
# whether these results should be pushed to the MLFlow database would be a super handy implementation. The idea being that
# this function could be run with or without pushing the data to MLflow.
def cross_validation(
    pipe: Pipeline,
    x_train: array,
    x_test: array,
    df_train_test: DataFrame,
    response_name: str,
    cv_method: str,
    model_name: str,
    features_selected: List[str],
):
    """Perform and organize cross validation results for ML model.

    Args:
        pipe (sklearn Pipeline): skLearn model to be trained and evaluated

        x_train (numpy.array): training dataset containing only the features
            to be used in the model with a row for each observation

        x_test (numpy.array): testing dataset containing the same features as
            contained in `x_train` with a row for each test observation; should be
            None if `cv_method` is None.

        df_train_test (pandas.DataFrame): dataframe containing all data (including
            identification features like "country") with all observations from training
            and testing, includes a column called "train_test" indicating which group an
            observations belongs to

        response_name (str): name of response variable (should be same as column name
            in `df_train_test`)

        cv_method (str): method to be used for cross validation; should be one of:
            "simple_random", "leave_year_out", or None.

        model_name (str): name of model which should uniquely identify `pipe`
        features_selected (list of str): list of feature names used in the model (`pipe`)
    """
    df_mi_template = DataFrame(columns=CV_DF_COLUMNS)  # model info template
    df_train = df_train_test[df_train_test["train_test"] == "train"]
    df_test = df_train_test[df_train_test["train_test"] == "test"]

    # use model to make predictions and evaluate error
    # evaluate training error if desired
    if x_train is not None:
        y_pred_train = pipe.predict(x_train)
        mae_train, _, rmse_train, _, r2_train = evaluate_error(
            y_test=df_train[response_name].to_numpy(),
            y_pred=y_pred_train,
            y_mean=df_train_test[response_name].mean(),  # all relevant observations
        )
    else:
        mae_train, rmse_train, r2_train = (None,) * 3

    # evaluate test error if desired
    if x_test is not None and x_test.shape[0] != 0:
        y_pred_test = pipe.predict(x_test)
        mae_test, mae_rel_test, rmse_test, rmse_rel_test, r2_test = evaluate_error(
            y_test=df_test[response_name].to_numpy(),
            y_pred=y_pred_test,
            y_mean=df_train_test[response_name].mean(),  # all relevant observations
        )
    else:
        mae_test, mae_rel_test, rmse_test, rmse_rel_test, r2_test = (None,) * 5

    # add details to dataframe
    model_info = df_mi_template.to_dict()
    model_info["model_name"] = model_name
    model_info["model_type"] = type(pipe.get_params()["model"]).__name__
    model_info["n_features"] = len(features_selected)
    model_info["feature_list"] = str(features_selected)
    model_info["response"] = response_name
    model_info["n"] = len(df_train_test)
    model_info["cv_strategy"] = cv_method
    model_info["n_train"] = len(df_train)
    model_info["n_test"] = len(df_test)
    model_info["r2_train"] = r2_train
    model_info["r2_test"] = r2_test
    model_info["rmse_train"] = rmse_train
    model_info["rmse_test"] = rmse_test
    model_info["rmse_rel_test"] = rmse_rel_test
    model_info["mae_train"] = mae_train
    model_info["mae_test"] = mae_test
    model_info["mae_rel_test"] = mae_rel_test

    return Series(model_info).to_frame().T.copy()


""