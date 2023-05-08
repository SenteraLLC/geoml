"""Refactored feature selection functions to be re-integrated into GeoML."""
import logging
from os.path import join
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array, copy
from numpy import e as np_e
from numpy import log, logspace
from pandas import DataFrame
from pandas import concat as pd_concat
from scipy import optimize
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

FS_PARAMS = {
    "Lasso": {
        "precompute": True,
        "max_iter": 100000,
        "tol": 0.001,
        "warm_start": True,
        "selection": "cyclic",
        "random_state": None,
        "alpha": None,
    },
    "LogisticRegression": {
        "max_iter": 100,
        "tol": 0.001,
        "warm_start": True,
        "solver": "saga",
        "penalty": "l1",
        "random_state": None,
        "C": None,
    },
    "LinearSVC": {
        "max_iter": 100,
        "dual": False,
        "tol": 0.001,
        "penalty": "l1",
        "random_state": None,
        "C": None,
    },
}


@ignore_warnings(category=ConvergenceWarning)
def _fit_model_fs(config: dict, alpha: float) -> Any:
    """Set parameters and fit feature selection model based on `config` and `alpha`."""
    # get feature selection model type based on `response_type`
    if config["response_type"] == "regression":
        model_fs = Lasso()
        par_name = "alpha"
    else:
        model_fs = LinearSVC()
        par_name = "C"

    model_fs_name = type(model_fs).__name__

    # get and set parameters
    model_fs_params = FS_PARAMS[model_fs_name]
    model_fs_params["random_state"] = config["random_state"]
    model_fs_params[par_name] = 1 / alpha

    model_fs.set_params(**model_fs_params)
    model_fs.fit(config["X"], config["y"])

    return model_fs


def _get_feats_for_alpha(
    config: dict,
    alpha: float,
) -> DataFrame:
    """Fit a Lasso model with alpha to predict `y_train` using `x_train`.

    Args:
        config (dict): Dictionary containing `x_train`, `y_train`, `labels_x`, `random_state`, & `response_type`
        alpha (float): Lasso alpha parameter to be used when fitting model

    Returns DataFrame row with information on the fitted model, including number of features,
    features selected, and their ranking by the Lasso model.
    """
    cols = [
        "model_fs",
        "params_fs",
        "feat_n",
        "feats_x_select",
        "labels_x_select",
        "rank_x_select",
    ]

    model_fs = _fit_model_fs(config=config, alpha=alpha)
    model_fs_name = type(model_fs).__name__

    model_bs = SelectFromModel(model_fs, prefit=True)
    feats = model_bs.get_support(indices=True)

    if len(model_fs.coef_.shape) == 2:
        coefs = model_fs.coef_[:, feats]
    else:
        coefs = model_fs.coef_[feats]  # get ranking coefficents

    feat_ranking = rankdata(-abs(coefs), method="min")

    feats_x_select = [config["labels_x"][i] for i in feats]
    data = [
        model_fs_name,
        model_fs.get_params(),
        len(feats),
        tuple(feats),
        tuple(feats_x_select),
        tuple(feat_ranking),
    ]
    df = DataFrame(data=[data], columns=cols)

    return df


def _get_n_feats_selected(
    config: dict,
    alpha: float,
) -> int:
    """Get the number of features selected based on a Lasso model setting."""
    df = _get_feats_for_alpha(
        config=config,
        alpha=alpha,
    )
    feat_n_sel = df["feat_n"].values[0]

    return feat_n_sel


def _find_alpha_to_minimize_n_feats(
    config: dict,
    alpha_initial: float = 1,
    iter_limit: int = 500,
) -> float:
    """Find (rough) minimum Lasso alpha parameter that results in the fitted model having one feature.

    For now, we want to minimize alpha so that we reduce the range of values that are tested in
    `lasso_feature_selection()` and, thereby, improve our chances of getting a variety of feature
    numbers in `df_model_fs` (i.e., smaller range + same number of intervals = smaller intervals).
    This will give the upper bound value for alpha, since higher alpha decreases # of selected features.

    Args:
        config (dict): Dictionary containing `x_train`, `y_train`, `labels_x`, `random_state`, & `response_type`
        alpha_initial (float): Initial value of alpha to test for n_feature minimization
        iter_limit (int): Total number of iterations to allow in while loop for alpha optimization
    """
    # test intial value of alpha
    feat_n_sel = _get_n_feats_selected(config=config, alpha=alpha_initial)
    alpha_now = alpha_initial

    iter_count = -1

    # if initial value results in 1 feature, roughly minimize alpha to obtain 1 feature
    if feat_n_sel <= 1:
        while feat_n_sel <= 1:
            iter_count += 1
            if iter_count >= iter_limit:
                raise Exception(
                    f"Iteration limit reached. Reducing alpha to {alpha_now} did not achieve 1 feature."
                )

            params_last = alpha_now
            alpha_now /= 1.2
            feat_n_sel = _get_n_feats_selected(config=config, alpha=alpha_now)

        alpha_final = params_last
    else:
        # if initial value is too low, need to increase alpha to eliminate more features
        while feat_n_sel > 1:
            iter_count += 1
            if iter_count >= iter_limit:
                raise Exception(
                    f"Iteration limit reached. Increasing alpha to {alpha_now} did not achieve 1 feature."
                )

            alpha_now *= 1.2
            feat_n_sel = _get_n_feats_selected(config=config, alpha=alpha_now)
        alpha_final = alpha_now

    return alpha_final


def _find_alpha_to_get_n_feats(
    n_feats: int,
    config: dict,
    alpha_initial: float = 1,
    iter_limit: int = 500,
) -> float:
    """Find (rough) maximum Lasso alpha parameter that results in the fitted model having `n_feats`.

    For now, we want to maximize alpha so that we reduce the range of values that are tested in
    `lasso_feature_selection()` and, thereby, improve our chances of getting a variety of feature
    numbers in `df_model_fs` (i.e., smaller range + same number of intervals = smaller intervals).
    This will give the lower bound value for alpha, since lower alpha increases # of selected features.

    This function operates in four parts:

    (1) Use "while" loops to search for a roughly maximum alpha value that achieves `n_feats`, which
    is called `alpha_now`. Construct an alpha range around `alpha_now` using factors of 10.

    (2) A custom lasso feature selection function is created which returns -alpha if `n_feats` is achieved.
    Otherwise, 0 is returned. To minimize this function over an alpha range would be to maximize alpha.

    (3) Narrow the alpha range created in (1) to ensure that the global "minimum" value is achieved.
    Determine where `n_feats` is achieved at regular intervals in the alpha range using the function in (2).
    Then, expand the range out one index in either direction.

    (4) Use scipy.optimize.minimize_scalar() to minimize the function in (2) across alpha range from (3).

    Raises Exception if the optimization step is unsuccessful.

    Args:
        n_feats (int): Number of features to obtain in model
        config (dict): Dictionary containing `x_train`, `y_train`, `labels_x`, `random_state`, & `response_type`
        alpha_initial (float): Initial value of alpha to test for n features
        iter_limit (int): Total number of iterations to allow in while loop for alpha optimization

    Returns:
        Maximum alpha value (float) to achieve `n_feats` features in model setting or None if `n_feats` could not be achieved.
    """
    msg = "The length of `labels_x` must be equal to the number of columns in `x_train`"
    assert len(config["labels_x"]) == config["X"].shape[1], msg

    n_feats_available = len(config["labels_x"])
    assert (
        n_feats_available >= n_feats
    ), f"We cannot select {n_feats} features when only {n_feats_available} are available in `x_train`."

    # test intial value of alpha
    feat_n_sel = _get_n_feats_selected(config=config, alpha=alpha_initial)
    alpha_now = alpha_initial

    iter_count = -1

    # if we only have one feature, let's just return 0 because we know that will give 1 feature
    if n_feats_available == 1:
        return 0

    # more feats selected than n_feats => need a greater alpha value
    if feat_n_sel >= n_feats:
        while feat_n_sel >= n_feats:
            iter_count += 1
            if iter_count >= iter_limit:
                raise Exception(
                    f"Iteration limit reached. Increasing alpha to {alpha_now} did not achieve {n_feats} features."
                )

            alpha_now *= 10
            feat_n_sel = _get_n_feats_selected(config=config, alpha=alpha_now)
        alpha_lower = alpha_now / 10
        alpha_upper = alpha_now
    else:
        # too few features => need a lower alpha value
        while (feat_n_sel < n_feats) and (alpha_now > 0):
            iter_count += 1
            if iter_count >= iter_limit:
                raise Exception(
                    f"Iteration limit reached. Reducing alpha to {alpha_now} did not achieve {n_feats} features."
                )

            alpha_now /= 10
            feat_n_sel = _get_n_feats_selected(config=config, alpha=alpha_now)

        # if alpha_now reaches 0, a model given `n_feats` seems unlikely to be useful
        # let's try a lower number of features
        if alpha_now == 0:
            if n_feats == 1:
                print(f"Could not reasonably achieve {n_feats} features.")
                return None
            else:
                print(
                    f"Could not reasonably achieve {n_feats} features. Trying with `n_feats`={n_feats-1}"
                )
                return _find_alpha_to_get_n_feats(
                    n_feats=n_feats - 1,
                    config=config,
                    alpha_initial=alpha_initial,
                    iter_limit=iter_limit,
                )
        else:
            alpha_lower = alpha_now
            alpha_upper = alpha_now * 10

    # then maximize the alpha value that can obtain this number of feats
    def maximize_alpha(alpha, n_feats):
        """Maximize alpha based on model setting using this helper function.

        If `n_feats` is achieved, the function returns negative `alpha`.
        Otherwise, it returns 0. Using this function in a minimization-
        optimization function, we find the greatest value of alpha to achieve
        `n_feats`.
        """
        feat_n_sel = _get_n_feats_selected(config=config, alpha=alpha)
        if feat_n_sel != n_feats:
            return 0
        else:
            return -alpha

    # To be sure we get the global minimum, we need to narrow in a bit more.
    # Here, we figure out where `n_feats` is achieved along `xline`. Then, we expand
    # one index outside of that range in both directions and pass to optimization function.
    xline = logspace(log(alpha_lower), log(alpha_upper), num=100, base=np_e)
    yline = [maximize_alpha(a, n_feats) for a in xline]
    x_inds = [i for i in range(len(yline)) if yline[i] != 0]
    first_ind = x_inds[0] if x_inds[0] == 0 else x_inds[0] - 1
    last_ind = x_inds[-1] if x_inds[-1] == 99 else x_inds[-1] + 1

    new_alpha_lower = xline[first_ind]
    new_alpha_upper = xline[last_ind]

    result = optimize.minimize_scalar(
        lambda a: maximize_alpha(a, n_feats),
        bounds=(new_alpha_lower, new_alpha_upper),
        method="Bounded",
        options={"maxiter": 100},
    )

    if result["success"] is True:
        return result["x"]
    else:
        raise Exception(
            "Optimization step was unsuccessful. This is a rare error that could result from a low sample size when narrowing alpha range in step (3)."
        )


def lasso_feature_selection(
    x_train: array,
    y_train: array,
    labels_x: List[str],
    n_feats: int,
    scaler: Any = None,
    response_type: str = "regression",
    random_state: float = 99,
    n_linspace: int = 100,
    plot_save: bool = False,
    model_dir: str = None,
    model_name: str = "model",
) -> DataFrame:
    """
    Adapted from GeoML's `FeatureSelection` class function "_lasso_fs_df()".

    Args:
        x_train (numpy.array): Array of size n x m containing m predictors for model fitting
        y_train (numpy.array): Array of size n x 1 with response variable
        labels_x (List[str]): List of length m which contains feature names in order of appearance in `x_train`
        alpha (float): Lasso alpha parameter to be used when fitting model
        n_feats (int): Maximum number of features desired.
        scaler (Any): Optional transformer/scaler to be applied in Pipeline object with Lasso(); defaults to None.
        random_state (float): Random state for given Lasso model for reproducibility
        n_linspace (int): Number of alpha values to consider between minimum and maximum values
        plot_save (bool): Indicates whether the alpha parameter space should be plotted against `n_feats` and saved
        model_dir (str): Directory location to store alpha figure if `plot_save` = True
        model_name (str): Model name to denote the lasso alpha figure in filename if `plot_save` = True
    """
    assert (
        len(labels_x) == x_train.shape[1]
    ), "`labels_x` and `x_train` do not agree on the total number of features."

    assert n_feats <= len(
        labels_x
    ), "Maximum number of features desired is greater than the number available."

    # transform data if scaler is set
    if scaler is not None:
        sclr = scaler()
        sclr.fit(x_train)
        x_train_copy = sclr.transform(x_train)
    else:
        x_train_copy = copy(x_train)

    # set up model config to keep things clean
    config = {
        "X": x_train_copy,
        "y": y_train,
        "labels_x": labels_x,
        "response_type": response_type,
        "random_state": random_state,
    }

    # determine bounds of alpha parameter to test
    logging.info("Finding alpha to minimize number of features...")
    alpha_min_feats = _find_alpha_to_minimize_n_feats(config=config)

    logging.info("Finding alpha to get maximum number of features...")
    if n_feats > 1:
        alpha_max_feats = _find_alpha_to_get_n_feats(n_feats=n_feats, config=config)
        assert alpha_max_feats is not None, "Did not find alpha to get `n_feats`"
    else:
        alpha_max_feats = alpha_min_feats * 0.1

    param_val_list = list(
        logspace(log(alpha_max_feats), log(alpha_min_feats), num=n_linspace, base=np_e)
    )

    # loop through all alpha values and get resulting features under each fit
    df = None
    for ind in tqdm(range(len(param_val_list)), desc="Creating `df_model_fs`:"):
        val = param_val_list[ind]
        df_temp = _get_feats_for_alpha(config=config, alpha=val)
        if df is None:
            df = df_temp.copy()
        else:
            df = pd_concat([df, df_temp], axis=0, ignore_index=True)

    # clean up df
    df = df.drop_duplicates(subset=["feats_x_select"], ignore_index=True)
    df = df[df["feat_n"] <= n_feats]
    df = df[df["feat_n"] > 0]

    if plot_save is True:
        plot_alpha_vs_n_features(
            alpha_min=alpha_min_feats,
            alpha_max=alpha_max_feats,
            config=config,
            plot_save=plot_save,
            model_dir=model_dir,
            model_name=model_name,
        )

    return df


def plot_alpha_vs_n_features(
    alpha_min: float,
    alpha_max: float,
    config: dict,
    n_logspace: int = 100,
    plot_save: bool = False,
    model_dir: str = None,
    model_name: str = "model",
) -> None:
    """Create plot to demonstrate relationship between alpha and number of selected features for given model setting.

    Args:
        alpha_min (float): Minimum value of alpha to plot
        alpha_max (float): Maximum value of alpha to plot
        x_train (numpy.array): Array of size n x m containing m predictors for model fitting
        y_train (numpy.array): Array of size n x 1 with response variable
        n_logspace (int): Number of alpha values to consider between minimum and maximum values; defaults to 1000.
        plot_save (bool): If True, plot is saved to `model_dir`; else, the plot is shown.
        model_dir (str): Directory location to store alpha figure if `plot_save` = True
        model_name (str): Model name to denote the lasso alpha figure in filename if `plot_save` = True; defaults to "model".
        random_state (float): Random state for given Lasso model for reproducibility
    """
    if plot_save:
        msg = "Please set directory where plot should be saved using `model_dir`."
        assert model_dir is not None, msg

    labels_x = [f"x{ind}" for ind in range(config["X"].shape[1])]
    config_copy = config.copy()
    config_copy["labels_x"] = labels_x

    xline = logspace(log(alpha_max), log(alpha_min), num=n_logspace, base=np_e)
    yline = [_get_n_feats_selected(config=config_copy, alpha=val) for val in xline]
    _, ax = plt.subplots(figsize=(6, 4))
    ax.set(
        title="Lasso alpha vs. number of features",
        xlabel="Alpha value",
        ylabel="Number of features selected",
    )
    sns.lineplot(x=xline, y=yline, color="k", ax=ax)

    if plot_save:
        fname = join(model_dir, f"{model_name}_lasso_alpha.png")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(fname)
    else:
        plt.show()

    plt.close()
