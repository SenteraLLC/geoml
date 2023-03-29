"""Refactored feature selection functions to be re-integrated into GeoML."""
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
from sklearn.utils._testing import ignore_warnings


@ignore_warnings(category=ConvergenceWarning)
def get_n_feats_selected(
    x_train: array,
    y_train: array,
    labels_x: List[str],
    alpha: float,
    random_state: float = 99,
) -> int:
    """Get the number of features selected based on a Lasso model setting."""
    df = get_lasso_feats(
        x_train=x_train,
        y_train=y_train,
        labels_x=labels_x,
        alpha=alpha,
        random_state=random_state,
    )
    feat_n_sel = df["feat_n"].values[0]
    return feat_n_sel


def find_alpha_to_minimize_n_feats(
    x_train: array,
    y_train: array,
    labels_x: List[str],
    alpha_initial: float = 1,
    random_state: float = 99,
    iter_limit: int = 500,
) -> float:
    """Find (rough) minimum Lasso alpha parameter that results in the fitted model having one feature.

    For now, we want to minimize alpha so that we reduce the range of values that are tested in
    `lasso_feature_selection()` and, thereby, improve our chances of getting a variety of feature
    numbers in `df_model_fs` (i.e., smaller range + same number of intervals = smaller intervals).
    This will give the upper bound value for alpha, since higher alpha decreases # of selected features.

    Args:
        x_train (numpy.array): Array of size n x m containing m predictors for model fitting
        y_train (numpy.array): Array of size n x 1 with response variable
        labels_x (List[str]): List of length m which contains feature names in order of appearance in `x_train`
        alpha_initial (float): Initial value of alpha to test for n_feature minimization
        random_state (float): Random state for given Lasso model for reproducibility
        iter_limit (int): Total number of iterations to allow in while loop for alpha optimization
    """
    # test intial value of alpha
    feat_n_sel = get_n_feats_selected(
        x_train=x_train,
        y_train=y_train,
        labels_x=labels_x,
        alpha=alpha_initial,
        random_state=random_state,
    )
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
            feat_n_sel = get_n_feats_selected(
                x_train=x_train,
                y_train=y_train,
                labels_x=labels_x,
                alpha=alpha_now,
                random_state=random_state,
            )

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
            feat_n_sel = get_n_feats_selected(
                x_train=x_train,
                y_train=y_train,
                labels_x=labels_x,
                alpha=alpha_now,
                random_state=random_state,
            )
        alpha_final = alpha_now

    return alpha_final


def find_alpha_to_get_n_feats(
    n_feats: int,
    x_train: array,
    y_train: array,
    labels_x: List[str],
    alpha_initial: float = 1,
    random_state: float = 99,
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
        x_train (numpy.array): Array of size n x m containing m predictors for model fitting
        y_train (numpy.array): Array of size n x 1 with response variable
        labels_x (List[str]): List of length m which contains feature names in order of appearance in `x_train`
        alpha_initial (float): Initial value of alpha to test for n features
        random_state (float): Random state for given Lasso model for reproducibility
        iter_limit (int): Total number of iterations to allow in while loop for alpha optimization

    Returns:
        Maximum alpha value (float) to achieve `n_feats` features in model setting or None if `n_feats` could not be achieved.
    """
    msg = "The length of `labels_x` must be equal to the number of columns in `x_train`"
    assert len(labels_x) == x_train.shape[1], msg

    assert (
        len(labels_x) >= n_feats
    ), f"We cannot select {n_feats} features when only {len(labels_x)} are available in `x_train`."

    # test intial value of alpha
    feat_n_sel = get_n_feats_selected(
        x_train=x_train,
        y_train=y_train,
        labels_x=labels_x,
        alpha=alpha_initial,
        random_state=random_state,
    )
    alpha_now = alpha_initial

    iter_count = -1

    # if we only have one feature, let's just return 0 because we know that will give 1 feature
    if len(labels_x) == 1:
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
            feat_n_sel = get_n_feats_selected(
                x_train=x_train,
                y_train=y_train,
                labels_x=labels_x,
                alpha=alpha_now,
                random_state=random_state,
            )
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
            feat_n_sel = get_n_feats_selected(
                x_train=x_train,
                y_train=y_train,
                labels_x=labels_x,
                alpha=alpha_now,
                random_state=random_state,
            )

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
                return find_alpha_to_get_n_feats(
                    n_feats=n_feats - 1,
                    x_train=x_train,
                    y_train=y_train,
                    labels_x=labels_x,
                    alpha_initial=alpha_initial,
                    random_state=random_state,
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
        feat_n_sel = get_n_feats_selected(
            x_train=x_train,
            y_train=y_train,
            labels_x=labels_x,
            alpha=alpha,
            random_state=random_state,
        )
        if feat_n_sel != n_feats:
            return 0
        else:
            return -alpha

    # To be sure we get the global minimum, we need to narrow in a bit more.
    # Here, we figure out where `n_feats` is achieved along `xline`. Then, we expand
    # one index outside of that range in both directions and pass to optimization function.
    xline = logspace(log(alpha_lower), log(alpha_upper), num=1000, base=np_e)
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


def get_lasso_feats(
    x_train: array,
    y_train: array,
    labels_x: List[str],
    alpha: float,
    random_state: float = 99,
) -> DataFrame:
    """Fit a Lasso model with alpha to predict `y_train` using `x_train`.

    Args:
        x_train (numpy.array): Array of size n x m containing m predictors for model fitting
        y_train (numpy.array): Array of size n x 1 with response variable
        labels_x (List[str]): List of length m which contains feature names in order of appearance in `x_train`
        alpha (float): Lasso alpha parameter to be used when fitting model
        random_state (float): Random state for given Lasso model for reproducibility

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

    model_fs = Lasso()
    model_fs_name = type(model_fs).__name__
    model_fs_params = {
        "precompute": True,
        "max_iter": 100000,
        "tol": 0.001,
        "warm_start": True,
        "selection": "cyclic",
        "random_state": random_state,
        "alpha": alpha,
    }
    model_fs.set_params(**model_fs_params)

    model_fs.fit(x_train, y_train)

    model_bs = SelectFromModel(model_fs, prefit=True)
    feats = model_bs.get_support(indices=True)
    coefs = model_fs.coef_[feats]  # get ranking coefficents
    feat_ranking = rankdata(-abs(coefs), method="min")

    feats_x_select = [labels_x[i] for i in feats]
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


def lasso_feature_selection(
    x_train: array,
    y_train: array,
    labels_x: List[str],
    n_feats: int,
    scaler: Any = None,
    random_state: float = 99,
    n_linspace: int = 1000,
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

    # determine bounds of alpha parameter to test
    alpha_min_feats = find_alpha_to_minimize_n_feats(
        x_train=x_train_copy,
        y_train=y_train,
        labels_x=labels_x,
        random_state=random_state,
    )
    if n_feats > 1:
        alpha_max_feats = find_alpha_to_get_n_feats(
            n_feats=n_feats,
            x_train=x_train_copy,
            y_train=y_train,
            labels_x=labels_x,
            random_state=random_state,
        )
        assert alpha_max_feats is not None, "Did not find alpha to get `n_feats`"
    else:
        alpha_max_feats = alpha_min_feats * 0.1

    param_val_list = list(
        logspace(log(alpha_max_feats), log(alpha_min_feats), num=n_linspace, base=np_e)
    )

    # loop through all alpha values and get resulting features under each fit
    df = None
    for val in param_val_list:
        df_temp = get_lasso_feats(
            x_train=x_train_copy,
            y_train=y_train,
            labels_x=labels_x,
            alpha=val,
            random_state=random_state,
        )
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
            x_train=x_train_copy,
            y_train=y_train,
            plot_save=plot_save,
            model_dir=model_dir,
            model_name=model_name,
            random_state=random_state,
        )

    return df


def plot_alpha_vs_n_features(
    alpha_min: float,
    alpha_max: float,
    x_train: array,
    y_train: array,
    n_logspace: int = 1000,
    plot_save: bool = False,
    model_dir: str = None,
    model_name: str = "model",
    random_state: int = 99,
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

    labels_x = [f"x{ind}" for ind in range(x_train.shape[1])]

    xline = logspace(log(alpha_max), log(alpha_min), num=n_logspace, base=np_e)
    yline = [
        get_n_feats_selected(
            x_train=x_train,
            y_train=y_train,
            labels_x=labels_x,
            alpha=val,
            random_state=random_state,
        )
        for val in xline
    ]
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
