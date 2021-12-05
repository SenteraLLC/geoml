from typing import Any, Dict, Optional, List

import inspect

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from ..utils import AnyDataFrame


def cv_method_check_random_seed(cv_method        : Any,
                                cv_method_kwargs : Dict[str, Any],
                                random_seed      : int,
                               ) -> Dict[str, Any]:
    '''
    If 'random_state' is a valid parameter in <cv_method>, sets from
    <random_seed>.
    '''
    # cv_method_args = inspect.getfullargspec(cv_method)[0]
    # TODO: Add tests for all supported <cv_method>s
    cv_method_args = inspect.signature(cv_method).parameters
    if 'random_state' in cv_method_args:  # ensure random_seed is set correctly
        cv_method_kwargs['random_state'] = random_seed  # if this will get passed to eval(), should be fine since it gets passed to str() first
    return cv_method_kwargs


def check_sklearn_splitter(cv_method        : Any,
                           cv_method_kwargs : Dict[str, Any],
                           cv_split_kwargs  : Optional[Dict[str, Any]] = None,
                           raise_error      : bool  = False
                          ) -> Dict[str, Any]:
    '''
    Checks <cv_method>, <cv_method_kwargs>, and <cv_split_kwargs> for
    continuity.

    Displays a UserWarning or raises ValueError if an invalid parameter
    keyword is provided.

    Parameters:
        raise_error (``bool``): If ``True``, raises a ``ValueError`` if
            parameters do not appear to be available. Otherwise, simply
            issues a warning, and will try to move forward anyways. This
            exists because <inspect.getfullargspec(cv_method)[0]> is
            used to get the arguments, but certain scikit-learn functions/
            methods do not expose their arguments to be screened by
            <inspect.getfullargspec>. Thus, the only way to use certain
            splitter functions is to bypass this check.

    Note:
        Does not check for the validity of the keyword argument(s). Also,
        the warnings does not work as fully intended because when
        <inspect.getfullargspec(cv_method)[0]> returns an empty list,
        there is no either a warning or ValueError can be raised.
    '''
    if cv_split_kwargs is None:
        cv_split_kwargs = {}

    cv_method_args = inspect.getfullargspec(cv_method)[0]
    cv_split_args = inspect.getfullargspec(cv_method.split)[0]

    return cv_split_kwargs


def stratify_set(stratify_cols : List[str] = ['owner', 'farm', 'year'],
                 train_test    : Optional[str] = None,
                 df            : Optional[AnyDataFrame] = None,
                 df_y          : Optional[pd.Series]    = None,
                ) -> AnyDataFrame:
    '''
    Creates a 1-D array of the stratification IDs (to be used by k-fold)
    for both the train and test sets: <stratify_train> and <stratify_test>

    Returns:
        groups (``numpy.ndarray): Array that asssigns each observation to
            a stratification group.
    '''
    if df is None and df_y is not None:
        df = df_y.copy()
    msg1 = ('All <stratify> strings must be columns in <df_y>')
    if df is not None:
      for c in stratify_cols:
          assert c in df.columns, msg1
    if train_test is None:
        groups = df.groupby(stratify_cols).ngroup().values
    else:
        groups = df[df['train_test'] == train_test].groupby(
            stratify_cols).ngroup().values

    unique, counts = np.unique(groups, return_counts=True)
    print('\nStratification groups: {0}'.format(stratify_cols))
    print('Number of stratification groups:  {0}'.format(len(unique)))
    print('Minimum number of splits allowed: {0}'.format(min(counts)))
    return groups


def splitter_eval(cv_split_kwargs : Dict[str, Any],
                  df              : Optional[AnyDataFrame] = None
                 ) -> Dict[str, Any]:
    '''
    Preps the CV split keyword arguments (evaluates them to variables).
    '''
    if cv_split_kwargs is None:
        cv_split_kwargs = {}
    if 'X' not in cv_split_kwargs and df is not None:  # sets X to <df>
        cv_split_kwargs['X'] = 'df'
    scope = locals()

    if df is None and 'df' in [
            i for i in [a for a in cv_split_kwargs.values()]]:
        raise ValueError(
            '<df> is None, but is present in <cv_split_kwargs>. Please '
            'pass <df> or ajust <cv_split_kwargs>')
    # evaluate any str; keep anything else as is
    cv_split_kwargs_eval = dict(
        (k, eval(str(cv_split_kwargs[k]), scope))
        if isinstance(cv_split_kwargs[k], str)
        else (k, cv_split_kwargs[k])
        for k in cv_split_kwargs)

    return cv_split_kwargs_eval


