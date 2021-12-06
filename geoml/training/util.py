import pandas as pd  # type: ignore

from ..utils import AnyDataFrame

from typing import List, Tuple


def fs_get_X_select(X_train          : AnyDataFrame,
                    X_test           : AnyDataFrame,
                    df_fs_params     : AnyDataFrame,
                    df_fs_params_idx : int,
                   ) -> Tuple[AnyDataFrame, AnyDataFrame, List[str], List[int]]:
    '''
    References <df_fs_params> to provide a new matrix X that only includes
    the selected features.

    Parameters:
        df_fs_params_idx (``int``): The index of <df_fs_params> to retrieve
            sklearn model parameters (the will be stored in the "params"
            column).

    Example:
        >>> from geoml import FeatureSelection
        >>> from geoml.tests import config

        >>> myfs = FeatureSelection(config_dict=config.config_dict)
        >>> myfs.fs_find_params()
        >>> X_train_select, X_test_select = myfs.fs_get_X_select(2)
    '''
    msg1 = ('<df_fs_params> must be populated; be sure to execute '
            '``find_feat_selection_params()`` prior to running '
            '``get_X_select()``.')
    assert isinstance(df_fs_params, pd.DataFrame), msg1

    feats_x_select = df_fs_params.loc[df_fs_params_idx]['feats_x_select']

    X_train_select = X_train[:,feats_x_select]
    X_test_select = X_test[:,feats_x_select]
    X_train_select = X_train_select
    X_test_select = X_test_select

    # TODO: Are these ever used?
    labels_x_select = df_fs_params.loc[df_fs_params_idx]['labels_x_select']
    rank_x_select = df_fs_params.loc[df_fs_params_idx]['rank_x_select']

    return X_train_select, X_test_select, labels_x_select, rank_x_select

