import pandas as pd  # type: ignore

from sklearn.experimental import enable_iterative_imputer  # type: ignore
from sklearn.impute import IterativeImputer  # type: ignore
from sklearn.impute import KNNImputer  # type: ignore

from typing import Tuple, List, Optional, Set, Dict

from typing import cast

from ..utils import AnyDataFrame
from ..config.config import GroupFeatures


def _handle_wl_cols(c        : str,
                    wl_range : Tuple[int, int],
                    labels_x : List[str],
                    prefix   : str = 'wl_'
                   ) -> List[str]:
    if not isinstance(c, str):
        c = str(c)
    col = c.replace(prefix, '') if prefix in c else c

    if (col.isnumeric() and int(col) >= wl_range[0] and
        int(col) <= wl_range[1]):
        labels_x.append(c)
    return labels_x


def _get_labels_x(group_feats : GroupFeatures,
                  cols        : Optional[Set[str]] = None
                 ) -> List[str]:
      '''
      Parses ``group_feats`` and returns a list of column headings so that
      a df can be subset to build the X matrix with only the appropriate
      features indicated by ``group_feats``.
      '''

      labels_x : List[str] = []

      for key, feature in group_feats.items():
          print('Loading <group_feats> key: {0}'.format(key))
          if 'wl_range' in key:
              wl_range = group_feats[key]  # type: ignore
              assert cols is not None, ('``cols`` must be passed.')
              for c in cols:
                  labels_x = _handle_wl_cols(c, wl_range, labels_x,
                                                  prefix='wl_')
          elif 'bands' in key or 'weather_derived' in key or 'weather_derived_res' in key:
              labels_x.extend(cast(List[str], feature))
          elif 'rate_ntd' in key:
              labels_x.append(cast(Dict[str, str], feature)['col_out'])
          else:
              labels_x.append(cast(str, feature))
      return labels_x


def _impute_missing_data(X           : AnyDataFrame,
                         random_seed : int,
                         method      : str = 'iterative'
                        ) -> AnyDataFrame:
    '''
    Imputes missing data in X - sk-learn models will not work with null data

    Parameters:
        method (``str``): should be one of "iterative" (takes more time)
            or "knn" (default: "iterative").
    '''
    if pd.isnull(X).any() is False:
        return X

    if method == 'iterative':
        imp = IterativeImputer(max_iter=10, random_state=random_seed)
    elif method == 'knn':
        imp = KNNImputer(n_neighbors=2, weights='uniform')
    elif method == None:
        return X
    X_out = imp.fit_transform(X)

    return X_out


def get_test_and_train(df            : AnyDataFrame,
                       label_y       : str,
                       group_feats   : GroupFeatures,
                       random_seed   : int,
                       impute_method : str = 'iterative'
                      ) -> Tuple[AnyDataFrame, AnyDataFrame, pd.Series, pd.Series, AnyDataFrame, List[str]]:
    msg = ('``impute_method`` must be one of: ["iterative", "knn", None]')
    assert impute_method in ['iterative', 'knn', None], msg

    if impute_method is None:
        df = df.dropna()
    else:
        df = df[pd.notnull(df[label_y])]
    labels_x = _get_labels_x(group_feats, cols=df.columns)

    df_train = df[df['train_test'] == 'train']
    df_test = df[df['train_test'] == 'test']

    # If number of cols are different, then remove from both and update labels_x
    cols_nan_train = df_train.columns[df_train.isnull().all(0)]  # gets columns with all nan
    cols_nan_test = df_test.columns[df_test.isnull().all(0)]
    if len(cols_nan_train) > 0 or len(cols_nan_test) > 0:
        df.drop(list(cols_nan_train) + list(cols_nan_test), axis='columns', inplace=True)
        df_train = df[df['train_test'] == 'train']
        df_test = df[df['train_test'] == 'test']
        labels_x = _get_labels_x(group_feats, cols=df.columns)

    X_train = df_train[labels_x].values
    X_test = df_test[labels_x].values
    y_train = df_train[label_y].values
    y_test = df_test[label_y].values

    X_train = _impute_missing_data(X_train, random_seed, method=impute_method)
    X_test = _impute_missing_data(X_test, random_seed, method=impute_method)

    msg = ('There is a different number of columns in <X_train> than in '
           '<X_test>.')
    assert X_train.shape[1] == X_test.shape[1], msg

    return X_train, X_test, y_train, y_test, df, labels_x




