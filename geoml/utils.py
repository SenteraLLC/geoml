import os

import pandas as pd     # type: ignore
import geopandas as gpd # type: ignore

from typing import Set, Any, Optional, Union


AnyDataFrame = Union[pd.DataFrame, gpd.GeoDataFrame]


def check_col_names(df            : AnyDataFrame,
                    cols_required : Set[str]
                   ):
    '''
    Checks to be sure all of the required columns are in <df>.

    Raises:
        AttributeError if <df> is missing any of the required columns.
    '''
    if not all(i in df.columns for i in cols_required):
        cols_missing = list(sorted(set(cols_required) - set(df.columns)))
        raise AttributeError('<df> is missing the following required '
                             'columns: {0}.'.format(cols_missing))

def write_to_readme(msg         : str,
                    dir_results : Optional[str],
                    msi_run_id  : Optional[int] = None,
                    row         : Any = None
                   ) -> None:
    '''
    Writes ``msg`` to the README.txt file
    '''
    # Note if I get here to modify foler_name or use msi_run_id:
    # Try to keep msi-run_id out of this class; instead, make all folder
    # names, etc. be reflected in the dir_results variable (?)
    if dir_results is None:
        print('<dir_results> must be set to create README file.')
        return
    else:
        with open(os.path.join(dir_results, 'README.txt'), 'a') as f:
            f.write(str(msg) + '\n')
