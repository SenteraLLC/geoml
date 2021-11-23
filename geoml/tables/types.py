import pandas as pd     # type: ignore
import geopandas as gpd # type: ignore

from typing import TypedDict, Dict, Union, Set


class DBConfig(TypedDict):
    name   : str
    host   : str
    user   : str
    schema : str
    port   : int

class TableConfig(TypedDict):
    database    : DBConfig
    table_names : Dict[str, str]

AnyDataFrame = Union[pd.DataFrame, gpd.DataFrame]


