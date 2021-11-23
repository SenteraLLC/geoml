# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:42:53 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
import json

import numpy as np
import os
import pandas as pd  # type: ignore
import geopandas as gpd # type: ignore
from sqlalchemy import inspect  # type: ignore

from ...db.db import utilities as db_utils
from ...db.db import DBHandler

from .types import DBConfig, TableConfig, AnyDataFrame
# TODO: Move this
from ..utils import check_col_names


from typing import Optional, Union, Dict



def connect_to_db(db_config : DBConfig,
                  password  : Optional[str],
                 ) -> Optional[DBHandler]:
    '''
    Using DBHandler, tries to make a connection to ``tables.db_name``.
    '''
    msg = ('To connect to the DB, all of the following variables must '
           'be passed to ``Tables`` (either directly or via the config '
           'file): [db_name, db_host, db_user, db_schema, db_port]')
    if any(v is None for v in db_config.values()):
        print(msg)
        return None

    # TODO: Throw and handle exception
    # 'Failed to connect to database via DBHanlder.'
    # 'Please check DB credentials.'

    return DBHandler(**db_config)  # TODO: Fix



def load_table_from_db(db : DBHandler, table_name : str) -> Optional[AnyDataFrame]:
    '''
    Loads <table_name> from database via <Table.db>
    '''
    inspector = inspect(db.engine)
    if table_name in inspector.get_table_names(schema=db.db_schema):
        df = db.get_table_df(table_name)
        if len(df) > 0:
            return df
    return None


def read_csv_geojson(fname : str) -> AnyDataFrame:
    '''
    Depending on file extension, will read from either pd or gpd
    '''
    if os.path.splitext(fname)[-1] == '.csv':
        df = pd.read_csv(fname)
    elif os.path.splitext(fname)[-1] == '.geojson':
        df = gpd.read_file(fname)
    else:
        raise TypeError('<fname_sentinel> must be either a .csv or '
                        '.geojson...')
    return df


def load_table_from_file(base_dir_data : str, fname : str) -> Optional[AnyDataFrame]:
    '''
    Loads <table_name> from file.
    '''
    fname_full = os.path.join(base_dir_data, fname)
    if os.path.isfile(fname_full):
        df = read_csv_geojson(fname_full)
        if len(df) > 0:
            return df
    return None


def verify_required(db : DBHandler,
                    table_name : str,
                    table : AnyDataFrame
                   ) -> None:
    msg = ('The following columns are required in "{0}". Missing columns: '
           '"{1}".')
    engine = None
    schema = None
    try:
      engine = db.engine
      schema = db.db_schema
    except AttributeError:
      pass
    cols_required, _ = db_utils.get_cols_nullable(table_name, engine, schema)
    check_col_names(table, cols_required)


def load_table(table_name    : str,
               db            : Optional[DBHandler],
               base_dir_data : Optional[str],
               config        : TableConfig,
              ) -> AnyDataFrame:
  if base_dir_data is not None:
      filename = config["table_names"][table_name]
      df = load_table_from_file(base_dir_data, filename)
      if df is not None:
          df = db_utils.cols_to_datetime(df)
  elif db is not None:
      df = load_table_from_db(db, table_name)
  else:
      raise Exception("No source provided for table data.")

  return df


def populate_field_bounds(field_bounds : AnyDataFrame,
                          df           : AnyDataFrame
                         ) -> AnyDataFrame:
   '''Fills/replaces empty geometry with field_bounds geometry'''
   subset = db_utils.get_primary_keys(df)
   df_geom = df[~df[df.geometry.name].is_empty]
   df_empty = df[df[df.geometry.name].is_empty]
   df_empty.drop(columns=[df_empty.geometry.name], inplace=True)
   df = df_geom.append(df_empty.merge(
                          field_bounds[subset + [field_bounds.geometry.name]],
                          on=subset))
   return df


def load_tables(db     : Optional[DBHandler],
                config : TableConfig,
                base_dir_data : Optional[str]
               ) -> Dict[str, AnyDataFrame]:
    tables : Dict[str, AnyDataFrame] = {}
    for table_name in config["table_names"]:
        tables[table_name] = load_table(table_name, db, base_dir_data, config)
    try:
        field_bounds = tables["field_bounds"]
        for table_name, table in tables.items():
            if table_name in ['as_planted', 'n_applications', 'obs_tissue']:
                tables[table_name] = populate_field_bounds(field_bounds, table)
    except KeyError:
      pass

    try:
        rs_cropscan_res = tables["rs_cropscan_res"]
        subset = db_utils.get_primary_keys(rs_cropscan_res)
        # TODO: the groupby() function removes geometry - add geometry
        # column back in by copying relevant info from rs_cropscan_res

        # in the meantime, chnage to regular DataFrame
        df_cs = rs_cropscan_res.groupby(subset + ['date']).mean().reset_index()
        tables["rs_cropscan_res"] = pd.DataFrame(df_cs)
    except KeyError:
      pass

    return tables


# TODO: Run-time arguments:
# "password"      : Optional[str],  # TODO: Remove
# "db"            : DBHandler,      # TODO: Remove
# "base_dir_data" : str,            # TODO: Remove

def main(base_dir_data     : str,
         table_config_path : str,
         password          : Optional[str],
        ):
  with open(table_config_path) as f:
     config : TableConfig = json.load(f)

     db = connect_to_db(config["database"], password=password)

     tables = load_tables(db, config, base_dir_data)


import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Load a field and all of its input data as JSON')
  parser.add_argument('--base_dir_data', type=str, help='Directory with data', required=True)
  parser.add_argument('--config', type=str, help='Table config', required=True)
  parser.add_argument('--password', type=str, help='Database password')
  args = parser.parse_args()

  main(args.base_dir_data, args.config, args.password)






