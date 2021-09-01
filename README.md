# GeoML

API to retrieve training data, create X matrix, and perform feature selection, hyperparameter tuning, training, and model testing.

## Build and test status

### master
[![codecov](https://codecov.io/gh/insight-sensing/geoml/branch/master/graph/badge.svg?token=45FYM8VS7H)](https://codecov.io/gh/insight-sensing/geoml)
[![build](https://circleci.com/gh/insight-sensing/geoml/tree/master.svg?style=svg&circle-token=4d961470ddaa2ed3b8a4b81d84d5e0edfb38f840)](https://app.circleci.com/pipelines/github/insight-sensing/geoml?branch=dev)

### dev
[![codecov](https://codecov.io/gh/insight-sensing/geoml/branch/dev/graph/badge.svg?token=45FYM8VS7H)](https://codecov.io/gh/insight-sensing/geoml)
[![build](https://circleci.com/gh/insight-sensing/geoml/tree/dev.svg?style=svg&circle-token=4d961470ddaa2ed3b8a4b81d84d5e0edfb38f840)](https://app.circleci.com/pipelines/github/insight-sensing/geoml?branch=dev)

## Setup and Installation (Windows)
There is an *environment_test.yml* file that can be used to create the environment and install the dependencies. After cloning from Github, create the environment:

`conda env create -n test_env -f .geoml\requirements\environment_test.yml`

### PyPI
Some packages are not available on `conda` and must be installed from PyPI:
```
pip install postgis
pip install -r .geoml\requirements\dev_pip.txt
pip install -r .geoml\requirements\testing_pip.txt
```

Note: On Windows, the `postgis` dependency must be installed via `pip` since it is not available on conda-forge. Also not that the `find_program` function of `testing.postgresql` should also be modified if using Windows (see [db issue #10](https://github.com/insight-sensing/db/issues/10)).

## Setup and Installation (Linux and MacOS)
There is an *environment.yml* file that can be used to create the environment and install the dependencies. After cloning from Github, create the environment:
```
conda env create -n test_env -f .geoml\requirements\environment_test.yml
conda install -n test_env -c conda-forge postgis
```
### PyPI
Some packages are not available on `conda` and must be installed from PyPI:
```
pip install -r .geoml\requirements\dev_pip.txt
pip install -r .geoml\requirements\testing_pip.txt
```

## Run tests
Run tests to be sure everything is installed appropriately:
`pytest geoml\tests`

## Use

### Step 1:
`GeoML` can be used via either flat files loaded from a local directory or via tables in a database connection. To connect to a database:

```
from copy import deepcopy
from datetime import datetime, date
from db import DBHandler
from geoml import Training
from geoml import Predict
from geoml.tests import config as geoml_config

db_name = 'insight_dev'
db_host = 'localhost'
db_user = 'postgres'
password = 'bnEtP7C$qKFc'
db_schema = 'dev_client'
db_port = 5432
base_dir_google = ('G:/Shared drives/Data')

db = DBHandler(database=db_name, host=db_host, user=db_user,
               password=password, port=db_port, schema=db_schema)

base_dir_google = ('G:/Shared drives/Data')
preds_dir = os.path.join(
    base_dir_google, 'client_data/CSS Farms/preds_{0}'.format(year))
```

### Step 2:
This assumes all the necessary tables have been loaded into `db` already. Next, add configuration settings and train a model:

```
date_master = datetime(2021, 7, 14)

config_dict = deepcopy(geoml_config.config_dict)
config_dict['Tables']['db'] = db
config_dict['Tables']['base_dir_data'] = None
config_dict['FeatureData']['group_feats'] = geoml_config.sentinel_test1
config_dict['FeatureData']['group_feats']['weather_derived'] = ['gdd_c_ptd_cum']
config_dict['FeatureData']['impute_method'] = None
config_dict['FeatureData']['date_train'] = date_master
config_dict['FeatureSelection']['n_feats'] = 6

train = Training(config_dict=config_dict)
train.fit()
```

### Step 3:
Grab an estimator to make prdictions with:
```
date_predict = date_master.date()
year = date_predict.year
config_dict['Predict']['train'] = train
config_dict['Predict']['date_predict'] = date_predict
# config_dict['Predict']['dir_out_pred'] = preds_dir
config_dict['Predict']['dir_out_pred'] = os.path.join(preds_dir, 'tyler_local')
config_dict['Predict']['refit_X_full'] = True
config_dict['Predict']['primary_keys_pred'] = {'owner': 'css-farms-dalhart',
                                               'farm': 'cabrillas',
                                               'field_id': 'c-08',
                                               'year': 2021}  # dummy keys

estimator = train.df_test.loc[
    train.df_test[train.df_test['feat_n'] == config_dict['FeatureSelection']['n_feats']].index,
    'regressor'].item()
feats_x_select = train.df_test.loc[
    train.df_test[train.df_test['feat_n'] == config_dict['FeatureSelection']['n_feats']].index,
    'feats_x_select'].item()

predict = Predict(estimator=estimator, feats_x_select=feats_x_select,
                  config_dict=config_dict)
```

### Step 4:
Make a prediction for each field and save output as a geotiff raster:

```
os.chdir(preds_dir)  # predict_functions helper functions are located here
import predict_functions as p

field_bounds = db.get_table_df('field_bounds', year=year)

for idx, row in field_bounds.iterrows():
    print(row['field_id'])
    try:
        predict = p.predict_and_save_as_raster(row, predict, date_predict)
    except RuntimeError as e1:
        print('\n{0}'.format(e1))
        print('Missing data, so {0} will be skipped.\n'.format(row['field_id']))
        continue
    except AttributeError as e2:
        print('\n{0}'.format(e2))
        print('Missing N application data, so {0} will be skipped.\n'.format(row['field_id']))
        continue
```

By loading the output raster for CSS Farms - Cabrillas - C-24 into QGIS, we can see the spatial variation in predicted petiole nitrate:
![Alt text](outputs/petiole-no3-ppm_2021-07-12_css-farms-dalhart_cabrillas_c-24_raw.png?raw=true "Petiole nitrate prediction from July 12, 2021")

## Classes
There are multiple classes that work together to perform all the necessary steps for training supervised regression estimators. Here is a brief summary:

### Tables
`Tables` loads in all the available tables that might contain data to build the X matrices for training and testing (`X_train` and `X_test`). If a PostgreSQL database is available, this can be connected by passing the appropriate credentials. The public/user functions derive new data feautes from the original data to be added to the X matrices for explaining the response variable being predicted.

### FeatureData
`FeatureData` inherits from `Tables`, and executes the ppropriate joins among tables (according to the `group_feats` variable) to actually construct the X matrices for training and testing (`X_train` and `X_test`).

### FeatureSelection
`FeatureSelection` inherits from `FeatureData`, and carries out all tasks related to feature selection before model tuning, training, and prediction. The information garnered from `FeatureSelection` is quite simply all the parameters required to duplicate a given number of features, as well as its cross-validation results (e.g., features used, ranking, training and validation scores, etc.).

### Training
`Training` inherits from an instance of `FeatureSelection` and consists of functions to carry out the hyperparameter tuning and chooses the most suitable hyperparameters for each unique number of features. Testing is then performed using the chosen hyperparameters and results recorded, then each estimator (i.e., for each number of features) is fit using the full dataset (i.e., train and test sets), being sure to use the hyperparameters and features selected from cross validation. After `Training.train()` is executed, each trained estimator is stored in `Training.df_test` under the "regressor" column. The full set of estimators (i.e., for all feature selection combinations, with potential duplicate estimators for the same number of features) is stored in `Training.df_test_full`. These estimators are fully trained and cross validated, and can be safely distributed to predict new observations. Care must be taken to ensure information about input features is tracked (not only the number of features, but specifications) so new data can be preocessed to be ingested by the estimator to make new predictions. Also note that care must be taken to ensure the cross-validation strategy (indicated via `cv_method`, `cv_method_kwargs`, and `cv_split_kwargs`) is suitable for your application and that it does not inadvertenly lead to model overfitting - for this reason, multiple cross-validation methods are encouraged.

### Prediction
`Predict` inherits from an instance of `Training`, and consists of variables and functions to make predictions on new data with the previously trained models. `Predict` accesses the appropriate data from the connected database to make predictions (e.g., as_planted, weather, sentinel reflectance images, etc.), and a prediction is made for each observation via the features in the feature set.


## License
TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.
© Insight Sensing Corporation, 2020

This software contains confidential and proprietary information of Insight Sensing Corporation and is protected by copyright, trade secret, and other State and Federal laws. Its receipt or possession does not convey any rights to reproduce, disclose its contents, or to manufacture, use or sell anything it may describe. Reproduction, disclosure, or use without specific written authorization of Insight Sensing Corporation is strictly forbidden.