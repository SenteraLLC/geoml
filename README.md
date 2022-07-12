# GeoML

API to retrieve training data, create X matrix, and perform feature selection, hyperparameter tuning, training, and model testing.

## Setup and Installation (for development)
1) [Set up SSH](https://github.com/SenteraLLC/install-instructions/blob/master/ssh_setup.md)
2) Install [pyenv](https://github.com/SenteraLLC/install-instructions/blob/master/pyenv.md) and [poetry](https://python-poetry.org/docs/#installation).
3) Install package
``` bash
git clone git@github.com:SenteraLLC/geoml.git
cd geoml
pyenv install $(cat .python-version)
poetry config virtualenvs.in-project true
poetry env use $(cat .python-version)
poetry install
```
4) Set up `pre-commit` to ensure all commits to adhere to **black** and **PEP8** style conventions.
``` bash
poetry run pre-commit install
```

## Setup and Installation (used as a library)
If using `geoml` as a dependency in your script, simply add it to the `pyproject.toml` in your project repo.

<h5 a><strong><code>pyproject.toml</code></strong></h5>

``` toml
[tool.poetry.dependencies]
geoml = { git = "https://github.com/SenteraLLC/geoml.git", branch = "main"}
```

Install `geoml` and all its dependencies via `poetry install`.

``` console
poetry install
```

## Environment Variables

`geoml` requires a direct connection to a PostgreSQL database to read and write data. It is recommended to store database connection credentials as environment variables. One way to do this is to create an ``.env`` (and/or and ``.envrc``) file within your project directory that includes the necessary enviroment variables.

*Note `.env` sets environment variables for IPyKernel/Jupyter, and `.envrc` sets environment variables for normal python console.*

<h5 a><strong><code>.env</code></strong></h5>

``` bash
DB_NAME=db_name
DB_HOST=localhost
DB_USER=analytics_user
DB_PASSWORD=secretpassword
DB_PORT=5432

SSH_HOST=bastion-lt-lb-<HOST_ID>.elb.us-east-1.amazonaws.com
SSH_USER=<ssh_user>
SSH_PRIVATE_KEY=<path/to/.ssh/id_rsa>
SSH_DB_HOST=<analytics.<DB_HOST_ID>.us-east-1.rds.amazonaws.com>
```

<h5 a><strong><code>.envrc</code></strong></h5>

``` bash
export DB_NAME=db_name
export DB_HOST=localhost
export DB_USER=analytics_user
export DB_PASSWORD=secretpassword
export DB_PORT=5432

export SSH_HOST=bastion-lt-lb-<HOST_ID>.elb.us-east-1.amazonaws.com
export SSH_USER=<ssh_user>
export SSH_PRIVATE_KEY=<path/to/.ssh/id_rsa>
export SSH_DB_HOST=<analytics.<DB_HOST_ID>.us-east-1.rds.amazonaws.com>
```

## Usage Example

### Step 1
Establish a connection to ``DBHandler`` for owner's schema. Note that this assumes all the necessary tables have been loaded into the owner's database schema already.

<h5 a><strong><code>connect_to_db.py</code></strong></h5>

``` python
from os import getenv
from db import DBHandler


owner = "css-farms-pasco"
db_name = getenv("DB_NAME")
db_host = getenv("DB_HOST")
db_user = getenv("DB_USER")
password = getenv("DB_PASSWORD")
db_schema = owner.replace("-", "_")
db_port = db_utils.ssh_bind_port(
    SSH_HOST=getenv("SSH_HOST"),
    SSH_USER=getenv("SSH_USER"),
    ssh_private_key=getenv("SSH_PRIVATE_KEY"),
    host=getenv("SSH_DB_HOST"),
)

db = DBHandler(
    database=db_name,
    host=db_host,
    user=db_user,
    password=password,
    port=db_port,
    schema=db_schema,
)
```

### Step 2
Edit configuration settings to train to model as you wish:

<h5 a><strong><code>edit_config.py</code></strong></h5>

``` python
from copy import deepcopy
from datetime import datetime
import json

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import PowerTransformer


with open("geoml/config.json") as f:
    config_dict = json.load(f)

config_dict["FeatureData"]["date_train"] = datetime.now().date()
config_dict["FeatureData"]["cv_method"] = train_test_split
config_dict["FeatureData"]["cv_method_kwargs"]["stratify"] = 'df[["owner", "year"]]'
config_dict["FeatureData"]["cv_method_tune"] = RepeatedStratifiedKFold

config_dict["FeatureSelection"]["model_fs"] = Lasso()
config_dict['FeatureSelection']['n_feats'] = 12

config_dict["Training"]["regressor"] = TransformedTargetRegressor(
    regressor=Lasso(),
    transformer=PowerTransformer(copy=True, method="yeo-johnson", standardize=True),
)
config_dict["Training"]["param_grid"]["alpha"] = list(np.logspace(-4, 0, 5))
config_dict["Training"]["scoring"] = (
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "r2",
)

config_dict["Predict"]["date_predict"] = datetime.now().date()
config_dict["Predict"]["dir_out_pred"] = "/mnt/c/Users/Tyler/Downloads"
```


### Step 3
Create `Training` instance (which loads data and performs feature selection) and train the model:

<h5 a><strong><code>train_geoml.py</code></strong></h5>

``` python
from geoml import Training


train = Training(config_dict=config_dict)
train.fit()
```

### Step 4
Grab an estimator to make predictions with:

<h5 a><strong><code>predict_geoml_part1.py</code></strong></h5>

``` python
from geoml import Predict


date_predict = date_master.date()
year = date_predict.year
config_dict['Predict']['train'] = train
config_dict['Predict']['primary_keys_pred'] = {
    'owner': 'css-farms-pasco',
    'farm': 'adams',
    'field_id': 'adams-e',
    'year': 2022
}

estimator = train.df_test.loc[
    train.df_test[train.df_test['feat_n'] == config_dict['FeatureSelection']['n_feats']].index,
    'regressor'
].item()
feats_x_select = train.df_test.loc[
    train.df_test[train.df_test['feat_n'] == config_dict['FeatureSelection']['n_feats']].index,
    'feats_x_select'
].item()

predict = Predict(
    estimator=estimator,
    feats_x_select=feats_x_select,
    config_dict=config_dict
)
```

### Step 5
Make a prediction for each field and save output as a geotiff raster:

<h5 a><strong><code>predict_geoml_part2.py</code></strong></h5>

``` python
from os import chdir
import satellite.utils as sat_utils

chdir(preds_dir)  # predict_functions helper functions are located here
field_bounds = db.get_table_df('field_bounds', year=year)

dir_out_pred = config_dict["Predict"]["dir_out_pred"]
date_predict = config_dict["Predict"]["date_predict"]
for idx, row in field_bounds.iterrows():
    pkeys = {
        "owner": row["owner"],
        "farm": row["farm"],
        "field_id": row["field_id"],
        "year": row["year"],
    }
    array_pred, profile = predict.predict(
        primary_keys_pred=pkeys
    )
    name_out = "petiole-no3-ppm_{0}_{1}_{2}_{3}_raw.tif".format(
        date_predict.strftime("%Y-%m-%d"),
        row["owner"],
        row["farm"],
        row["field_id"]
    )
    fname_out = os.path.join(
        dir_out_pred, "by_date",
        date_predict.strftime("%Y-%m-%d"),
        name_out
    )
    sat_utils.save_image(
        array_pred, profile, fname_out, driver="Gtiff", keep_xml=False
    )
```

Here is an example raster loaded into QGIS. We can see the spatial variation in predicted petiole nitrate:
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
