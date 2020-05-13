# research_tools

API to process and analyze all of the historical data collected from the Rosen lab.

The overall goal is to determine the baseline prediction accuracy we should expect for each response variable (e.g., petiole nitrate, vine N, biomass, etc.) for a given set of input features (e.g., cropscan bands, hyperspectral imagery, weather, etc.).

[![CircleCI](https://circleci.com/gh/insight-sensing/research_tools/tree/dev.svg?style=svg&circle-token=4d961470ddaa2ed3b8a4b81d84d5e0edfb38f840)](https://app.circleci.com/pipelines/github/insight-sensing/research_tools?branch=dev)
[![GitHub issues](https://img.shields.io/github/issues/insight-sensing/research_tools.svg)](https://github.com/insight-sensing/research_tools/issues)
[![Code coverage](https://codecov.io/gh/insight-sensing/research_tools/branch/dev/graph/badge.svg)]()

## Classes
There are multiple classes that work together to perform all the necessary steps for training supervised regression estimators. Here is a brief summary:

### JoinTables
Assists with joining tables that contain training data. In addition to the join, many of the user functions available to add new columns/features to the input DataFrame (or X) that hopefully explain the response variable being predicted.

### FeatureSelection
`FeatureSelection` inherits from `FeatureData`, and carries out all tasks related to feature selection before model tuning, training, and prediction. The information garnered from `FeatureSelection` is quite simply all the parameters required to duplicate a given number of features, as well as its cross-validation results (e.g., features used, ranking, training and validation scores, etc.).

### Training
`Training` inherits from an instance of `FeatureSelection` (which inherits from `FeatureData`), and consists of functions to carry out the hyperparameter tuning and chooses the most suitable hyperparameters for each unique number of features. Testing is then performed using the chosen hyperparameters and results recorded, then each estimator (i.e., for each number of features) is fit using the full dataset (i.e., train and test sets), being sure to use the hyperparameters and features selected from cross validation. After `Training.train()` is executed, each trained estimator is stored in `Training.df_test` under the "regressor" column. The full set of estimators (i.e., for all feature selection combinations, with potential duplicate estimators for the same number of features) is stored in `Training.df_test_full`. These estimators are fully trained and cross validated, and can be safely distributed to predict new observations. Care must be taken to ensure information about input features is tracked (not only the number of features, but specifications) so new data can be preocessed to be ingested by the estimator to make new predictions.

## Setup and Installation
Dependencies are provided for both a production environment (e.g., one that would be deployed on a cloud DB), and for a development or testing environment.

### Production
`conda create -n insight_prod python=3.8`
`conda install -c anaconda scikit-learn`
`conda install -c conda-forge pandas`

### Dev/Test
`conda create -n insight_dev python=3.8`
`conda install -c anaconda scikit-learn`
`conda install -c conda-forge pandas`
`conda install -c conda-forge pytest`

Run tests to be sure everything is installed appropriately:
`pytest research_tools\tests`

## License
TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.
© Insight Sensing Corporation, 2020

This software contains confidential and proprietary information of Insight Sensing Corporation and is protected by copyright, trade secret, and other State and Federal laws. Its receipt or possession does not convey any rights to reproduce, disclose its contents, or to manufacture, use or sell anything it may describe. Reproduction, disclosure, or use without specific written authorization of Insight Sensing Corporation is strictly forbidden.