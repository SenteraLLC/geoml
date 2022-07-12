
Current workflow
================

This is the current workflow for predicting potato N using GeoML. I'm following along with the Pasco train script. 

1. The configuration file is created using the ``load_config`` function for a given site. Right now, 
this function does the following things: 

* Loads the "default" configuration parameters as described in ``load_config_defaults()``
* Adds the previously-connected ``db`` object to the configuration file, which is then used for connection throughout the rest of the 
    training process
* Set the response data information (``table_name``, ``value_col``, ``owner``, ``tissue``, ``measure``)
* Change impute method to ``None``
* Set the training regressor to be: 
    >>> TransformedTargetRegressor(regressor=Lasso(),
    >>>     transformer=PowerTransformer(copy=True, method="yeo-johnson", standardize=True),
    >>> )

    and sets the main parameters. 
* Set ``param_grid`` for training.
* Set ``dir_out_pred`` for saving predictions. 
* Set ``date_train`` if given.

