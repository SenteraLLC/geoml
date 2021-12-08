from .config import ConfigGroup, FeatureDataConfig, FeatureSelectionConfig, TrainingConfig, PredictConfig

from datetime import datetime
from enum import Enum
from functools import partial

from typing import Dict, Any, Callable, Type
from typing import cast

# Utility

def parseEnum(enum         : Type[Enum],
              enum_to_type : Dict[Enum, Any],
              field_name   : str,
              field_value  : str,
             ) -> Any:
  allowed = enum.__members__.keys()
  if field_value not in allowed:
    raise Exception(f"Unknown value '{field_value}' for '{field_name}'")
  e = enum[field_value]

  enum_value = enum_to_type[e]

  return enum_value


def checkConfigKeys(config_type : Type[ConfigGroup],
                    candidate   : Dict[str, Any],
                   ) -> None:
  config_keys = set(config_type.__annotations__.keys())
  candidate_keys = set(candidate.keys())

  extraneous = candidate_keys - config_keys

  if len(extraneous) > 0:
    e = str(list(extraneous))
    raise Exception(f"Extraneous config arguments: {e}")

  missing = config_keys - candidate_keys
  if len(missing) > 0:
    e = str(list(missing))
    raise Exception(f"Missing config arguments: {e}")



# TODO: This probably needs to be more flexible
parseDate = lambda _, date_as_string : datetime.strptime(date_as_string, "%d/%m/%Y").date()


### Feature Data ###

from sklearn.model_selection import LeavePGroupsOut # type: ignore

class CVMethod(Enum):
    LeavePGroupsOut = 1

cv_method : Dict[CVMethod, Any] = {
    CVMethod.LeavePGroupsOut : LeavePGroupsOut,
}

parseCVMethod = partial(parseEnum, CVMethod, cv_method)


from sklearn.model_selection import RepeatedStratifiedKFold

class CVMethodTune(Enum):
    RepeatedStratifiedKFold = 1

cv_method_tune : Dict[CVMethodTune, Any] = {
    CVMethodTune.RepeatedStratifiedKFold : RepeatedStratifiedKFold,
}

parseCVMethodTune = partial(parseEnum, CVMethodTune, cv_method_tune)

feature_data_keys : Dict[str, Callable[[str, str], Any]] = {
  "cv_method"      : parseCVMethod,
  "cv_method_tune" : parseCVMethodTune,
  "date_train"     : parseDate,
}

def parseFeatureData(to_parse : Dict[str, Any]) -> FeatureDataConfig:
  checkConfigKeys(FeatureDataConfig, to_parse)  # type: ignore

  feature_data = to_parse.copy()
  for field_name, parse_function in feature_data_keys.items():
      try:
          field_value = feature_data[field_name]
          feature_data[field_name] = parse_function(field_name, field_value)
      except KeyError as e:
          pass

  return cast(FeatureDataConfig, feature_data)



### Feature Selection ###

class ModelFS(Enum):
    Lasso = 1

from sklearn.linear_model import Lasso  # type: ignore

model_fs : Dict[ModelFS, Any] = {
    ModelFS.Lasso : Lasso(),
}

parseModelFS = partial(parseEnum, ModelFS, model_fs)

feature_selection_keys : Dict[str, Callable[[str, str], Any]] = {
    "model_fs" : parseModelFS,
}

def parseFeatureSelection(to_parse : Dict[str, Any]) -> FeatureSelectionConfig:
  checkConfigKeys(FeatureSelectionConfig, to_parse)  # type: ignore

  feature_selection = to_parse.copy()
  for field_name, parse_function in feature_selection_keys.items():
      try:
          field_value = feature_selection[field_name]
          feature_selection[field_name] = parse_function(field_name, field_value)
      except KeyError:
          pass

  return cast(FeatureSelectionConfig, feature_selection)


### Training ###

class Regressor(Enum):
    TransformedTargetRegressor = 1

from sklearn.compose import TransformedTargetRegressor # type: ignore
from sklearn.preprocessing import PowerTransformer # type: ignore

regressor : Dict[Regressor, Any] = {
    Regressor.TransformedTargetRegressor : TransformedTargetRegressor(regressor=Lasso(), transformer=PowerTransformer(copy=True, method="yeo-johnson", standardize=True))
}

parseRegressor = partial(parseEnum, Regressor, regressor)

training_keys : Dict[str, Callable[[str, str], Any]] = {
    "regressor" : parseRegressor,
}

def parseTraining(to_parse : Dict[str, Any]) -> TrainingConfig:
  checkConfigKeys(TrainingConfig, to_parse)  # type: ignore

  training = to_parse.copy()
  for field_name, parse_function in training_keys.items():
      try:
          field_value = training[field_name]
          training[field_name] = parse_function(field_name, field_value)
      except KeyError:
          pass

  return cast(TrainingConfig, training)


### Predict ###

predict_keys : Dict[str, Callable[[str, str], Any]] = {
    "date_predict"     : parseDate,
}

def parsePredict(to_parse : Dict[str, Any]) -> PredictConfig:
  checkConfigKeys(PredictConfig, to_parse)  # type: ignore

  predict = to_parse.copy()
  for field_name, parse_function in predict_keys.items():
      try:
          field_value = predict[field_name]
          predict[field_name] = parse_function(field_name, field_value)
      except KeyError:
          pass

  return cast(PredictConfig, predict)



def parseConfig(to_parse : Dict[str, Any]):
  return {"table"             : to_parse["table"],
          "feature_data"      : parseFeatureData(to_parse["feature_data"]),
          "feature_selection" : parseFeatureSelection(to_parse["feature_selection"]),
          "training"          : parseTraining(to_parse["training"]),
          "predict"           : parsePredict(to_parse["predict"]),
         }








