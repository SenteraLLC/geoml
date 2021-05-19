# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:24:40 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from geoml import FeatureSelection


class Training(FeatureSelection):
    '''
    ``Training`` inherits from an instance of ``FeatureSelection`` (which
    inherits from ``FeatureData``), and consists of functions to carry out the
    hyperparameter tuning and chooses the most suitable hyperparameters for
    each unique number of features. Testing is then performed using the chosen
    hyperparameters and results recorded, then each estimator (i.e., for each
    number of features) is fit using the full dataset (i.e., train and test
    sets), being sure to use the hyperparameters and features selected from
    cross validation. After ``Training.train()`` is executed, each trained
    estimator is stored in ``Training.df_test`` under the "regressor" column.
    The full set of estimators (i.e., for all feature selection combinations,
    with potential duplicate estimators for the same number of features) is
    stored in ``Training.df_test_full``. These estimators are fully trained
    and cross validated, and can be safely distributed to predict new
    observations. Care must be taken to ensure information about input features
    is tracked (not only the number of features, but specifications) so new
    data can be processed to be ingested by the estimator to make new
    predictions.
    '''
    __allowed_params = (
        'base_data_dir', 'regressor', 'regressor_params', 'param_grid',
        'n_jobs_tune', 'scoring', 'refit', 'rank_scoring', 'print_out_train',
        'base_dir_results')

    def __init__(self, **kwargs):
        '''
        '''
        super(Training, self).__init__(**kwargs)

        self.fs_find_params(**kwargs)

        # Training defaults
        self.regressor = None
        self.regressor_name = None
        self.regressor_params = {}
        self.regressor_key = None
        self.param_grid = {'alpha': list(np.logspace(-4, 0, 5))}
        self.n_jobs_tune = 1
        self.scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2')
        self.refit = self.scoring[0]
        self.rank_scoring = self.scoring[0]
        self.print_out_train = False

        self._set_params_from_kwargs_train(**kwargs)
        self._set_attributes_train()
        # self._set_regressor()


    def _set_params_from_dict_train(self, config_dict):
        '''
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        '''
        if config_dict is not None and 'Training' in config_dict:
            params_fd = config_dict['Training']
        elif config_dict is not None and 'Training' not in config_dict:
            params_fd = config_dict
        else:  # config_dict is None
            return
        for k, v in params_fd.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, v)

    def _set_params_from_kwargs_train(self, **kwargs):
        '''
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'config_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``Training`` more explicitly.
        '''
        if 'config_dict' in kwargs:
            self._set_params_from_dict_train(kwargs.get('config_dict'))
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)
                    # print(k)
                    # print(v)
                    # print('')
        # Now, after all kwargs are set, set the regressor
        if 'regressor' in kwargs:
            self._set_regressor()
        elif kwargs.get('config_dict') is None:
            pass
        elif 'config_dict' in kwargs and 'Training' in kwargs.get('config_dict'):
            params_fd = kwargs.get('config_dict')['Training']
            # print(params_fd.keys())
            if 'regressor' in kwargs.get('config_dict')['Training'].keys():
                self._set_regressor()
        #     ('config_dict' in kwargs and 'regressor' in kwargs.get('config_dict')['Training'].items())):
        #     print('setting regressor')
            # if k == 'regressor':
        # self._set_regressor()

    def _set_attributes_train(self):
        '''
        Sets any class attribute to ``None`` that will be created in one of the
        user functions from the ``feature_selection`` class
        '''
        self.df_tune = None
        self.df_test_full = None
        self.df_pred = None
        self.df_pred_full = None
        self.df_test = None

    def _set_regressor(self):
        '''
        Applies tuning parameters to the sklearn model(s) listed in
        <params_dict>. If the <random_seed> was not included in
        <model_tun_params> (or if there is a discrepancy), the model's random
        state is set/reset to avoid any discrepancy.
        '''
        if self.regressor_params is None:
            self.regressor_params = {}
        if 'regressor' in self.regressor.get_params().keys():
            self.regressor_key = 'regressor__'
            self.regressor_params = self._param_grid_add_key(
                self.regressor_params, self.regressor_key)
            self.regressor.set_params(**self.regressor_params)
            # check if setting only a single parameter inside "regressor" keeps
            # all the old parameters!
            # Yes, as long as it is accessed via regressor.regressor instead of
            # reseting regressor (via regressor.set_params(regressor=Lasso()))
            # estimator_level1 = type(my_train.regressor).__name__
            # estimator_level2 = type(my_train.regressor.regressor).__name__
            self.regressor_name = type(self.regressor.regressor).__name__
            try:
                self.regressor.set_params(**{self.regressor_key + 'random_state': self.random_seed})
            except ValueError:
                print('Invalid parameter <random_state> for estimator, thus '
                      '<random_state> cannot be set.\n')
        else:
            self.regressor.set_params(**self.regressor_params)
            self.regressor_name = type(self.regressor).__name__
            try:
                self.regressor.set_params(**{'random_state': self.random_seed})
            except ValueError:
                print('Invalid parameter <random_state> for estimator, thus '
                      '<random_state> cannot be set.\n')

    def _param_grid_add_key(self, param_grid_dict, key='regressor__'):
        '''
        Define tuning parameter grids for pipeline or transformed regressor
        key.

        Parameters:
            param_grid_dict (``dict``): The parameter dictionary.
            key (``str``, optional): String to prepend to the ``sklearn`` parameter
                keys; should either be "transformedtargetregressor__regressor__"
                for pipe key or "regressor__" for transformer key (default:
                "regressor__").

        Returns:
            param_grid_mod: A modified version of <param_grid_dict>
        '''
        param_grid_mod = {f'{key}{k}': v for k, v in param_grid_dict.items()}
        return param_grid_mod

    def _tune_grid_search(self):
        '''
        Performs the CPU intensive hyperparameter tuning via GridSearchCV

        Returns:
            df_tune (``pd.DataFrame``): Results of ``GridSearchCV()``.
        '''
        if self.n_jobs_tune > 0:
            pre_dispatch = int(self.n_jobs_tune*2)
        if 'regressor' in self.regressor.get_params().keys():
            msg = ('The <regressor> estimator appears to be a nested object '
                   '(such as a pipeline or TransformedTargetRegressor). Thus,'
                   '<regrssor_key> must be properly set via '
                   '``_set_regressor()``.')
            assert self.regressor_key == 'regressor__', msg
        param_grid = self._param_grid_add_key(self.param_grid,
                                              self.regressor_key)

        kwargs_grid_search = {'estimator': self.regressor,
                              'param_grid': param_grid,
                              'scoring': self.scoring,
                              'n_jobs': self.n_jobs_tune,
                              'pre_dispatch': pre_dispatch,
                              # 'cv': self.kfold_repeated_stratified(),
                              'cv': self.get_tuning_splitter(),
                              'refit': self.refit,
                              'return_train_score': True}
        clf = GridSearchCV(**kwargs_grid_search, verbose=0)

        try:
            clf.fit(self.X_train_select, self.y_train)
            df_tune = pd.DataFrame(clf.cv_results_)
            return df_tune
        except ValueError as e:
            print('Estimator was unable to fit due to "{0}"'.format(e))
            return None

    def _get_df_tune_cols(self):
        '''
        Gets column names for tuning dataframe
        '''
        cols = ['uid', 'model_fs', 'feat_n', 'feats_x_select', 'rank_x_select',
                'regressor_name', 'regressor', 'params_regressor',
                'params_tuning']
        prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
        for obj_str in self.scoring:
            for prefix in prefixes:
                col = prefix + obj_str
                cols.extend([col])
        return cols

    def _get_df_test_cols(self):
        '''
        Gets column names for test dataframe
        '''
        cols = ['uid', 'model_fs', 'feat_n', 'feats_x_select', 'rank_x_select',
                'regressor_name', 'regressor', 'params_regressor']
        prefixes = ['train_', 'test_']
        scoring = ['neg_mae', 'neg_rmse', 'r2']
        for obj_str in scoring:
            for prefix in prefixes:
                col = prefix + obj_str
                cols.extend([col])
        return cols

    def _get_tune_results(self, df, rank=1):
        '''
        Retrieves all training and validation scores for a given <rank>. The
        first scoring string (from ``self.scoring``) is used to

        Parameters:
            df (``pd.DataFrame``): Dataframe containing results from
                ``_tune_grid_search``.
            rank (``int``): The rank to retrieve values for (1 is highest rank).
        '''
        data = [np.nan, self.model_fs_name, len(self.labels_x_select),
                self.labels_x_select, self.rank_x_select,
                self.regressor_name, deepcopy(self.regressor)]
        if not isinstance(df, pd.DataFrame):
            data.extend([np.nan] * (len(self._get_df_tune_cols()) - len(data)))
            df_tune1 = pd.DataFrame(
                data=[data], columns=self._get_df_tune_cols())
            return df_tune1
        if self.rank_scoring not in self.scoring or self.rank_scoring is None:
            self.rank_scoring = self.scoring[0]
        rank_scoring = 'rank_test_' + self.rank_scoring

        params_tuning = df[df[rank_scoring] == rank]['params'].values[0]
        self.regressor.set_params(**params_tuning)
        # params_all = self.regressor.regressor.get_params()
        params_all = self.regressor.get_params()
        data.extend([params_all, params_tuning])
        # plist = [params_all]
        # print(params_tuning)
        # # print(params_all['regressor'])
        # print(self.regressor.regressor)
        # self.df = df
        for scoring in self.scoring:
            score_train_s = 'mean_train_' + scoring
            std_train_s = 'std_train_' + scoring
            score_test_s = 'mean_test_' + scoring
            std_test_s = 'std_test_' + scoring
            score_train = df[df[rank_scoring] == rank][score_train_s].values[0]
            std_train = df[df[rank_scoring] == rank][std_train_s].values[0]
            score_val = df[df[rank_scoring] == rank][score_test_s].values[0]
            std_val = df[df[rank_scoring] == rank][std_test_s].values[0]
            data.extend([score_train, std_train, score_val, std_val])
            # print(data)

        df_tune1 = pd.DataFrame(data=[data], columns=self._get_df_tune_cols())
        return df_tune1

    # def _prep_pred_dfs(df_test, feat_n_list, y_label='nup_kgha'):
    #     cols_scores = ['feat_n', 'feats', 'score_train_mae', 'score_test_mae',
    #                    'score_train_rmse', 'score_test_rmse',
    #                    'score_train_r2', 'score_test_r2']

    #     cols_meta = ['study', 'date', 'plot_id', 'trt', 'rate_n_pp_kgha',
    #                  'rate_n_sd_plan_kgha', 'rate_n_total_kgha', 'growth_stage',
    #     cols = list(df_y.columns)
    #     cols.remove('value')
    #     cols.extend(['value_obs', 'value_pred'])

    #     feat_n_list = list(my_train.df_tune['feat_n'])
    #     cols_preds = cols_meta + feat_n_list
    #     df_pred = pd.DataFrame(columns=cols_preds)
    #     df_pred[cols_meta] = df_test[cols_meta]
    #     df_score = pd.DataFrame(columns=cols_scores)
    #     return df_pred, df_score

    def _error(self, train_or_test='train'):
        '''
        Returns the MAE, RMSE, MSLE, and R2 for a fit model
        '''
        if train_or_test == 'train':
            X = self.X_train_select
            y = self.y_train
        elif train_or_test == 'test':
            X = self.X_test_select
            y = self.y_test
        y_pred = self.regressor.predict(X)
        # sns.scatterplot(x=my_train.y_test, y=y_pred)


        neg_mae = -mean_absolute_error(y, y_pred)
        neg_rmse = -np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return y_pred, neg_mae, neg_rmse, r2

    def _fit_all_data(self):
        '''
        Fits ``Training.regressor`` using the full dataset (i.e., training and
        test dataset).

        Cross validation is required to be sure our model is not overfit (i.e,
        that the model is not too complex); after hyperparameter tuning and
        testing, it is fine to train the model with the full dataset as long
        as the hyperparameters are set according to the results from the
        cross-validation.

        Caution: there should not be any feature selection, tuning,
        optimization, etc. after this function is executed.
        '''
        X = np.concatenate((self.X_train_select, self.X_test_select))
        y = np.concatenate((self.y_train, self.y_test))
        self.regressor.fit(X, y)
        # print(self.regressor.score(X, y))

    def _get_test_results(self, df):
        '''
        Trains the model for "current" tuning scenario and computes the
        train and test errors. The train errors in df_train are different than
        that of the train errors in df_tune because tuning uses k-fold cross-
        validation of the training set, whereas df_train uses the full training
        set.

        Parameters:
            df (``pd.DataFrame``):
        '''
        data = [df.iloc[0]['uid'], self.model_fs_name, len(self.labels_x_select),
                self.labels_x_select, self.rank_x_select,
                self.regressor_name, deepcopy(self.regressor)]
        if pd.isnull(df.iloc[0]['params_regressor']):
            data.extend([np.nan] * (len(self._get_df_test_cols()) - len(data)))
            df_test_full1 = pd.DataFrame(
                data=[data], index=[df.index[0]], columns=self._get_df_test_cols())
            return df_test_full1, None, None

        msg = ('<params_regressor> are not equal. (this is a bug)')
        assert self.regressor.get_params() == df['params_regressor'].values[0], msg

        self.regressor.fit(self.X_train_select, self.y_train)

        y_pred_train, train_neg_mae, train_neg_rmse, train_r2 = self._error(train_or_test='train')
        y_pred_test, test_neg_mae, test_neg_rmse, test_r2 = self._error(train_or_test='test')

        self._fit_all_data()  # Fit using both train and test data
        data[-1] = deepcopy(self.regressor)
        data.extend([self.regressor.get_params()])
        data.extend([train_neg_mae, test_neg_mae, train_neg_rmse, test_neg_rmse,
                     train_r2, test_r2])
        df_test_full1 = pd.DataFrame([data], index=[df.index[0]], columns=self._get_df_test_cols())
        return df_test_full1, y_pred_test, y_pred_train


        # estimator = df_tune_filtered2.iloc[0]['regressor']
        # estimator1 = estimator.replace('\n', '')

    # def _execute_tuning(X, y, model_list, param_grid_dict,
    #                    alpha, standardize, scoring, scoring_refit,
    #                    max_iter, random_seed, key, df_train, n_splits, n_repeats,
    #                    print_results=False):
    #     '''
    #     Execute model tuning, saving gridsearch hyperparameters for each number
    #     of features.
    #     '''
    #     df_tune = None
    #     for idx in self.df_fs_params.index:
    #         X_train_select, X_test_select = self.fs_get_X_select(idx)
    #         print('Number of features: {0}'.format(len(feats)))

    #         param_grid_dict = param_grid_add_key(param_grid_dict, key)

    #         df_tune_grid = self._tune_grid_search()
    #         df_tune_rank = self._get_tune_results(df_tune_grid, rank=1)
    #         if df_tune is None:
    #             df_tune = df_tune_rank.copy()
    #         else:
    #             df_tune.append(df_tune_rank)

    #         if print_results is True:
    #             print('{0}:'.format(self.regressor_name))
    #             print('R2: {0:.3f}\n'.format(df_temp['score_val_r2'].values[0]))
    #     df_tune = df_tune.sort_values('feat_n').reset_index(drop=True)
    #     self.df_tune = df_tune


    # def _execute_tuning_pp(
    #         logspace_list, X1, y1, model_list, param_grid_dict, standardize,
    #         scoring, scoring_refit, max_iter, random_seed, key, df_train,
    #         n_splits, n_repeats, df_tune_all_list):
    #     '''
    #     Actual execution of hyperparameter tuning via multi-core processing
    #     '''
    #     # chunks = chunk_by_n(reversed(logspace_list))
    #     chunk_size = int(len(logspace_list) / (os.cpu_count()*2)) + 1
    #     with ProcessPoolExecutor() as executor:
    #         # for alpha, df_tune_feat_list in zip(reversed(logspace_list), executor.map(execute_tuning, it.repeat(X1), it.repeat(y1), it.repeat(model_list), it.repeat(param_grid_dict), reversed(logspace_list),
    #         #                                                                           it.repeat(standardize), it.repeat(scoring), it.repeat(scoring_refit), it.repeat(max_iter), it.repeat(random_seed),
    #         #                                                                           it.repeat(key), it.repeat(df_train), it.repeat(n_splits), it.repeat(n_repeats))):
    #         for df_tune_feat_list in executor.map(execute_tuning, it.repeat(X1), it.repeat(y1), it.repeat(model_list), it.repeat(param_grid_dict), reversed(logspace_list),
    #                                               it.repeat(standardize), it.repeat(scoring), it.repeat(scoring_refit), it.repeat(max_iter), it.repeat(random_seed),
    #                                               it.repeat(key), it.repeat(df_train), it.repeat(n_splits), it.repeat(n_repeats), chunksize=chunk_size):
    #                 # chunksize=chunk_size))

    #             # print('df: {0}'.format(df_tune_feat_list))

    #             # print('type: {0}'.format(type(df_tune_feat_list[0])))
    #             df_tune_all_list = append_tuning_results(df_tune_all_list, df_tune_feat_list)
    #     return df_tune_all_list

    # def _set_df_pred_idx(self):
    #     df = self.df_test
    #     idx_full = self.df_pred.columns.get_level_values(level=0)
    #     idx_filtered = []
    #     for i in idx_full:
    #         # print(i)
    #         if i in self.df_y.columns:
    #             idx_filtered.append(i)
    #         elif i in list(df['index_full']):
    #             idx_filtered.append(df[df['index_full'] == i].index[0])
    #         else:
    #             idx_filtered.append(np.nan)  # keep -1
    #     self.df_pred.columns = pd.MultiIndex.from_arrays([idx_full, idx_filtered], names=('full', 'filtered'))

    def _filter_test_results(self, scoring='test_neg_mae'):
        '''
        Remove dupilate number of features (keep only lowest error)

        Parameters:
            scoring (``str``): If there are multiple scenarios with the same
                number of features, <scoring> corresponds to the <df_test_full>
                column that will be used to determine which scenario to keep
                (keeps the highest).
        '''
        msg = ('<df_tune> must be populated with parameters and test scroes. '
               'Have you executed ``tune_and_train()`` yet?')
        assert isinstance(self.df_tune, pd.DataFrame), msg

        df = self.df_test_full
        idx = df.groupby(['regressor_name', 'feat_n'])[scoring].transform(max) == df[scoring]
        idx_feat1 = df['feat_n'].searchsorted(1, side='left')
        if np.isnan(df.iloc[idx_feat1][scoring]):
            idx.iloc[idx_feat1] = True

        df_filtered = self.df_test_full[idx].drop_duplicates(['regressor_name', 'feat_n'])
        # df_filtered.reset_index(level=df_filtered.index.names, inplace=True)
        # df_filtered = df_filtered.rename(columns={'index': 'index_full'})
        df_filtered.reset_index(drop=True, inplace=True)
        self.df_test = df_filtered
        # self._set_df_pred_idx()

    def _get_uid(self, idx):
        if self.df_test_full is None:
            idx_max = 0
        else:
            idx_max = self.df_test_full.index.max() + 1
        return int(idx_max + idx)

    def fit(self, **kwargs):
        '''
        Perform tuning for each unique scenario from ``FeatureSelection``
        (i.e., for each row in <df_fs_params>).

        Example:
            >>> from geoml import Training
            >>> from geoml.tests import config

            >>> my_train = Training(config_dict=config.config_dict)
            >>> my_train.train()
        '''
        print('Executing hyperparameter tuning and estimator training...')
        self._set_params_from_kwargs_train(**kwargs)

        df_tune = self.df_tune
        df_test_full = self.df_test_full
        df_pred = self.df_pred
        df_pred_full = self.df_pred_full
        print_splitter_info = self.print_splitter_info
        _ = self.get_tuning_splitter(print_splitter_info=True)  # prints the number of obs
        self.print_splitter_info = print_splitter_info
        for idx in self.df_fs_params.index:
            X_train_select, X_test_select = self.fs_get_X_select(idx)
            n_feats = len(self.df_fs_params.loc[idx]['feats_x_select'])
            if self.print_out_train == True:
                print('Number of features: {0}'.format(n_feats))
            df_tune_grid = self._tune_grid_search()
            df_tune_rank = self._get_tune_results(df_tune_grid, rank=1)
            uid = self._get_uid(idx)
            df_tune_rank.loc[0, 'uid'] = uid
            df_tune_rank = df_tune_rank.rename(index={0: uid})
            if df_tune is None:
                df_tune = df_tune_rank.copy()
            else:
                df_tune = df_tune.append(df_tune_rank)
            df_test_full1, y_pred_test, y_pred_train = self._get_test_results(df_tune_rank)
            if df_test_full is None:
                df_test_full = df_test_full1.copy()
            else:
                df_test_full = df_test_full.append(df_test_full1)

            if self.print_out_train is True:
                print('{0}:'.format(self.regressor_name))
                print('R2: {0:.3f}\n'.format(df_tune_rank['score_val_r2'].values[0]))

            if df_pred is None:
                df_pred = self.df_y[self.df_y['train_test'] == 'test'].copy()
                df_pred_full = self.df_y.copy()

            if y_pred_test is not None:  # have to store y_pred while we have it
                df_pred[uid] = y_pred_test
                df_pred_full[uid] = np.concatenate([y_pred_train, y_pred_test])
                # df_pred[(uid, np.nan)] = y_pred

        self.df_tune = df_tune
        self.df_test_full = df_test_full
        self.df_pred = df_pred
        self.df_pred_full = df_pred_full
        self._filter_test_results(scoring='test_neg_mae')
