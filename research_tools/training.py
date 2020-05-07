# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:24:40 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from research_tools import FeatureSelection


class Training(FeatureSelection):
    '''
    ``Training`` inherits from an instance of ``FeatureSelection`` (which
    inherits from ``FeatureData``), then includes functions to carry out the
    hyperparameter tuning steps.
    '''
    __allowed_params = (
        'regressor', 'regressor_params', 'param_grid', 'n_jobs_tune',
        'scoring', 'refit', 'rank_scoring', 'print_out_train')

    def __init__(self, **kwargs):
        '''
        '''
        super(Training, self).__init__(**kwargs)
        # self._set_params_from_kwargs_fs(**kwargs)
        # self._set_attributes_fs()
        # self._set_model_fs()
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

        self._set_params_from_kwargs_tune(**kwargs)
        self._set_attributes_tune()
        # self._set_regressor()


    def _set_params_from_dict_tune(self, param_dict):
        '''
        Sets any of the parameters in ``param_dict`` to self as long as they
        are in the ``__allowed_params`` list
        '''
        if param_dict is not None and 'Training' in param_dict:
            params_fd = param_dict['Training']
        elif param_dict is not None and 'Training' not in param_dict:
            params_fd = param_dict
        else:  # param_dict is None
            return
        for k, v in params_fd.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, v)

    def _set_params_from_kwargs_tune(self, **kwargs):
        '''
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'param_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``Training`` more explicitly.
        '''
        if 'param_dict' in kwargs:
            self._set_params_from_dict_tune(kwargs.get('param_dict'))
        if kwargs is not None:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)
                    # print(k)
                    # print(v)
                    # print('')
        # Now, after all kwargs are set, set the regressor
        if 'regressor' in kwargs:
            self._set_regressor()
        elif kwargs.get('param_dict') is None:
            pass
        elif 'param_dict' in kwargs and 'Training' in kwargs.get('param_dict'):
            params_fd = kwargs.get('param_dict')['Training']
            # print(params_fd.keys())
            if 'regressor' in kwargs.get('param_dict')['Training'].keys():
                self._set_regressor()
        #     ('param_dict' in kwargs and 'regressor' in kwargs.get('param_dict')['Training'].items())):
        #     print('setting regressor')
            # if k == 'regressor':
        # self._set_regressor()

    def _set_attributes_tune(self):
        '''
        Sets any class attribute to ``None`` that will be created in one of the
        user functions from the ``feature_selection`` class
        '''
        self.df_tune = None
        self.df_tune_filtered = None
        self.df_test = None
        self.df_test_filtered = None

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
            # estimator_level1 = type(my_tune.regressor).__name__
            # estimator_level2 = type(my_tune.regressor.regressor).__name__
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
                              'cv': self.kfold_repeated_stratified(),
                              'refit': self.refit,
                              'return_train_score': True}
        clf = GridSearchCV(**kwargs_grid_search)
        try:
            clf.fit(self.X_train_select, self.y_train)
            df_tune = pd.DataFrame(clf.cv_results_)
            return df_tune
        except ValueError as e:
            print('Estimator was unable to fit due to {0}'.format(e))
            return None

    # def _get_df_tune_cols(self):
    #     '''
    #     Gets column names for tuning dataframe
    #     '''
    #     cols = ['model_fs', 'feat_n', 'feats_x_select', 'rank_x_select',
    #             'regressor_name', 'regressor', 'params_regressor',
    #             'params_tuning']
    #     prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
    #     for obj_str in self.scoring:
    #         for prefix in prefixes:
    #             col = prefix + obj_str
    #             cols.extend([col])
    #     return cols

    def _get_df_tune_cols(self):
        '''
        Gets column names for tuning dataframe
        '''
        cols = ['model_fs', 'feat_n', 'feats_x_select', 'rank_x_select',
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
        cols = ['model_fs', 'feat_n', 'feats_x_select', 'rank_x_select',
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
        data = [self.model_fs_name, len(self.labels_x_select),
                self.labels_x_select, self.rank_x_select,
                self.regressor_name, self.regressor]
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

    #     feat_n_list = list(my_tune.df_tune['feat_n'])
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
        # sns.scatterplot(x=my_tune.y_test, y=y_pred)


        neg_mae = -mean_absolute_error(y, y_pred)
        neg_rmse = -np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return y_pred, neg_mae, neg_rmse, r2

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
        data = [self.model_fs_name, len(self.labels_x_select),
                self.labels_x_select, self.rank_x_select,
                self.regressor_name, self.regressor]
        if pd.isnull(df['params_regressor'][0]):
            data.extend([np.nan] * (len(self._get_df_test_cols()) - len(data)))
            df_test1 = pd.DataFrame(
                data=[data], columns=self._get_df_test_cols())
            return df_test1

        msg = ('<params_regressor> are not equal. (this is a bug)')
        assert self.regressor.get_params() == df['params_regressor'].values[0], msg

        data.extend([self.regressor.get_params()])
        self.regressor.fit(self.X_train_select, self.y_train)
        _, train_neg_mae, train_neg_rmse, train_r2 = self._error(train_or_test='train')
        y_pred, test_neg_mae, test_neg_rmse, test_r2 = self._error(train_or_test='test')
        data.extend([train_neg_mae, test_neg_mae, train_neg_rmse, test_neg_rmse,
                     train_r2, test_r2])
        df_test1 = pd.DataFrame([data], columns=self._get_df_test_cols())
        return df_test1


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

    def _filter_test_results(self, scoring='test_neg_mae'):
        '''
        Remove dupilate number of features (keep only lowest error)

        Parameters:
            scoring (``str``): If there are multiple scenarios with the same
                number of features, <scoring> corresponds to the <df_test>
                column that will be used to determine which scenario to keep
                (keeps the highest).
        '''
        msg = ('<df_tune> must be populated with parameters and test scroes. '
               'Have you executed ``tune_and_train()`` yet?')
        assert isinstance(self.df_tune, pd.DataFrame), msg

        df = self.df_test
        idx = df.groupby(['regressor_name', 'feat_n'])[scoring].transform(max) == df[scoring]
        idx_feat1 = df['feat_n'].searchsorted(1, side='left')
        if np.isnan(df.iloc[idx_feat1][scoring]):
            idx.iloc[idx_feat1] = True
        df['feat_n'] = df['feat_n'].apply(pd.to_numeric)
        df_filtered = df[idx].drop_duplicates(['regressor_name', 'feat_n']).sort_values(['regressor_name', 'feat_n']).reset_index(drop=True)
        self.df_test_filtered = df_filtered

    def train(self, **kwargs):
        '''
        Perform tuning for each unique scenario from ``FeatureSelection``
        (i.e., for each row in <df_fs_params>).
        '''
        print('Executing hyperparameter tuning...')
        self._set_params_from_kwargs_tune(**kwargs)

        df_tune = self.df_tune
        df_test = self.df_test
        for idx in self.df_fs_params.index:
            X_train_select, X_test_select = self.fs_get_X_select(idx)
            n_feats = len(self.df_fs_params.iloc[idx]['feats_x_select'])
            if self.print_out_train == True:
                print('Number of features: {0}'.format(n_feats))

            # param_grid_dict = self._param_grid_add_key(param_grid_dict, key)

            df_tune_grid = self._tune_grid_search()
            df_tune_rank = self._get_tune_results(df_tune_grid, rank=1)
            if df_tune is None:
                df_tune = df_tune_rank.copy()
            else:
                df_tune = df_tune.append(df_tune_rank)
            if df_test is None:
                df_test = self._get_test_results(df_tune_rank)
            else:
                df_test = df_test.append(self._get_test_results(df_tune_rank))

            if self.print_out_train is True:
                print('{0}:'.format(self.regressor_name))
                print('R2: {0:.3f}\n'.format(df_tune_rank['score_val_r2'].values[0]))

        df_tune = df_tune.sort_values('feat_n').reset_index(drop=True)
        self.df_tune = df_tune
        self.df_test = df_test
        self._filter_test_results(scoring='test_neg_mae')
