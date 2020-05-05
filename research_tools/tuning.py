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

from research_tools import FeatureSelection


class Tuning(FeatureSelection):
    '''
    ``Tuning`` inherits from an instance of ``FeatureSelection`` (which inherits
    from ``FeatureData``), then includes new functions to specifically carry
    out the hyperparameter tuning steps.
    '''
    __allowed_params = (
        'regressor', 'regressor_params', 'param_grid', 'n_jobs_tune',
        'scoring', 'refit', 'rank_scoring', 'print_out_tune')

    def __init__(self, **kwargs):
        '''
        '''
        super(Tuning, self).__init__(**kwargs)
        # self._set_params_from_kwargs_fs(**kwargs)
        # self._set_attributes_fs()
        # self._set_model_fs()
        self.fs_find_params(**kwargs)


        # Tuning defaults
        self.regressor = None
        self.regressor_name = None
        self.regressor_params = {}
        self.regressor_key = None
        self.param_grid = {'alpha': list(np.logspace(-4, 0, 5))}
        self.n_jobs_tune = None
        self.scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2')
        self.refit = self.scoring[0]
        self.rank_scoring = self.scoring[0]
        self.print_out_tune = False

        self._set_params_from_kwargs_tune(**kwargs)
        self._set_attributes_tune()
        # self._set_regressor()


    def _set_params_from_dict_tune(self, param_dict):
        '''
        Sets any of the parameters in ``param_dict`` to self as long as they
        are in the ``__allowed_params`` list
        '''
        if param_dict is not None and 'Tuning' in param_dict:
            params_fd = param_dict['Tuning']
        elif param_dict is not None and 'Tuning' not in param_dict:
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
        passed to ``Tuning`` more explicitly.
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
        elif 'param_dict' in kwargs and 'Tuning' in kwargs.get('param_dict'):
            params_fd = kwargs.get('param_dict')['Tuning']
            # print(params_fd.keys())
            if 'regressor' in kwargs.get('param_dict')['Tuning'].keys():
                self._set_regressor()
        #     ('param_dict' in kwargs and 'regressor' in kwargs.get('param_dict')['Tuning'].items())):
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

    def _get_df_tune_cols(self):
        '''
        Gets column names for tuning dataframe
        '''
        cols = ['model_fs', 'feat_n', 'feats_x_select', 'rank_x_select',
                'regressor', 'params_regressor', 'params_tuning']
        prefixes = ['score_train_', 'std_train_', 'score_val_', 'std_val_']
        for obj_str in self.scoring:
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
                self.regressor_name]
        if not isinstance(df, pd.DataFrame):
            data.extend([np.nan] * (len(self._get_df_tune_cols()) - len(data)))
            return pd.DataFrame(data=[data], columns=self._get_df_tune_cols())
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

        df_temp = pd.DataFrame(data=[data], columns=self._get_df_tune_cols())
        return df_temp

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

    def _filter_tuning_results(self):
        '''
        Remove dupilate number of features (keep only lowest error)
        '''
        msg = ('<df_tune> must be populated with parameters and test scroes. '
               'Have you executed ``tune_model()`` yet?')
        assert isinstance(self.df_tune, pd.DataFrame), msg

        scoring = 'score_val_' + self.rank_scoring

        df = self.df_tune
        idx = df.groupby(['regressor', 'feat_n'])[scoring].transform(max) == df[scoring]
        idx_feat1 = df['feat_n'].searchsorted(1, side='left')  # if first non-zero feat_n row is NaN, include that so other dfs have same number of rows (PLS)
        if np.isnan(df.iloc[idx_feat1][scoring]):
            idx.iloc[idx_feat1] = True
        df['feat_n'] = df['feat_n'].apply(pd.to_numeric)
        df_filtered = df[idx].drop_duplicates(['regressor', 'feat_n']).sort_values(['regressor', 'feat_n']).reset_index(drop=True)
        self.df_tune_filtered = df_filtered

    def tune_regressor(self, **kwargs):
        '''
        Perform tuning for each unique scenario from ``FeatureSelection``
        (i.e., for each row in <df_fs_params>).
        '''
        print('Executing hyperparameter tuning...')
        self._set_params_from_kwargs_tune(**kwargs)

        df_tune = self.df_tune
        for idx in self.df_fs_params.index:
            X_train_select, X_test_select = self.fs_get_X_select(idx)
            n_feats = len(self.df_fs_params.iloc[idx]['feats_x_select'])
            if self.print_out_tune == True:
                print('Number of features: {0}'.format(n_feats))

            # param_grid_dict = self._param_grid_add_key(param_grid_dict, key)

            df_tune_grid = self._tune_grid_search()
            df_tune_rank = self._get_tune_results(df_tune_grid, rank=1)
            if df_tune is None:
                df_tune = df_tune_rank.copy()
            else:
                df_tune = df_tune.append(df_tune_rank)

            if self.print_out_tune is True:
                print('{0}:'.format(self.regressor_name))
                print('R2: {0:.3f}\n'.format(df_tune_rank['score_val_r2'].values[0]))
        df_tune = df_tune.sort_values('feat_n').reset_index(drop=True)
        self.df_tune = df_tune
        self._filter_tuning_results()
