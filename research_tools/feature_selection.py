# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:47:40 2020

TRADE SECRET: CONFIDENTIAL AND PROPRIETARY INFORMATION.
Insight Sensing Corporation. All rights reserved.

@copyright: © Insight Sensing Corporation, 2020
@author: Tyler J. Nigon
@contributors: [Tyler J. Nigon]
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import SelectFromModel

from research_tools import FeatureData


class FeatureSelection(FeatureData):
    '''
    FeatureSelection inherits from ``FeatureData``, and carries out all tasks
    related to feature selection before model tuning, training, and prediction.
    The information garnered from ``FeatureSelection`` is quite simply all the
    parameters required to duplicate a given number of features, as well as its
    cross-validation results (e.g., features used, ranking, training and
    validation scores, etc.)

    Ideally, we should be able to gather all necessary information for
    performing feature selection either through:
        1. inheritance (from ``FeatureData``), or
        2. kwargs passed to ``FeatureSelection`` on initialization

    Parameters:
        model_fs (``str``): The algorithm used to carry out feature selection
        n_feats (``int``): The maximum number of features to consider
    '''
    __allowed_params = (
        'model_fs', 'model_fs_params_set', 'model_fs_params_adjust_min', 'model_fs_params_adjust_max',
        'n_feats', 'n_linspace', 'exit_on_stagnant_n',
        'step_pct', 'print_out_fs')

    def __init__(self, **kwargs):
        super(FeatureSelection, self).__init__(**kwargs)
        self.get_feat_group_X_y()
        cv_rep_strat = self.kfold_repeated_stratified()
        # FeatureData defaults
        self.model_fs = None  # params below are specific to this model
        self.model_fs_name = None
        self.model_fs_params_set = {'precompute': True,
                                    'max_iter': 100000,
                                    'tol': 0.001,
                                    'warm_start': True,
                                    'selection': 'cyclic'}
        self.model_fs_params_adjust_min = {'alpha': 1}
        self.model_fs_params_adjust_max = {'alpha': 1e-3}
        self.n_feats = 3
        self.n_linspace = 100
        # self.method_alpha_min = 'full'
        self.exit_on_stagnant_n = 5
        self.step_pct = 0.01
        self.print_out_fs = False

        self._set_params_from_kwargs_fs(**kwargs)
        self._set_attributes_fs()
        # self._set_model_fs()

    def _set_params_from_dict_fs(self, config_dict):
        '''
        Sets any of the parameters in ``config_dict`` to self as long as they
        are in the ``__allowed_params`` list
        '''
        if config_dict is not None and 'FeatureSelection' in config_dict:
            params_fd = config_dict['FeatureSelection']
        elif config_dict is not None and 'FeatureSelection' not in config_dict:
            params_fd = config_dict
        else:  # config_dict is None
            return
        for k, v in params_fd.items():
            if k in self.__class__.__allowed_params:
                setattr(self, k, v)
        if 'model_fs' in params_fd.keys():
            self._set_model_fs()

    def _set_params_from_kwargs_fs(self, **kwargs):
        '''
        Sets any of the passed kwargs to self as long as long as they are in
        the ``__allowed_params`` list. Notice that if 'config_dict' is passed,
        then its contents are set before the rest of the kwargs, which are
        passed to ``FeatureSelection`` more explicitly.
        '''
        if 'config_dict' in kwargs:
            self._set_params_from_dict_fs(kwargs.get('config_dict'))
        if kwargs is not None:
            for k, v in kwargs.items():
                if k in self.__class__.__allowed_params:
                    setattr(self, k, v)
            if 'model_fs' in kwargs.keys():
                self._set_model_fs()

    def _set_attributes_fs(self):
        '''
        Sets any class attribute to ``None`` that will be created in one of the
        user functions from the ``feature_selection`` class
        '''
        self.model_fs_params_feats_min = None
        self.model_fs_params_feats_max = None
        self.df_fs_params = None

        self.X_train_select = None
        self.X_test_select = None
        self.labels_x_select = None

    def _set_model_fs(self):
        '''
        Actually initializes the sklearn model based on the model ``str``. If
        the <random_seed> was not included in the <model_fs_params> (or if
        there is a discrepancy), the model's random state is set/reset to avoid
        any discrepancy.
        '''
        if self.model_fs_params_set is None:
            self.model_fs_params_set = {}
        self.model_fs.set_params(**self.model_fs_params_set)

        if 'regressor' in self.model_fs.get_params().keys():
            regressor_key_fs = 'regressor__'
            self.model_fs_name = type(self.model_fs.regressor).__name__
            try:
                self.model_fs.set_params(**{regressor_key_fs + 'random_state': self.random_seed})
            except ValueError:
                print('Invalid parameter <random_state> for estimator, thus '
                      '<random_state> cannot be set.\n')
        else:
            self.model_fs_name = type(self.model_fs).__name__
            try:
                self.model_fs.set_params(**{'random_state': self.random_seed})
            except ValueError:
                print('Invalid parameter <random_state> for estimator, thus '
                      '<random_state> cannot be set.\n')

    def _f_feat_n(self, **kwargs):
        '''
        Uses an ``sklearn`` model to determine the number of features selected
        from a given set of parameters. The parameters to the model should be
        passed via <kwargs>, which will be passed to the sklearn model
        instance.
        '''
        cols = ['model_fs', 'params_fs', 'feat_n', 'feats_x_select',
                'labels_x_select', 'rank_x_select']
        self.model_fs.set_params(**kwargs)
        self.model_fs.fit(self.X_train, self.y_train)
        model_bs = SelectFromModel(self.model_fs, prefit=True)
        feats = model_bs.get_support(indices=True)
        coefs = self.model_fs.coef_[feats]  # get ranking coefficients
        feat_ranking = rankdata(-coefs, method='min')
        feats_x_select = [self.labels_x[i] for i in feats]
        data = [self.model_fs_name, self.model_fs.get_params(),
                len(feats), tuple(feats), tuple(feats_x_select),
                tuple(feat_ranking)]
        df = pd.DataFrame(data=[data], columns=cols)
        return df

    def _params_adjust(self, config_dict, key, increase=True, factor=10):
        val = config_dict[key]
        if increase is True:
            config_dict[key] = val*factor
        else:
            config_dict[key] = val/factor
        return config_dict

    def _gradient_descent_step_pct_feat_max(
            self, feat_n_sel, feat_n_last, n_feats, step_pct):
        '''
        Adjusts step_pct dynamically based on progress of reaching n_feats

        Ideally, step_pct should be large if we're a long way from n_feats, and
        much smaller if we're close to n_feats
        '''
        # find relative distance in a single step
        n_feats_closer = feat_n_sel - feat_n_last
        pct_closer = n_feats_closer/n_feats
        pct_left = (n_feats-feat_n_sel)/n_feats
        # print(pct_closer)
        # print(pct_left)
        step_pct_old = step_pct
        if pct_closer < 0.08 and pct_left > 0.5 and step_pct * 10 < 1:  # if we've gotten less than 8% the way there
            step_pct *= 10
        elif pct_closer < 0.15 and pct_left > 0.4 and step_pct * 5 < 1:  # if we've gotten less than 8% the way there
            step_pct *= 5
        elif pct_closer < 0.3 and pct_left > 0.3 and step_pct * 2 < 1:  # if we've gotten less than 8% the way there
            step_pct *= 2

        elif pct_closer > 0.1 and pct_left < pct_closer*1.3:  # if % gain is 77% of what is left, slow down a bit
            step_pct /= 5
        elif pct_closer > 0.05 and pct_left < pct_closer*1.3:  # if % gain is 77% of what is left, slow down a bit
            step_pct /= 2
        else:  # keep step_pct the same
            pass
        if step_pct != step_pct_old and self.print_out_fs == True:
            print('<step_pct> adjusted from {0} to {1}'.format(step_pct_old, step_pct))
        # print('Old "step_pct": {0}'.format(step_pct))
        # print('New "step_pct": {0}'.format(step_pct))
        return step_pct

    def _find_features_max(self, n_feats, step_pct=0.01, exit_on_stagnant_n=5):
        '''
        Finds the model parameters that will result in having the max number
        of features as indicated by <n_feats>. <model_fs_params_feats_max>
        can be passed to ``_f_feat_n()`` via ``**kwargs`` to return
        a dataframe with number of features, ranking, etc.

        Returns:
            self.model_fs_params_feats_max

        Parameters:
            n_feats (``int``):
            step_pct (``float``): indicates the percentage to adjust alpha by on each
                iteration.
            exit_on_stagnant_n (``int``): Will stop searching for minimum alpha value
                if number of selected features do not change after this many
                iterations.
        '''
        msg = ('Leaving while loop before finding the alpha value that achieves '
               'selection of {0} feature(s) ({1} alpha value to use).\n')
        feat_n_sel = n_feats+1  # initialize to enter the while loop
        while feat_n_sel > n_feats:  # adjust the parameter(s) that control n_feats
            df = self._f_feat_n(**self.model_fs_params_adjust_max)
            feat_n_sel = df['feat_n'].values[0]
            self.model_fs_params_adjust_max = self._params_adjust(
                self.model_fs_params_adjust_max, key='alpha', increase=True,
                factor=10)

        same_n = 0
        while feat_n_sel != n_feats:
            feat_n_last = feat_n_sel
            df = self._f_feat_n(**self.model_fs_params_adjust_max)
            feat_n_sel = df['feat_n'].values[0]
            same_n += 1 if feat_n_last == feat_n_sel else -same_n
            if same_n > exit_on_stagnant_n:
                print(msg.format(n_feats, 'minimum'))
                break
            elif feat_n_sel == n_feats:
                break
            elif feat_n_sel < n_feats:
                step_pct = self._gradient_descent_step_pct_feat_max(
                    feat_n_sel, feat_n_last, n_feats, step_pct)
                self.model_fs_params_adjust_max = self._params_adjust(
                    self.model_fs_params_adjust_max, key='alpha',
                    increase=True, factor=(1-step_pct))
            else:  # we went over; go back to prvious step, make much smaller, and adjust alpha down a bit
                self.model_fs_params_adjust_max = self._params_adjust(
                    self.model_fs_params_adjust_max, key='alpha',
                    increase=False, factor=(1-step_pct))
                step_pct /= 10
                self.model_fs_params_adjust_max = self._params_adjust(
                    self.model_fs_params_adjust_max, key='alpha',
                    increase=True, factor=(1-step_pct))
            if self.print_out_fs == True:
                print('Adjusted parameters: {0}'.format(self.model_fs_params_adjust_max))
                print('Features selected: {0}\n'.format(feat_n_sel))
                # print('Iterations without progress: {0}\n'.format(same_n))
        if self.print_out_fs == True:
            print('Using up to {0} selected features\n'.format(feat_n_sel))
        self.model_fs_params_feats_max = self.model_fs_params_adjust_max
        self.step_pct = step_pct

    def _find_features_min(self):
        '''
        Finds the model parameters that will result in having just a single
        feature. <model_fs_params_feats_min> can be passed to ``_f_feat_n()``
        via ``**kwargs`` to return a dataframe with number of features,
        ranking, etc.
        '''
        df = self._f_feat_n(**self.model_fs_params_adjust_min)
        feat_n_sel = df['feat_n'].values[0]

        if feat_n_sel <= 1:  # the initial value already results in 1 (or 0) feats
            while feat_n_sel <= 1:
                params_last = self.model_fs_params_adjust_min
                self.model_fs_params_adjust_min = self._params_adjust(
                    self.model_fs_params_adjust_min, key='alpha',
                    increase=False, factor=1.2)
                df = self._f_feat_n(**self.model_fs_params_adjust_min)
                feat_n_sel = df['feat_n'].values[0]
            self.model_fs_params_adjust_min = params_last  # set it back to 1 feat
        else:
            while feat_n_sel > 1:
                self.model_fs_params_adjust_min = self._params_adjust(
                    self.model_fs_params_adjust_min, key='alpha',
                    increase=True, factor=1.2)
                df = self._f_feat_n(**self.model_fs_params_adjust_min)
                feat_n_sel = df['feat_n'].values[0]
        self.model_fs_params_feats_min = self.model_fs_params_adjust_min

    def _lasso_fs_df(self):
        '''
        Creates a "template" dataframe that provides all the necessary
        information for the ``Tuning`` class to achieve the specific number of
        features determined by the ``FeatureSelection`` class.
        '''
        self._find_features_min()
        self._find_features_max(
            self.n_feats, step_pct=self.step_pct,
            exit_on_stagnant_n=self.exit_on_stagnant_n)
        params_max = np.log(self.model_fs_params_feats_max['alpha'])
        params_min = np.log(self.model_fs_params_feats_min['alpha'])
        param_val_list = list(np.logspace(params_min, params_max,
                                          num=self.n_linspace, base=np.e))
        df = None
        param_adjust_temp = self.model_fs_params_feats_min.copy()
        for val in param_val_list:
            param_adjust_temp['alpha'] = val
            df_temp = self._f_feat_n(**param_adjust_temp)
            if df is None:
                df = df_temp.copy()
            else:
                df = df.append(df_temp)
        df = df.drop_duplicates(subset=['feats_x_select'], ignore_index=True)
        self.df_fs_params = df

    def _fs_get_X_select_test(self, df_fs_params_idx):
        '''
        References <df_fs_params> to provide a new matrix X that only includes
        the selected features.

        Parameters:
            df_fs_params_idx (``int``): The index of <df_fs_params> to retrieve
                sklearn model parameters (the will be stored in the "params"
                column).
        '''
        msg1 = ('<df_fs_params> must be populated; be sure to execute '
                '``find_feat_selection_params()`` prior to running '
                '``get_X_select()``.')
        msg2 = ('The selected features are different from those listed in '
                'index [{0}] of <df_fs_params>, but the regressor parameters '
                'are the same. Please be sure <X_train> or '
                '<y_train> were not modified, or that the contents of '
                '<df_fs_params> were not modified. (this is probably a '
                'limitation of repeatability by ``sklearn.SelectFromModel``).'
                ''.format(df_fs_params_idx))
        msg3 = ('The selected features are different from those listed in '
                'index [{0}] of <df_fs_params> because the regressor parameters '
                'are DIFFERENT. Please be sure the contents of '
                '<df_fs_params> were not modified. (this is probably a bug in '
                'your workflow)'
                ''.format(df_fs_params_idx))
        assert isinstance(self.df_fs_params, pd.DataFrame), msg1

        self.model_fs.set_params(**self.df_fs_params.iloc[df_fs_params_idx]['params_fs'])
        self.model_fs.fit(self.X_train, self.y_train)
        model_select = SelectFromModel(self.model_fs, prefit=True)

        feats_now = tuple(model_select.get_support(indices=True))
        labels_x_select_now = tuple([self.labels_x[i] for i in feats_now])


        labels_x_select = self.df_fs_params.iloc[df_fs_params_idx]['labels_x_select']
        if labels_x_select_now != labels_x_select:
            params_from_df = self.df_fs_params.iloc[df_fs_params_idx]['params_fs']
            params_from_model = self.model_fs.get_params()
            if params_from_df == params_from_model:
                print('params: {0}'.format(params_from_df))
                print('Features selected just now: {0}'.format(labels_x_select_now))
                print('<df_fs_features>: {0}'.format(labels_x_select))
                print(msg2 + '\n')
            else:
                print('from_df: {0}'.format(params_from_df))
                print('from_model: {0}\n'.format(params_from_model))
                print('Features selected just now: {0}'.format(labels_x_select_now))
                print('<df_fs_features>: {0}'.format(labels_x_select))
                print(msg3 + '\n')
            X_train_select, X_test_select = self._fs_get_X_select_simple(df_fs_params_idx)
        else:
            X_train_select = model_select.transform(self.X_train)
            X_test_select = model_select.transform(self.X_test)
            self.X_train_select = X_train_select
            self.X_test_select = X_test_select
            self.labels_x_select = labels_x_select
            self.rank_x_select = self.df_fs_params.iloc[df_fs_params_idx]['rank_x_select']
        return X_train_select, X_test_select

    def fs_find_params(self, **kwargs):
        '''
        Constructs a dataframe (df_fs_params) that contains all the information
        necessary to recreate unique feature selection scenarios (i.e.,
        duplicate feature list/feature ranking entries are dropped).

        The goal of this function is to cover the full range of features from 1
        to all features (max number of features varies by dataset), and be able
        to repeat the methods to derive any specific number of features.

        Returns:
            self.df_fs_params (``pd.DataFrame``): A "template" dataframe that
                provides the sklearn model (``str``), the number of features
                (``int``), the feature indices (``list``), the feature ranking
                (``list``), and the parameters to recreate the feature
                selection scenario (``dict``).
        '''
        print('Performing feature selection...')
        self._set_params_from_kwargs_fs(**kwargs)

        # msg = ('``method_alpha_min`` must be either "full" or'
        #        '"convergence_warning"')
        # assert self.method_alpha_min in ["full", "convergence_warning"], msg
        if self.n_feats is None:
            self.n_feats = self.X_train.shape[1]
        else:
            self.n_feats = int(self.n_feats)
        if self.n_feats > self.X_train.shape[1]:
            self.n_feats = self.X_train.shape[1]

        if self.model_fs_name == 'Lasso':
            self._lasso_fs_df()
        elif self.model_fs_name == 'PCA':
            raise NotImplementedError('{0} is not implemented.'.format(self.model_fs_name))
        # else:
        #     raise NotImplementedError('{0} is not implemented.'.format(self.model_fs_name))

    def fs_get_X_select(self, df_fs_params_idx):
        '''
        References <df_fs_params> to provide a new matrix X that only includes
        the selected features.

        Parameters:
            df_fs_params_idx (``int``): The index of <df_fs_params> to retrieve
                sklearn model parameters (the will be stored in the "params"
                column).
        '''
        msg1 = ('<df_fs_params> must be populated; be sure to execute '
                '``find_feat_selection_params()`` prior to running '
                '``get_X_select()``.')
        assert isinstance(self.df_fs_params, pd.DataFrame), msg1

        feats_x_select = self.df_fs_params.iloc[df_fs_params_idx]['feats_x_select']

        X_train_select = self.X_train[:,feats_x_select]
        X_test_select = self.X_test[:,feats_x_select]
        self.X_train_select = X_train_select
        self.X_test_select = X_test_select
        self.labels_x_select = self.df_fs_params.iloc[df_fs_params_idx]['labels_x_select']
        self.rank_x_select = self.df_fs_params.iloc[df_fs_params_idx]['rank_x_select']
        return X_train_select, X_test_select

