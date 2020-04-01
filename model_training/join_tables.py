# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:43:09 2020

@author: nigon
"""

class join_tables(object):
    '''
    Class for joining tables that contain training data. In addition to the
    join, there are functions available to add new columns to the table to
    act as unique features to explain the response variable being predicted.
    '''
    def __init__(self, base_dir=None, search_ext='.bip', dir_level=0):
        '''
        Parameters:
            base_dir (``str``, optional): directory path to search for files to
                spectrally clip; if ``fname_list`` is not ``None``, ``base_dir`` will
                be ignored (default: ``None``).
            search_ext (``str``): file format/extension to search for in all
                directories and subdirectories to determine which files to
                process; if ``fname_list`` is not ``None``, ``search_ext`` will
                be ignored (default: 'bip').
            dir_level (``int``): The number of directory levels to search; if
                ``None``, searches all directory levels (default: 0).
        '''

# In[Join functions]
base_dir_data = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Data'
base_dir_out = r'I:\Shared drives\NSF STTR Phase I – Potato Remote Sensing\Historical Data\Rosen Lab\Small Plot Data\Analysis Results\feat_dae'
fname_petiole = os.path.join(base_dir_data, 'tissue_petiole_NO3_ppm.csv')
fname_total_n = os.path.join(base_dir_data, 'tissue_wp_N_pct.csv')
fname_cropscan = os.path.join(base_dir_data, 'cropscan.csv')
fname_dates = os.path.join(base_dir_data, 'metadata_dates.csv')
fname_experiments = os.path.join(base_dir_data, 'metadata_exp.csv')
fname_treatments = os.path.join(base_dir_data, 'metadata_trt.csv')
fname_n_apps = os.path.join(base_dir_data, 'metadata_trt_n.csv')
fname_n_crf = os.path.join(base_dir_data, 'metadata_trt_n_crf.csv')


df_pet_no3 = pd.read_csv(fname_petiole)
df_total_n = pd.read_csv(fname_total_n)
df_cs = pd.read_csv(fname_cropscan)
df_dates = pd.read_csv(fname_dates)

df_exp = pd.read_csv(fname_experiments)
df_trt = pd.read_csv(fname_treatments)
df_n_apps = pd.read_csv(fname_n_apps)
df_n_cr = pd.read_csv(fname_n_crf)

df_pet_no3 = get_n_to_date(df_pet_no3, df_exp, df_trt, df_n_apps)
df_total_n = get_n_to_date(df_total_n, df_exp, df_trt, df_n_apps)

df_pet = join_closest_date(join_and_calc_dae(df_pet_no3, df_dates), df_cs)
df_n = join_closest_date(join_and_calc_dae(df_total_n, df_dates), df_cs)

df_pet_no3 = join_and_calc_dae(df_pet_no3, df_dates)
df_total_n = join_and_calc_dae(df_total_n, df_dates)
df_pet_no3.to_csv(os.path.join(base_dir_data, 'pet_no3_N_to_date.csv'), index=False)
df_total_n.to_csv(os.path.join(base_dir_data, 'total_N_to_date.csv'), index=False)

def join_and_calc_dae(df_left, df_dates):
    '''
    Adds columns 'dap' and 'dae' (days after planting/emergence) to df_left
    '''
    df_left['date'] = pd.to_datetime(df_left['date'])
    df_dates[['date_plant', 'date_emerge']] = df_dates[['date_plant','date_emerge']].apply(pd.to_datetime, format='%Y-%m-%d')
    df_join = df_left.merge(df_dates, on=['study', 'year'], validate='many_to_one')
    df_join['dap'] = (df_join['date']-df_join['date_plant']).dt.days
    df_join['dae'] = (df_join['date']-df_join['date_emerge']).dt.days
    df_out = df_join[['study', 'year', 'plot_id', 'date', 'dap', 'dae']]
    df_out = df_left.merge(df_out, on=['study', 'year', 'plot_id', 'date'])
    return df_out
df_left = df_total_n.copy()
def get_n_to_date(df_left, df_exp, df_trt, df_n_apps):
    '''
    Adds a column "rate_n_kgha_to_date" indicating the amount of N applied
    before "date" (not inclusive)
    First joins df_exp to get unique treatment ids for each plot
    Second joins df_trt to get "breakout" treatment ids for "trt_n"
    Third, joins df_n_apps to get date_applied and rate information
    Fourth, calculates the rate N applied to date (sum of N applied before the
    "date" column; "date" must be a column in df_left)
    '''
    msg = ('"date" must be a column in ``df_left`` to be able to calculate '
           'amount of N applied thus far for each observation.')
    assert 'date' in df_left.columns, msg
    df_left['date'] = pd.to_datetime(df_left['date'])
    df_join = df_left.merge(df_exp, on=['study', 'year', 'plot_id'])
    df_join = df_join.merge(df_trt[['study', 'year', 'trt_id', 'trt_n']],
                            on=['study', 'year', 'trt_id'])
    df_n_apps['date_applied'] = pd.to_datetime(df_n_apps['date_applied'])
    df_join = df_join.merge(
        df_n_apps[['study', 'year', 'trt_n', 'date_applied', 'rate_n_kgha']],
        on=['study', 'year', 'trt_n'], validate='many_to_many')

    # remove all rows where date_applied is after date
    df_join = df_join[df_join['date'] >= df_join['date_applied']]
    df_sum = df_join.groupby(['study','year', 'plot_id', 'date', 'tissue', 'measure'])['rate_n_kgha'].sum().reset_index()
    df_sum.rename(columns={'rate_n_kgha':'rate_n_kgha_to_date'}, inplace=True)
    df_out = df_left.merge(df_sum, on=['study', 'year', 'plot_id', 'date'])
    return df_out