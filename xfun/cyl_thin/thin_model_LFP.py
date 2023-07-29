# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     pre_data
   Description :
   Author :       ASUS
   date：          2023/3/23
-------------------------------------------------
   Change Activity:
                   2023/3/23:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
from xfun.train_model.corr_features_calcu import FeatureAnsys
from xfun.train_model.models_calcu import ModelTrain
from xfun.pred_model.model_preds import ModelPredict
from utils.params_config import ConfPath, ConfVars, ConstVars
from log_conf.logger import logger
from xfun.z_post_plot.plt_mdl_data import PostVisual

# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)
        print()


class PreClean(ConfParams):

    def __init__(self, x_step, y_step):
        ConfParams.__init__(self)
        self.summary_sheet_name = '循环统计'
        self.date_col = ['恒流充电时间(Sec)', '恒压充电时间(Sec)', '恒流放电时间(Sec)', '恒压放电时间(Sec)',
                         '总放电时间(Sec)']
        self.x_step = x_step
        self.y_step = y_step
        print()

    def xy_shift(self, df_icycl, x_step, y_step):
        """
        1.SPLIT CYL_NUM
           src_step:     1-1
                        ....
                        1-100
                        2-1
                        ...
                        2-100
           CAST INTO:
                       1
                       ...
                       100
                       101
                       ...
                       200
                       201
                       ...

        2.SHIFT K_STEP
        3.merge shifted data ,set_index ['SORT_DATE','CYCLE_NUM']
        4. given set_num (x_step, y_step)=(50,150) see belowing 5_step,
           A. SHIFT_NUM= 150-100
           B. DROP *  all record rows by axis=0
           C. drop record Y_STEP>150
        5.      X_STEP    Y_STEP
                    *     0
                    *     50
                    0     100
           set_num: 50    150
                    100   200
                    150   *
                    200   *
        6. timevar_features = ['DISCHG_ENDENERGY', 'CHG_ENDCAPACITY', 'CHG_ENDENERGY']
           can only used: T(N-1) ,
              forbidden:  T(N)

       7. shift  or scala
               df_shfit = df_shfit[(df_shfit['CYCLE_INT'] >= y_step - 10) & (df_shfit['CYCLE_INT'] <= y_step)]


        # df_x_cut = df_x.drop(columns=self.timevar_features, axis=1)
        # df_x_tvar = df_x.loc[:, self.timevar_features].shift(periods=k_step)
        # df_x_rev = pd.concat([df_x_cut, df_x_tvar], axis=1)

        """
        k_step = y_step - x_step
        step_x_idx = df_icycl[self.cycle_num].min() < x_step < df_icycl[self.cycle_num].max()
        step_y_idx = df_icycl[self.cycle_num].min() < y_step < df_icycl[self.cycle_num].max()
        if not (step_x_idx or step_y_idx):
            logger.info('set step shift errors,step within({},{})'.format(x_step, y_step))
        # df_icycl = df_icycl.reset_index().set_index(self.drop_cols)
        df_x = df_icycl.drop([self.y_label], axis=1)
        df_yshift = df_icycl[self.y_label].shift(periods=-k_step)
        df_shift = pd.concat([df_x, df_yshift], axis=1).dropna(how='any', axis=0)
        return df_shift

    def gen_mdldata(self, RUN_MODE):
        src_path = os.path.join(self.raw_data_path, RUN_MODE)
        df_tr_pred = []
        for file in os.listdir(src_path):
            if not file.endswith('.xlsx'):
                continue
            file_path = os.path.join(src_path, file)
            df = pd.read_excel(file_path, sheet_name=self.summary_sheet_name,
                               usecols=self.x_columns + [self.y_label])
            df_cyl = self.date2hours(df)
            df_cyl['CYCLE_NUM'] = df_cyl[self.cycle_num]
            df_cyl['SN'] = file.split('.')[0]
            df_cyl = df_cyl.set_index(['SN', 'CYCLE_NUM'])
            df_shift = self.xy_shift(df_cyl, self.x_step, self.y_step)
            df_tr_pred.append(df_shift[df_shift[self.cycle_num] <= self.x_step])

        if RUN_MODE == 'TRAIN':
            df_mode = pd.concat(df_tr_pred, axis=0)
            df_mode.to_excel(os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_mdldata.xls'),
                             sheet_name='cyl_data',
                             index_label=['SN', 'CYCLE_NUM'])
            tr_path = os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_cap_join_mdldata.pkl')
            with open(tr_path, 'wb') as trd:
                pickle.dump(df_mode, trd)
        elif RUN_MODE == 'PRED':
            pred_path = os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_cap_join_mdldata.pkl')
            with open(pred_path, 'wb') as prd:
                pickle.dump(df_tr_pred, prd)

    def convert_hour(self, x):
        return x.hour * 60 + x.minute + x.second / 60

    def date2hours(self, df):
        for date_x in self.date_columns:
            df[date_x] = df[date_x].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S.%f'))
            df[date_x] = df[date_x].apply(self.convert_hour)
        return df

    def gen_mdldata_split(self, RUN_MODE, x_step, y_step):
        """
        a joke......
        when x(t+k) is tested,y(t+k) is gotten,
        why should we prdict y(t+k)
        """
        df_tr = []
        df_pred = []
        for file in os.listdir(self.raw_data_path):
            if not file.endswith('.xlsx'):
                continue
            file_path = os.path.join(self.raw_data_path, file)
            df = pd.read_excel(file_path, sheet_name=self.summary_sheet_name, usecols=self.x_columns + [self.y_label])
            df_rev = self.date2hours(df)
            df_rev['CYCLE_NUM'] = df_rev['循环号']
            df_rev['SN'] = file.split('.')[0]
            df_rev = df_rev.set_index(['SN', 'CYCLE_NUM'])
            df_tr.append(df_rev[df_rev['循环号'] <= x_step])
            df_pred.append(df_rev[(y_step - 50 <= df_rev['循环号']) & (df_rev['循环号'] <= y_step)])

        df_tr = pd.concat(df_tr, axis=0)

        tr_path = os.path.join(self.ts_avg_fatures_path, 'TRAIN' + '_cap_join_mdldata.pkl')
        with open(tr_path, 'wb') as trd:
            pickle.dump(df_tr, trd)

        pred_path = os.path.join(self.ts_avg_fatures_path, 'PRED' + '_cap_join_mdldata.pkl')
        with open(pred_path, 'wb') as prd:
            pickle.dump(df_pred, prd)


class ModelCalcu(ConfParams):

    def __init__(self, step_x, step_y):
        ConfParams.__init__(self)
        self.step_x = step_x
        self.step_y = step_y

    def train_data(self, RUN_MODE, mdl_name):
        tr_path = os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_cap_join_mdldata.pkl')
        with open(tr_path, 'rb') as trd:
            self.df_tr = pickle.load(trd)
        params = {
            'learning_rate': 0.1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 4,
            'objective': 'regression'
        }
        self.mdl = lgb.LGBMRegressor(**params)
        self.mdl.fit(self.df_tr.drop(columns=self.y_label, axis=1), self.df_tr[self.y_label])
        logger.info('{} fit done'.format(mdl_name))

    def pred_data(self, RUN_MODE, mdl_name):
        pred_path = os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_cap_join_mdldata.pkl')
        with open(pred_path, 'rb') as prd:
            self.df_pred = pickle.load(prd)
        for i_df in self.df_pred:
            try:
                file_name = i_df.index[0][0]
                y_pred = self.mdl.predict(i_df.drop(columns=self.y_label, axis=1))
            except Exception as err:
                logger.info('cycle errors:{},{}-{}.xlsx'.format(err, self.step_x, self.step_y))
            df_pred = pd.DataFrame(columns=['TRUE', 'PRED', 'MAPE'])
            df_pred['TRUE'] = i_df[self.y_label]
            df_pred['PRED'] = y_pred
            df_pred['MAPE'] = 1 - abs(df_pred['TRUE'] - df_pred['PRED']) / df_pred['TRUE']
            pred_path_t = os.path.join(self.pred_path, mdl_name, file_name)
            if not os.path.exists(pred_path_t):
                os.makedirs(pred_path_t)
            pred_sv_name = os.path.join(pred_path_t, '{}-{}.xlsx'.format(self.step_x, self.step_y))
            df_pred.to_excel(pred_sv_name, sheet_name='cyl_data', index=True, encoding="utf_8_sig")
        logger.info('{} pred done'.format(mdl_name))


if __name__ == "__main__":
    """
        mc = ModelCalcu(step_x, step_y)
        mc.train_data('TRAIN', mdl_name)
        # 4.model predict
        mc.pred_data('PRED', mdl_name)
    """
    x_range = np.arange(50, 201, 50)
    y_range = np.arange(500, 3001, 250)
    for step_x in x_range:
        for step_y in y_range:
            pcl = PreClean(step_x, step_y)
            pcl.gen_mdldata('TRAIN')
            pcl.gen_mdldata('PRED')
            # 3.model train
            for mdl_name in ['lgbm']:
                fas = FeatureAnsys()
                fas.features_calcu('TRAIN', mdl_name)
                mt = ModelTrain(mdl_name)
                mt.fit_mdl(mdl_name)
                # 4.model predict
                mt = ModelPredict('PRED', mdl_name, step_x, step_y)
                mt.pred_tmp(mdl_name)

    # 5.plot postvisual
    pv = PostVisual('TRAIN', 'lgbm', x_range, y_range)
    pv.plt_dist()
    pv.plt_heatmap_corr()
    pv.plt_cap_corr()
    pv.plt_feature_imp()
    pv.plt_trend1_u()
    pv.plt_cyl_pred()
