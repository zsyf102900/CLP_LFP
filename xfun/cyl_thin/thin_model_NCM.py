# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     thin_model_NCM.py
   Description :
   Author :       ASUS
   date：          2023/4/3
-------------------------------------------------
   Change Activity:
                   2023/4/3:
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
import time
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


# ************************@@@@@***************************#
#   gen_mdldata :keep in shifted patterns
# ************************@@@@@***************************#


class PreClean(ConfParams):

    def __init__(self, x_step, y_step):
        ConfParams.__init__(self)
        self.summary_sheet_name = '循环'
        self.x_columns = [
            '循环序号', '充电容量/mAh', '充电能量/mWh', '放电能量/mWh', '放电中压/V', '恒流充入容量/mAh',
            '恒流充入比例/%',
            '平台容量/mAh', '放电终压/V'
        ]
        self.cycle_num = '循环序号'
        self.y_label = '放电容量/mAh'
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
            if not file.endswith('.xls'):
                continue
            file_path = os.path.join(src_path, file)
            df_cyl = pd.read_excel(file_path, sheet_name=self.summary_sheet_name,
                                   usecols=self.x_columns + [self.y_label])

            df_cyl['CYCLE_NUM'] = df_cyl[self.cycle_num].astype(dtype=np.int)
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


if __name__ == "__main__":
    """
        mc = ModelCalcu(step_x, step_y)
        mc.train_data('TRAIN', mdl_name)
        # 4.model predict
        mc.pred_data('PRED', mdl_name)
    """
    # x_range = np.arange(50, 201, 50)
    # y_range = np.arange(800, 1460, 50)
    x_range = np.arange(50, 80, 10)
    y_range = np.arange(180, 220, 10)
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
    pv.plt_cyl_pred()
    # pv.plt_dist()
    # pv.plt_heatmap_corr()
    # pv.plt_cap_corr()
    # pv.plt_feature_imp()


