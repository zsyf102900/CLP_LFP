# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name：     clean_merge
   Description :
   Author :       ASUS
   date：          2023/3/6
-------------------------------------------------
   Change Activity:
                   2023/3/6:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import pickle
import numpy as np
import pandas as pd
from utils.params_config import ConfPath, ConfVars, ConstVars
from log_conf.logger import logger


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)


# ************************@@@@@***************************#
#   merge datasets: dqdv/dvdq peak features and count-values
#   construct train data in shift order
# ************************@@@@@***************************#

class PreClean(ConfParams):
    """
    class PreClean:
               lfp_pv_join: join dqdv features and count features
               gen_mdldata:construct train data in shifted patterns
    Attributes:
              lfp_pv_join
              gen_mdldata
    """

    def __int__(self):
        ConfParams.__init__(self)

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
        8. drop topk cycles data

        """
        k_step = y_step - x_step
        step_x_idx = df_icycl['CYCLE_INT'].min() < x_step < df_icycl['CYCLE_INT'].max()
        step_y_idx = df_icycl['CYCLE_INT'].min() < y_step < df_icycl['CYCLE_INT'].max()
        if not (step_x_idx or step_y_idx):
            logger.info('set step shift errors,step within({},{})'.format(x_step, y_step))
        df_icycl = df_icycl.reset_index().set_index(self.drop_cols)
        df_icycl = df_icycl[df_icycl['CYCLE_INT'] >= self.dqdv_initcycle]
        df_x = df_icycl.drop([self.y_label], axis=1)
        df_yshift = df_icycl[self.y_label].shift(periods=-k_step)
        df_shift = pd.concat([df_x, df_yshift], axis=1).dropna(how='any', axis=0)
        df_shift = df_shift[df_shift['CYCLE_INT'] <= x_step]
        return df_shift

    def gen_mdldata(self, RUN_MODE, x_step, y_step):
        """
        1.for train_set,join all sn_record,without considering order
        format below:
                index       capacity  voltage
                SN1-cycle1    100.5         3.2
                 ............
                SN1-cycle100    98.2      2.83
                SN2-cycle1    100.4         3.2
                 ............
                SN2-cycle100    98.1      2.85
                SN3-cycle1    100.5         3.2
                 ............
                SN3-cycle100    98.8      2.84

        2. for pred_set ,pred by SN separately
           a.format below:
                [SN1,
                SN2,
                .....
                SN100]
           b.pred_set
             SN2.iloc[-1,:] represent pred_sets

        3.  xy_shift keeep in shift order

        """
        df_tr_pred = []
        mode_path = os.path.join(self.data_join_path, RUN_MODE)
        for sub_sysfold in os.listdir(mode_path):
            sysfold_path = os.path.join(mode_path, sub_sysfold)
            for fold in os.listdir(sysfold_path):
                tr_fold = os.path.join(sysfold_path, fold)
                for file in os.listdir(tr_fold):
                    if not file.endswith('summary.xlsx'):
                        continue
                    file_path = os.path.join(tr_fold, file)
                    df_cyl = pd.read_excel(file_path, sheet_name='cyl_data', index_col=self.date_col)
                    df_cyl = df_cyl.reset_index().set_index(self.drop_cols)
                    df_shift = self.xy_shift(df_cyl, x_step, y_step)
                    if not df_shift.shape[0]:
                        logger.info('{} not enough shift data: {} to {}'.format(fold, x_step, y_step))
                        continue
                    df_tr_pred.append(df_shift)

        if RUN_MODE == 'TRAIN':
            df_mode = pd.concat(df_tr_pred, axis=0)
            df_mode.to_excel(os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_mdldata.xlsx'),
                             sheet_name='cyl_data',
                             index_label=self.drop_cols)
            tr_path = os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_cap_join_mdldata.pkl')
            with open(tr_path, 'wb') as trd:
                pickle.dump(df_mode, trd)
        elif RUN_MODE == 'PRED':
            df_mode = pd.concat(df_tr_pred, axis=0)
            df_mode.to_excel(os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_mdldata.xlsx'),
                             sheet_name='cyl_data',
                             index_label=self.drop_cols)

            pred_path = os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_cap_join_mdldata.pkl')
            with open(pred_path, 'wb') as prd:
                pickle.dump(df_tr_pred, prd)

    def gen_pred_only(self, RUN_MODE, x_step, y_step):

        """
        1.for train_set,join all sn_record,without considering order
        format below:
                index       capacity  voltage
                SN1-cycle1    100.5         3.2
                 ............
                SN1-cycle100    98.2      2.83
                SN2-cycle1    100.4         3.2
                 ............
                SN2-cycle100    98.1      2.85
                SN3-cycle1    100.5         3.2
                 ............
                SN3-cycle100    98.8      2.84

        2. for pred_set ,pred by SN separately
           a.format below:
                [SN1,
                SN2,
                .....
                SN100]
           b.pred_set
             SN2.iloc[-1,:] represent pred_sets

        3.  pred data donot need any y data

        """
        df_tr_pred = []
        mode_path = os.path.join(self.data_join_path, RUN_MODE)
        for sub_sysfold in os.listdir(mode_path):
            sysfold_path = os.path.join(mode_path, sub_sysfold)
            for fold in os.listdir(sysfold_path):
                tr_fold = os.path.join(sysfold_path, fold)
                for file in os.listdir(tr_fold):
                    if not file.endswith('summary.xlsx'):
                        continue
                    file_path = os.path.join(tr_fold, file)
                    df_cyl = pd.read_excel(file_path, sheet_name='cyl_data', index_col=self.date_col)
                    df_cyl = df_cyl.reset_index().set_index(self.drop_cols)
                    df_shift = df_cyl.loc[df_cyl['CYCLE_INT'] < x_step, :]
                    if not df_shift.shape[0]:
                        logger.info('pred_only,{} not enough step data to pred: {} to {}'.format(fold, x_step, y_step))
                        continue
                    df_tr_pred.append(df_shift)

        df_mode = pd.concat(df_tr_pred, axis=0)
        df_mode.to_excel(os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_mdldata.xlsx'),
                         sheet_name='cyl_data',
                         index_label=self.drop_cols)
        pred_path = os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_cap_join_mdldata.pkl')
        with open(pred_path, 'wb') as prd:
            pickle.dump(df_tr_pred, prd)


if __name__ == "__main__":
    pcl = PreClean()
    pcl.gen_mdldata('TRAIN', 80, 180)
    pcl.gen_mdldata('PRED', 80, 180)
