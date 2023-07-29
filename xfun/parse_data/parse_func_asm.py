# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     parse_func_asm.py
   Description :
   Author :       ASCEND
   date：          2023/1/12
-------------------------------------------------
   Change Activity:
                   2023/1/12:
-------------------------------------------------
"""
__author__ = 'ASCEND'

import os
import shutil
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool
from log_conf.logger import logger
from utils.params_config import ConfPath, ConfVars, ConstVars


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#
class ParseData(ConfParams):

    def __init__(self):
        ConfParams.__init__(self)


# ************************@@@@@***************************#
#   Class split cycle_data
# ************************@@@@@***************************#
class LoadSmall(ConfParams):
    """
    class LFPLoadSmall:
                     split cycle data ,
                     generate dqdv  count features

    Atttibutes:
        split_cycle:
        gen_count_features:

    """

    def __init__(self):
        ConfParams.__init__(self)

    def split_cyl_fun(self, df_cycllist, cyl_list, cylfile_path):
        # for cycles：75A 0.5C test  drop out
        for cyl in cyl_list:
            df_cycl = df_cycllist.loc[df_cycllist[self.vars['CYCLESID']].isin([cyl]), :]
            cyl_lambda = lambda x: (int(x.split('-')[0]) - 1) * 100 + int(x.split('-')[1])
            df_cycl.loc[:, 'CYCLE_INT'] = df_cycl.loc[:, self.vars['CYCLESID']].apply(cyl_lambda)
            index_current = df_cycl[self.vars['STATUS']] == self.vars_CD['DC']
            df_cd = df_cycl.loc[index_current, self.vars['CURRENCY']]
            # if exists small current discharge,skip
            if df_cd.apply(lambda x: np.all(x > 10 and x < 80)).any():
                continue
            save_name = os.path.join(cylfile_path, cyl + '.xlsx')
            df_cycl.to_excel(save_name, sheet_name='cyl_data')
        logger.info('asm_list: {} \n cycle split save done'.format(cyl_list))

    def split_cyl_fun_ncm(self, df_cycllist, cyl_list, cylfile_path):
        # for cycles：75A 0.5C test  drop out
        for cyl in cyl_list:
            df_cycl = df_cycllist.loc[df_cycllist[self.vars['CYCLESID']].isin([cyl]), :]
            save_name = os.path.join(cylfile_path, str(cyl) + '.xlsx')
            df_cycl.to_excel(save_name, sheet_name='cyl_data')
        logger.info('asm_list: {} \n cycle split save done'.format(cyl_list))

    def split_cyl_fun_lfp_hyn(self, df_cycllist, cyl_list, cylfile_path):
        for cyl in cyl_list:
            df_cycl = df_cycllist.loc[df_cycllist[self.vars['CYCLESID']].isin([cyl]), :]
            save_name = os.path.join(cylfile_path, str(cyl) + '.xlsx')
            df_cycl.to_excel(save_name, sheet_name='cyl_data')
        logger.info('asm_list: {} \n cycle split save done'.format(cyl_list))

    def split_cycle(self, RUN_MODE):
        """
        1. split into separate cycle file
        2. drop int cycle DCR TEST
        # df = pd.read_excel(file_path, sheet_name='记录层', usecols=list(self.vars.values()))
        """
        mode_path = os.path.join(self.raw_data_path, RUN_MODE)
        for sys_fold in os.listdir(mode_path):
            sys_fold_path = os.path.join(mode_path, sys_fold)
            for file in os.listdir(sys_fold_path):
                if not (file.endswith('.xlsx') or file.endswith('.xls')):
                    continue
                file_path = os.path.join(sys_fold_path, file)
                logger.info('load file:{}'.format(file))
                # df = pd.read_excel(file_path, sheet_name='记录层', usecols=list(self.vars.values()))
                df = pd.ExcelFile(file_path)
                df = pd.concat([pd.read_excel(df, sheet, usecols=list(self.vars.values()))
                                for sheet in df.sheet_names if sheet.startswith(self.sheet_src)])
                df = df.set_index([self.vars['STEPTIME']])

                if self.MODEL_ID.startswith('LFPHYN') and ('G44' in file or 'P2E24' in file or 'P2E72' in file):
                    df[self.vars['CAPACITY']] = df[self.vars['CAPACITY']].apply(lambda x: x / 1000 if x > 1000 else x)
                    df[self.vars['ENERGY']] = df[self.vars['ENERGY']].apply(lambda x: x / 1000 if x > 1000 else x)

                logger.info('load file:{} done'.format(file))
                cylfile_path = os.path.join(self.cycle_data_path, RUN_MODE, sys_fold, file.split('.')[0])
                if not os.path.exists(cylfile_path):
                    os.makedirs(cylfile_path)

                dcr_test_file = [str(i) for i in range(0, 1000, 1)]
                cycle_all = [file for file in df[self.vars['CYCLESID']].unique() if file not in dcr_test_file]
                cyl_slice = [cycle_all[i:i + self.MAX_SLICE_PAREL] for i in
                             range(0, len(cycle_all), self.MAX_SLICE_PAREL)]

                pool = Pool(self.POOL_NUM)
                for cyl_slc in cyl_slice:
                    df_cylist = df.loc[df[self.vars['CYCLESID']].isin(cyl_slc), :]
                    if self.MODEL_ID.startswith('LFPXY'):
                        # self.split_cyl_fun(df_cylist, cyl_slc, cylfile_path)
                        pool.apply_async(self.split_cyl_fun, args=(df_cylist, cyl_slc, cylfile_path,))
                    elif self.MODEL_ID.startswith('NCM'):
                        # self.split_cyl_fun_ncm(df_cylist, cyl_slc, cylfile_path)
                        pool.apply_async(self.split_cyl_fun_ncm, args=(df_cylist, cyl_slc, cylfile_path,))
                    elif self.MODEL_ID.startswith('LFPHYN'):
                        pool.apply_async(self.split_cyl_fun_ncm, args=(df_cylist, cyl_slc, cylfile_path,))

                pool.close()
                pool.join()
            logger.info('split_cycle done ')

    def dqdv_dvdq_fun(self, df_cycl):
        df_cycl.loc[:, 'dv'] = df_cycl.loc[:, self.vars['VOLTAGE']].diff()
        df_cycl.loc[:, 'dq'] = df_cycl.loc[:, self.vars['CAPACITY']].diff()
        df_cycl.dropna(how='any', inplace=True)
        df_cycle_p = df_cycl.loc[:, ['dv', 'dq']].rolling(window=self.dqdv_window, min_periods=1).mean()
        df_cycle_p.loc[:, 'dqdv_roll'] = df_cycle_p.loc[:, 'dq'] / df_cycle_p.loc[:, 'dv']
        df_cycle_p.loc[:, 'dvdq_roll'] = df_cycle_p.loc[:, 'dv'] / df_cycle_p.loc[:, 'dq']
        df_cycle_p.rename(columns={'dv': 'dv_roll', 'dq': 'dq_roll'}, inplace=True)
        df_cycl = df_cycl.merge(df_cycle_p, how='left', left_index=True, right_index=True)
        df_cycl.dropna(how='any', inplace=True)
        return df_cycl

    def gen_count_features_fun(self, sys_fold_path, slice_filelist, k, save_path):
        """
        1. drop such file do dcr_test  int cycle startswith： 0.xls, 1.xls ,2.xls , 3.xls
        2. define raw_features may vary in different datasets
        # .isin(['充电 CC', '静置', '放电 DC'])

            # dcr_test_file = [str(i) + '.xlsx' for i in range(0, 100, 1)]
            # if file in dcr_test_file:
            #     continue
        """
        cyl_initendval = pd.DataFrame(columns=['END_CAPACITY', 'DISCHG_ENDENERGY', 'DISCHG_INITVOL', 'DISCHG_AVGVOL',
                                               'CHG_ENDCAPACITY', 'CHG_ENDENERGY', 'CHG_INITVOL', 'CHG_AVGVOL',
                                               'STAT_ENDVOL', 'DELTA_STATVOL',
                                               'MEAN_CHGVOL', 'MEAN_DISCHGVOL', 'RV_VOLTAGE', 'SV_VOLTAGE',
                                               'Q_EPSILON', 'CC_EPSILON',
                                               'CC_Q', 'CV_Q',
                                               'soc1000_du', 'soc2000_du', 'soc3000_du',
                                               'soc1000_avgu', 'soc2000_avgu', 'soc3000_avgu',
                                               'CYCLE_INT', 'CYCLE_NUM']
                                      )
        for i, file in enumerate(slice_filelist):
            file_path = os.path.join(sys_fold_path, file)
            df_cycl = pd.read_excel(file_path, sheet_name='cyl_data').reset_index()
            df_cycl.set_index(self.vars['STEPTIME'], inplace=True)
            df_cycl.drop_duplicates(keep='first', inplace=True)
            try:
                df_CD_R = df_cycl.loc[df_cycl[self.vars['STATUS']] == self.vars_CD['DC'], :] \
                    .sort_index(ascending=True)
                endindex = df_CD_R.index[-1]
            except Exception as err:
                logger.info('fold_cycle:{}_{} not enough data,index_warning: {}'.format(sys_fold_path, file, err))
                continue
            try:
                df_CD_R = df_cycl.loc[df_cycl.index <= endindex, :].sort_index()
                df_cycl_chg = df_CD_R.loc[df_CD_R[self.vars['STATUS']].isin([self.vars_CD['CC']]), :]
                df_cycl_chg_cv = df_CD_R.loc[df_CD_R[self.vars['STATUS']].isin([self.vars_CD['CV']]), :]
                df_cyl_static = df_CD_R.loc[df_CD_R[self.vars['STATUS']].isin([self.vars_CD['REST']]), :]
                df_cycl_dischg = df_CD_R.loc[df_CD_R[self.vars['STATUS']].isin([self.vars_CD['DC']]), :]
                mark_index = df_cycl_dischg.index[-1]
            except Exception as err:
                logger.info('last cycle mark_index not enough data warning :{}'.format(err))
                continue
            # if self.MODEL_ID.startswith('NCM5.0'):
            #     cycleid_num = int(file.split('.')[0])
            # elif self.MODEL_ID.startswith('LFP'):
            #     cycleid_num = df_cycl['CYCLE_INT'].values[-1]
            try:
                cycleid_num = int(file.split('.')[0])
                mean_chgu = df_cycl_chg[self.vars['ENERGY']].values[-1] / df_cycl_chg[self.vars['CAPACITY']].values[-1]
                mean_dischgu = df_cycl_dischg[self.vars['ENERGY']].values[-1] / \
                               df_cycl_dischg[self.vars['CAPACITY']].values[-1]

                df_q1000 = df_cycl_chg[df_cycl_chg[self.vars['CAPACITY']] <= 1000][self.vars['VOLTAGE']]
                soc_diff_0_1000 = df_q1000.values[-1] - df_q1000.values[0]

                df_q2000 = df_cycl_chg[df_cycl_chg[self.vars['CAPACITY']] <= 2000][self.vars['VOLTAGE']]
                soc_diff_0_2000 = df_q2000.values[-1] - df_q2000.values[0]

                df_q3000 = df_cycl_chg[df_cycl_chg[self.vars['CAPACITY']] <= 3000][self.vars['VOLTAGE']]
                soc_diff_0_3000 = df_q3000.values[-1] - df_q3000.values[0]

                soc_avg_0_1000 = df_q1000.mean()
                soc_avg_0_2000 = df_q2000.mean()
                soc_avg_0_3000 = df_q3000.mean()

                record_val = \
                    [df_cycl_dischg[self.vars['CAPACITY']].values[-1],
                     df_cycl_dischg[self.vars['ENERGY']].values[-1],
                     df_cycl_dischg[self.vars['VOLTAGE']].values[0],
                     df_cycl_dischg[self.vars['VOLTAGE']].values.mean(),

                     df_cycl_chg[self.vars['CAPACITY']].values[-1],
                     df_cycl_chg[self.vars['ENERGY']].values[-1],
                     df_cycl_chg[self.vars['VOLTAGE']].values[0],
                     df_cycl_chg[self.vars['VOLTAGE']].values.mean(),

                     df_cyl_static[self.vars['VOLTAGE']].values[-1],
                     df_cyl_static[self.vars['VOLTAGE']].values[-1] -
                     df_cycl_dischg[self.vars['VOLTAGE']].values[0],

                     mean_chgu, mean_dischgu, (mean_chgu - mean_dischgu) / 2, (mean_chgu + mean_dischgu) / 2,
                     df_cycl_dischg[self.vars['CAPACITY']].values[-1] /
                     df_cycl_chg[self.vars['CAPACITY']].values[-1],

                     df_cycl_chg[self.vars['CAPACITY']].values[-1] /
                     (df_cycl_chg[self.vars['CAPACITY']].values[-1] + df_cycl_chg_cv[self.vars['CAPACITY']].values[-1]),

                     df_cycl_chg[self.vars['CAPACITY']].values[-1],
                     df_cycl_chg_cv[self.vars['CAPACITY']].values[-1],

                     soc_diff_0_1000, soc_diff_0_2000, soc_diff_0_3000,
                     soc_avg_0_1000, soc_avg_0_2000, soc_avg_0_3000,

                     cycleid_num,
                     file.split('.')[0]
                     ]
                cyl_initendval.loc[mark_index, :] = record_val
                df_cycl_chg = self.dqdv_dvdq_fun(df_cycl_chg)
                df_cycl_dischg = self.dqdv_dvdq_fun(df_cycl_dischg)
                df_cycl = pd.concat([df_cycl_chg, df_cycl_dischg], axis=0)
                save_file_dval = os.path.join(save_path, file)
                df_cycl.to_excel(save_file_dval, sheet_name='cyl_data')
            except Exception as err:
                logger.info('sys_cellid:{}, err:{}'.format(file, err))

        return cyl_initendval

    def gen_count_features(self, RUN_MODE):
        """
        output: final data by  sys_batch|cellid|*.xlsx

        1. gen count features into data_join_path
        2. gen dqdq and dqdv time series into plt_data_path

        """
        logger.info('gen_count_features ing')
        mode_path = os.path.join(self.cycle_data_path, RUN_MODE)
        for sys_fold in os.listdir(mode_path):
            sys_fold_path = os.path.join(mode_path, sys_fold)
            for cell_fold in os.listdir(sys_fold_path):
                cell_fold_path = os.path.join(sys_fold_path, cell_fold)
                file_list = os.listdir(cell_fold_path)
                file_list.sort()
                save_plt_data_path = os.path.join(self.plt_data_path, RUN_MODE, sys_fold, cell_fold)
                if not os.path.exists(save_plt_data_path):
                    os.makedirs(save_plt_data_path)
                end_v_data_join_path = os.path.join(self.data_join_path, RUN_MODE, sys_fold, cell_fold)
                if not os.path.exists(end_v_data_join_path):
                    os.makedirs(end_v_data_join_path)

                asmslice_filelist = [file_list[i:i + self.MAX_SLICE_PAREL] for i in
                                     range(0, len(file_list), self.MAX_SLICE_PAREL)]
                logger.info('fold_file:{} cut sliceNum:{}'.format(sys_fold, len(asmslice_filelist)))
                pool = Pool(self.POOL_NUM)
                res_asyc = []
                for k, slice_filelist in enumerate(asmslice_filelist):
                    # res_asyc.append(self.gen_count_features_fun(file_fold, slice_filelist, k, save_plt_data_path))
                    res_asyc.append(
                        pool.apply_async(self.gen_count_features_fun, args=(cell_fold_path,
                                                                            slice_filelist,
                                                                            k,
                                                                            save_plt_data_path,)))
                res_df = []
                for res in res_asyc:
                    try:
                        res_df.append(res.get())
                    except Exception as err:
                        logger.info('errors:{}'.format(err))
                pool.close()
                pool.join()
                cyl_initendval = pd.concat(res_df, axis=0)
                cyl_initendval.to_excel(os.path.join(end_v_data_join_path, 'endq_asm.xlsx'), sheet_name='cyl_data')
                logger.info('fold_file:{} done\n'.format(sys_fold))
        logger.info('gen_count_features done')


def run_lfp_small():
    lfp = LoadSmall()
    lfp.split_cycle('TRAIN')
    lfp.split_cycle('PRED')
    lfp.gen_count_features('TRAIN')
    lfp.gen_count_features('PRED')


if __name__ == "__main__":
    run_lfp_small()
