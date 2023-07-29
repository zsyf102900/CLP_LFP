# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     parse_data_zz
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


class NCMLoadSmalltmp(ConfParams):
    """
    class LFPLoadSmall:
                     split cycle data ,
                     generate dqdv  count features

    Atttibutes:
        split_cycle:
        gen_pltdata:

    """

    def __init__(self):
        ConfParams.__init__(self)

    def cycle_fun(self, chg_status, dump_cycle_path, cycle_num_p, df):
        """
        1. charge state: CC:2498.5 mAH + CV:4.2V
        2. discharge state:CC:5000 mAH
        """
        for cycle_num in cycle_num_p:
            df_cycl = df[df[self.vars['CYCLESID']] == cycle_num]
            df_cycl[self.vars['STEPTIME']] = df_cycl[self.vars['STEPTIME']].apply(lambda x: str(x))
            if chg_status in ('charge'):
                try:
                    cccv_index = (df_cycl[self.vars['VOLTAGE']] > self.proc_parms['CV_VOLTAGE']) & \
                                 (df_cycl[self.vars['STEPTIME']] == '00:00:00')
                    cv_time = df_cycl[cccv_index].index[0]
                    cv_index = (df_cycl.index >= cv_time)
                    df_cycl.loc[cv_index, self.vars['STEP']] = 'CV'
                    df_cycl.loc[~cv_index, self.vars['STEP']] = 'CC'
                    df_cycl = df_cycl[df_cycl[self.vars['STEP']] == 'CC']
                except Exception as err:
                    logger.info("file:{} cycle:{} err:{}".format(dump_cycle_path, cycle_num, err))

            df_cycl.sort_index(ascending=True, inplace=True)
            df_cycl['dv'] = df_cycl[self.vars['VOLTAGE']].diff()
            df_cycl['dq'] = df_cycl[self.vars['CAPACITY']].diff()
            df_cycl.dropna(how='any', inplace=True)
            df_cycl['dqdv'] = df_cycl['dq'] / df_cycl['dv']
            df_cycl['dqdv'] = df_cycl['dq'] / df_cycl['dv']

            df_cycle_p = df_cycl.loc[:, ['dv', 'dq', 'dqdv']].rolling(window=10, min_periods=3).mean()
            df_cycle_p['dqdv_roll'] = df_cycle_p['dq'] / df_cycle_p['dv']
            df_cycle_p['dvdq_roll'] = df_cycle_p['dv'] / df_cycle_p['dq']
            df_cycle_p.rename(columns={'dv': 'dv_roll', 'dq': 'dq_roll'}, inplace=True)
            df_cycl = df_cycl.merge(df_cycle_p, how='left', left_index=True, right_index=True)
            df_cycl.dropna(how='any', inplace=True)
            dump_cycl_name = os.path.join(dump_cycle_path, str(cycle_num) + '.csv')
            df_cycl.to_csv(dump_cycl_name, sep=',', encoding='utf_8_sig', index=True)
            print(cycle_num)

    def set_cyclesid1(self, chg_status):
        """
        1.set cycles numbers ID
        2.split cycle data by cycleID
        old:
            1.filter each step capacity ==0
            2. set step init cycles NUMBERS ID
            3. join init cycles NUMBERS ID
            4. fill all cycles NUMBERS ID
        """
        for file in os.listdir(os.path.join(self.src_path, chg_status)):
            if not file.endswith('.xlsx'):
                continue
            path_file_name = os.path.join(self.src_path, chg_status, file)
            df = pd.read_excel(path_file_name, self.sheet_name)

            df[self.vars['STEPTIME']] = df[self.vars['STEPTIME']].map(self.cycle_dtime_mapfun)
            df[self.vars['STEPTIME']] = df[self.vars['STEPTIME']].str[1]
            # df = df.set_index([self.vars['CYCLESID'], self.vars['STEPTIME']])
            # # df.index.name = self.vars['STEPTIME']
            cycle_names = df[self.vars['CYCLESID']].unique()
            cycle_list = [cycle_names[i:i + self.MAX_SLICE_PAREL] for i in
                          range(0, len(cycle_names), self.MAX_SLICE_PAREL)]
            pool = Pool(6)

            for cycle_num_p in cycle_list:
                dump_cycle_path = os.path.join(self.cycle_data_path, chg_status, file)
                if not os.path.exists(dump_cycle_path):
                    os.makedirs(dump_cycle_path)
                self.cycle_fun(dump_cycle_path, cycle_num_p, df)
                # pool.apply_async(self.cycle_fun, args=(dump_cycle_path, cycle_num_p, df,))

                print()

    def split_cycle(self, RUN_MODE):
        """
        1. split into separate cycle file

        """
        raw_path = os.path.join(self.raw_data_path, RUN_MODE)
        for file in os.listdir(raw_path):
            if not file.endswith('.xlsx'):
                continue
            file_path = os.path.join(raw_path, file)
            logger.info('load file:{}'.format(file))
            df = pd.read_excel(file_path, sheet_name='记录层', usecols=list(self.vars.values()))
            logger.info('load file:{} done'.format(file))
            cylfile_path = os.path.join(self.cycle_data_path, RUN_MODE, file)
            if not os.path.exists(cylfile_path):
                os.makedirs(cylfile_path)
            cycle_all = df[self.vars['CYCLESID']].unique()
            cyl_slice = [cycle_all[i:i + self.MAX_SLICE_PAREL] for i in range(0, len(cycle_all), self.MAX_SLICE_PAREL)]
            df = df.set_index([self.vars['CYCLESID']])
            pool = Pool(self.POOL_NUM)
            for cyl_slc in cyl_slice:
                # self.split_cyl_fun(df, cyl_slc, cylfile_path)
                pool.apply_async(self.split_cyl_fun, args=(df, cyl_slc, cylfile_path,))
            pool.close()
            pool.join()
        logger.info('split_cycle done ')

    def dqdv_fun(self, cycle_file_p, chg_sts_path):
        for file in cycle_file_p:
            if not file.endswith('.csv'):
                continue
            file_path = os.path.join(chg_sts_path, file)
            df_p = pd.read_csv(file_path, sep=',')

    def dqdv_curve(self, chg_type):
        pool = Pool(6)
        chg_sts_path = os.path.join(self.cycle_data_path, chg_type)
        for cell_file in os.listdir(chg_sts_path):
            cycle_names = os.listdir(os.path.join(chg_sts_path, cell_file))
            cycle_list = [cycle_names[i:i + self.MAX_SLICE_PAREL] for i in
                          range(0, len(cycle_names), self.MAX_SLICE_PAREL)]
            for cycle_file_p in cycle_list:
                self.dqdv_fun(cycle_file_p, chg_sts_path)


class NCMCycleID(ConfParams):

    def __init__(self):
        ConfParams.__init__(self)
        self.tmp_drop_cols = ['Unnamed: 9', '比容量/mAh/g', '比能量/Wh/kg']

    @staticmethod
    def cycle_dtime_mapfun(x):
        """
        for  src_data stored in dateformat
        1. first day format: %H:%M:%S (datetime)
            Nth day format: n-%H:%M:%S (string)
        2.return  absoulte datetime

        """
        if isinstance(x, datetime.time):
            date_t = datetime.datetime.strptime(str(x), '%H:%M:%S') + datetime.timedelta(days=0)
        else:
            date_t = datetime.datetime.strptime(x.split('-')[1], '%H:%M:%S') + datetime.timedelta(
                days=int(x.split('-')[0]))
        return date_t

    def set_cyclesid(self, RUN_MODE):
        """
        define full cycles：CC-CV-DC-REST
        set cycles numbers ID
        1. filter each step capacity ==0
        2. set step init cycles NUMBERS ID
        3. join init cycles NUMBERS ID
        4. fill all cycles NUMBERS ID
        """

        for file in os.listdir(os.path.join(self.raw_srcdata_path, RUN_MODE)):
            if not (file.endswith('.xls') or file.endswith('.xlsx')):
                continue
            path_file_name = os.path.join(self.raw_srcdata_path, RUN_MODE, file)
            df_info = pd.ExcelFile(path_file_name)
            columns_col = [pd.read_excel(df_info, sheet).columns.tolist()
                           for sheet in df_info.sheet_names if sheet == self.sheet_src][0]
            try:
                mg_data = np.vstack([pd.read_excel(df_info, sheet).values
                                     for sheet in df_info.sheet_names if sheet.startswith(self.sheet_src)])
            except Exception as err:
                logger.warning('sysid:{},file:{} read errors'.format(RUN_MODE, file))
                continue
            df = pd.DataFrame(data=mg_data, columns=columns_col)
            drop_cols = [col for col in df.columns if 'Unnamed' in col]
            df = df.drop(columns=drop_cols, axis=1)
            logger.info('mg size:{}'.format(df.shape))
            df[self.vars['STEPTIME']] = df[self.vars['STEPTIME']].map(self.cycle_dtime_mapfun)
            # df.set_index([self.vars['STEPTIME'], self.vars['STEPTIME']], inplace=True)
            df.set_index(self.vars['STEPTIME'], inplace=True)
            logger.info('split STEPTIME')
            # filter qcapacity==0
            df_p = df.loc[:, [self.vars['STATUS'], self.vars['CAPACITY']]]
            df_qzeros = df_p[(df_p[self.vars['CAPACITY']] == 0) & (df_p[self.vars['STATUS']] == self.vars_CD['CC'])]. \
                sort_index(ascending=True, axis=0)
            # drop first cycles
            # set qcapacity==0, cyclesID
            # drop capacity,keep CYCLESID only
            cycleID_list = np.arange(2, df_qzeros.shape[0] + 2, 1)
            df_qzeros[self.vars['CYCLESID']] = cycleID_list
            df_qzeros.drop(columns=[self.vars['STATUS'], self.vars['CAPACITY']], axis=1, inplace=True)
            # JOIN cyclesID df:109355 ,df_raw:109379
            df_raw = df.merge(df_qzeros, how='left', left_index=True, right_index=True)
            # df_raw = df_raw.drop(self.tmp_drop_cols, axis=1)
            # fill all cyclesID, Add CC+CV state, 奇数-偶数循环填充
            k_step = 2
            cycleID_slice = [cycleID_list[i:i + k_step] for i in range(0, len(cycleID_list))]
            for id_slice in cycleID_slice:
                if len(id_slice) == 2:
                    try:
                        df_t = df_qzeros.loc[df_qzeros[self.vars['CYCLESID']].isin(id_slice), :]
                        time_stamp, cylID = df_t.index, df_t[self.vars['CYCLESID']]
                        df_raw.loc[(df_raw.index >= time_stamp[0]) & (df_raw.index < time_stamp[1]),
                        self.vars['CYCLESID']] = cylID[0]
                    except Exception as err:
                        print(err)
                else:
                    try:
                        df_t = df_qzeros.iloc[-1:, :]
                        time_stamp, cylID = df_t.index, df_t[self.vars['CYCLESID']]
                        df_raw.loc[df_raw.index == time_stamp[0], self.vars['CYCLESID']] = cylID[0]
                    except Exception as err:
                        logger.info('sys_cellid:{},err:{}'.format(RUN_MODE + file, err))
                        continue
            df_raw.dropna(how='any', axis=0, inplace=True)
            df = df_raw.reset_index().sort_values(by=[self.vars['STEPTIME']],
                                                  ascending=True)
            # rename STEPTIME , keep in same name LFP NCM
            df = df.loc[:, list(self.vars.values())].set_index(self.vars['STEPTIME'])
            logger.info('file:{} add cyclesID DONE'.format(file))
            path_sv = os.path.join(self.raw_data_path, RUN_MODE)
            if not os.path.exists(path_sv):
                os.makedirs(path_sv)
            file = ['.'.join([file.split('.')[0], 'xlsx']) if file.endswith('xls') else file][0]
            path_file_name = os.path.join(path_sv, file)
            df.to_excel(path_file_name, sheet_name='记录')


    def run_batch(self):
        pool = Pool(40)
        fold_list = os.listdir(self.raw_srcdata_path)
        for fold in fold_list:
            # if (fold.startswith('SC9') or fold.startswith('SC19') or fold.startswith('G6')
            #         or fold.startswith('SC14') or fold.startswith('SC17') or fold.startswith('SC18')
            #         or fold.startswith('I21')):
            #     continue
            # if not fold.startswith('G17-22C06TA'):
            #     continue
            # re_tmp = self.set_cyclesid(fold)
            pool.apply_async(self.set_cyclesid, args=(fold,))

        pool.close()
        pool.join()


if __name__ == "__main__":
    ncm = NCMCycleID()
    # ncm.set_cyclesid('TRAIN')
    # ncm.set_cyclesid('PRED')
    ncm.run_batch()
