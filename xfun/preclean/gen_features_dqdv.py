# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_features
   Description :
   Author :       ASCEND
   date：          2023-02-23
-------------------------------------------------
   Change Activity:
                   2023-02-23:
-------------------------------------------------
"""
__author__ = 'ASCEND'

import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from multiprocessing import Pool
import matplotlib.pyplot as plt
from findpeaks import findpeaks
from matplotlib.pyplot import cm
from utils.params_config import ConfPath, ConfVars, ConstVars
from log_conf.logger import logger

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)


# ************************@@@@@***************************#
#   peak calcu
# ************************@@@@@***************************#

class ExtractFeaturesEC(ConfParams):
    """
    class ExtractFeaturesEC:
                        calcu dqdv features
    Attributes:
        get_peak_valley
          get_peak_fun
            calcu_peak_fun
    """

    def __init__(self):
        ConfParams.__init__(self)

    def minmax_pv_calcu(self, df_ewm, pv_type):
        x_vars_dict = {'dqdv': {'X': self.vars['VOLTAGE'], 'Y': 'dqdv_roll'},
                       'dvdq': {'X': self.vars['CAPACITY'], 'Y': 'dvdq_roll'},
                       }
        pv_vars = x_vars_dict[pv_type]

        df_x_ewm = df_ewm[pv_vars['X']]
        df_y_ewm = df_ewm[pv_vars['Y']]
        fp = findpeaks(method='topology', whitelist=['peak', 'valley'])
        df_res = fp.fit(df_y_ewm)
        # fp.plot()
        pv_res = df_res['df']
        index_peaks = pv_res.loc[pv_res['peak'], 'x'].values
        index_valleys = pv_res.loc[pv_res['valley'], 'x'].values
        df_p_asm = pd.DataFrame(columns=['PEAK', 'PEAK_X'])
        df_p_asm.loc[:, 'PEAK'] = df_y_ewm.values[index_peaks]
        df_p_asm.loc[:, 'PEAK_X'] = df_x_ewm.values[index_peaks]
        df_v_asm = pd.DataFrame(columns=['VALLEY', 'VALLEY_X'])
        df_v_asm.loc[:, 'VALLEY'] = df_y_ewm.values[index_valleys]
        df_v_asm.loc[:, 'VALLEY_X'] = df_x_ewm.values[index_valleys]
        return df_v_asm, df_p_asm

    def chg_dqdv_lfpsrc(self, df_p_asm, df_v_asm, df_ewm, file_path, cyl_num):
        try:

            # peak
            fst_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P1'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P1'][1]), :] \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            sec_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P2'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P2'][1]), :].sort_values(by='PEAK') \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            thd_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] <= self.DQDV_LMT['P3'][1]), :].sort_values(by='PEAK') \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            # valley
            fst_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V1'][0])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]
            sec_valley_p = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V2P'][0]) &
                                        (df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V2P'][1])] \
                               .sort_values(by='VALLEY', ascending=True).iloc[0, :]
            sec_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V2'][0]) &
                                      (df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V2'][1])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]
            thd_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V3'][1])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]

            df_c1 = df_ewm.loc[df_ewm[self.vars['VOLTAGE']] >= sec_valley_p['VALLEY_X'], :]
            df_c2 = df_ewm.loc[(df_ewm[self.vars['VOLTAGE']] >= sec_valley['VALLEY_X']) &
                               (df_ewm[self.vars['VOLTAGE']] <= sec_valley_p['VALLEY_X']), :]
            df_c3 = df_ewm.loc[(df_ewm[self.vars['VOLTAGE']] >= thd_valley['VALLEY_X']) &
                               (df_ewm[self.vars['VOLTAGE']] <= sec_valley['VALLEY_X']), :]

            u_start, u_end = df_ewm[self.vars['VOLTAGE']].values[0], df_ewm[self.vars['VOLTAGE']].values[-1],
        except Exception as err:
            logger.info('cyl_num:{},peak valley section errors {} :{}'.format(cyl_num, file_path, err))

        try:
            # integ areas
            fun1_itegr = interp1d(df_c1[self.vars['VOLTAGE']], df_c1['dqdv_roll'], kind='quadratic')
            fun2_itegr = interp1d(df_c2[self.vars['VOLTAGE']], df_c2['dqdv_roll'], kind='quadratic')
            fun3_itegr = interp1d(df_c3[self.vars['VOLTAGE']], df_c3['dqdv_roll'], kind='quadratic')
            area1 = self.simpson_integrate(fun1_itegr, sec_valley_p['VALLEY_X'] + 0.0025,
                                           fst_valley['VALLEY_X'] - 0.0025)
            area2 = self.simpson_integrate(fun2_itegr, sec_valley['VALLEY_X'] + 0.0025,
                                           sec_valley_p['VALLEY_X'] - 0.0025)
            area3 = self.simpson_integrate(fun3_itegr, thd_valley['VALLEY_X'] + 0.0025, sec_valley['VALLEY_X'] - 0.0025)
        except Exception as err:
            logger.info('cyl_num:{},peak valley section integration errors {} :{}'.format(cyl_num, file_path, err))

        df_pv_row_chg = pd.DataFrame(
            columns=['fst_peak', 'sec_peak', 'thd_peak',
                     'fst_peak_x', 'sec_peak_x', 'thd_peak_x',
                     'fst_peak_x_diff_st', 'fst_peak_x_diff_end',
                     'sec_peak_x_diff_st', 'sec_peak_x_diff_end',
                     'thd_peak_x_diff_st', 'thd_peak_x_diff_end',
                     'area_q1', 'area1_q2', 'area_q3',
                     ],
            data=np.array([fst_peak['PEAK'], sec_peak['PEAK'], thd_peak['PEAK'],
                           fst_peak['PEAK_X'], sec_peak['PEAK_X'], thd_peak['PEAK_X'],
                           fst_peak['PEAK_X'] - u_start, fst_peak['PEAK_X'] - u_end,
                           sec_peak['PEAK_X'] - u_start, sec_peak['PEAK_X'] - u_end,
                           thd_peak['PEAK_X'] - u_start, thd_peak['PEAK_X'] - u_end,
                           area1, area2, area3]).reshape(1, -1),
            index=[cyl_num.split('.')[0]])
        return df_pv_row_chg

    def chg_dqdv(self, df_p_asm, df_v_asm, df_ewm, file_path, cyl_num):
        try:
            # peak
            fst_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P1'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P1'][1]), :] \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            sec_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P2'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P2'][1]), :].sort_values(by='PEAK') \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]


        except Exception as err:
            logger.info('cyl_num:{},peak valley section errors {} :{}'.format(cyl_num, file_path, err))
        df_pv_row_chg = pd.DataFrame(
            columns=['fst_peak', 'sec_peak',
                     'fst_peak_x', 'sec_peak_x',
                     ],
            data=np.array([fst_peak['PEAK'], sec_peak['PEAK'],
                           fst_peak['PEAK_X'], sec_peak['PEAK_X']]).reshape(1, -1),
            index=[cyl_num.split('.')[0]])
        return df_pv_row_chg

    def dischg_dqdv(self, df_p_asm, df_v_asm, df_ewm, file_path, cyl_num):
        pass

    def chg_dqdv_ncx(self, df_p_asm, df_v_asm, df_ewm, file_path, cyl_num):

        try:
            # peak
            fst_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P1'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P1'][1]), :] \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            sec_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P2'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P2'][1]), :].sort_values(by='PEAK') \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]
            # valley
            fst_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V1'][0])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]
            sec_valley_p = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V2P'][0]) &
                                        (df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V2P'][1])] \
                               .sort_values(by='VALLEY', ascending=True).iloc[0, :]
            thd_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V3'][1])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]
            df_c1 = df_ewm.loc[df_ewm[self.vars['VOLTAGE']] >= sec_valley_p['VALLEY_X'], :]
            df_c23 = df_ewm.loc[(df_ewm[self.vars['VOLTAGE']] >= thd_valley['VALLEY_X']) &
                                (df_ewm[self.vars['VOLTAGE']] <= sec_valley_p['VALLEY_X']), :]

            df_c1.drop_duplicates(subset=[self.vars['VOLTAGE']], keep='first', inplace=True)
            df_c23.drop_duplicates(subset=[self.vars['VOLTAGE']], keep='first', inplace=True)

        except Exception as err:
            logger.info('cyl_num:{},peak valley section errors {} :{}'
                        .format(cyl_num, file_path.split('plt_data')[1], err))
        try:
            # integ areas
            fun1_itegr = interp1d(df_c1[self.vars['VOLTAGE']], df_c1['dqdv_roll'], kind='quadratic')
            fun2_itegr = interp1d(df_c23[self.vars['VOLTAGE']], df_c23['dqdv_roll'], kind='quadratic')
            area1 = self.simpson_integrate(fun1_itegr, sec_valley_p['VALLEY_X'] + 0.0025,
                                           fst_valley['VALLEY_X'] - 0.0025)
            area23 = self.simpson_integrate(fun2_itegr, thd_valley['VALLEY_X'] + 0.0025,
                                            sec_valley_p['VALLEY_X'] - 0.0025)
        except Exception as err:
            logger.info('cyl_num:{},peak valley section integration errors {}:{}'
                        .format(cyl_num,
                                file_path.split('plt_data')[1],
                                err
                                ))

        finally:

            df_pv_row_chg = pd.DataFrame(
                columns=['fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23'],
                data=np.array([fst_peak['PEAK'], sec_peak['PEAK'], fst_peak['PEAK_X'], sec_peak['PEAK_X'],
                               area1, area23]).reshape(1, -1),
                index=[cyl_num.split('.')[0]])
            return df_pv_row_chg

    def dischg_dqdv_ncx(self, df_p_asm, df_v_asm, df_ewm, file_path, cyl_num):

        try:
            # peak
            fst_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P1'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P1'][1]), :] \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            sec_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P2'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P2'][1]), :].sort_values(by='PEAK') \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            # valley
            fst_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V1'][0])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]
            sec_valley_p = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V2P'][0]) &
                                        (df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V2P'][1])] \
                               .sort_values(by='VALLEY', ascending=True).iloc[0, :]

            thd_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V3'][1])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]

            df_c1 = df_ewm.loc[df_ewm[self.vars['VOLTAGE']] >= sec_valley_p['VALLEY_X'], :]
            df_c23 = df_ewm.loc[(df_ewm[self.vars['VOLTAGE']] >= thd_valley['VALLEY_X']) &
                                (df_ewm[self.vars['VOLTAGE']] <= sec_valley_p['VALLEY_X']), :]

        except Exception as err:
            logger.info('cyl_num:{},peak valley section errors {} :{}'.format(cyl_num, file_path, err))

        try:
            # integ areas
            fun1_itegr = interp1d(df_c1[self.vars['VOLTAGE']], df_c1['dqdv_roll'], kind='quadratic')
            fun2_itegr = interp1d(df_c23[self.vars['VOLTAGE']], df_c23['dqdv_roll'], kind='quadratic')

            area1 = self.simpson_integrate(fun1_itegr, sec_valley_p['VALLEY_X'] + 0.0025,
                                           fst_valley['VALLEY_X'] - 0.0025)
            area23 = self.simpson_integrate(fun2_itegr, thd_valley['VALLEY_X'] + 0.0025,
                                            sec_valley_p['VALLEY_X'] - 0.0025)

        except Exception as err:
            logger.info('cyl_num:{},peak valley section integration errors {} :{}'.format(cyl_num, file_path, err))

        df_pv_row_chg = pd.DataFrame(
            columns=['fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23'],
            data=np.array([fst_peak['PEAK'], sec_peak['PEAK'], fst_peak['PEAK_X'], sec_peak['PEAK_X'],
                           area1, area23]).reshape(1, -1),
            index=[cyl_num.split('.')[0]])
        return df_pv_row_chg

    def dischg_dqdv_ncx_1C(self, df_p_asm, df_v_asm, df_ewm, file_path, cyl_num):

        try:
            # peak
            fst_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P1'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P1'][1]), :] \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            sec_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DQDV_LMT['P2'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DQDV_LMT['P2'][1]), :].sort_values(by='PEAK') \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            # valley
            fst_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V1'][0])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]
            sec_valley_p = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DQDV_LMT['V2P'][0]) &
                                        (df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V2P'][1])] \
                               .sort_values(by='VALLEY', ascending=True).iloc[0, :]

            thd_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] <= self.DQDV_LMT['V3'][1])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]

            df_c1 = df_ewm.loc[df_ewm[self.vars['VOLTAGE']] >= sec_valley_p['VALLEY_X'], :]
            df_c23 = df_ewm.loc[(df_ewm[self.vars['VOLTAGE']] >= thd_valley['VALLEY_X']) &
                                (df_ewm[self.vars['VOLTAGE']] <= sec_valley_p['VALLEY_X']), :]

        except Exception as err:
            logger.info('cyl_num:{},peak valley section errors {} :{}'.format(cyl_num, file_path, err))

        try:
            # integ areas
            fun1_itegr = interp1d(df_c1[self.vars['VOLTAGE']], df_c1['dqdv_roll'], kind='quadratic')
            fun2_itegr = interp1d(df_c23[self.vars['VOLTAGE']], df_c23['dqdv_roll'], kind='quadratic')

            area1 = self.simpson_integrate(fun1_itegr, sec_valley_p['VALLEY_X'] + 0.0025,
                                           fst_valley['VALLEY_X'] - 0.0025)
            area23 = self.simpson_integrate(fun2_itegr, thd_valley['VALLEY_X'] + 0.0025,
                                            sec_valley_p['VALLEY_X'] - 0.0025)

        except Exception as err:
            logger.info('cyl_num:{},peak valley section integration errors {} :{}'.format(cyl_num, file_path, err))

        df_pv_row_chg = pd.DataFrame(
            columns=['fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23'],
            data=np.array([fst_peak['PEAK'], sec_peak['PEAK'], fst_peak['PEAK_X'], sec_peak['PEAK_X'],
                           area1, area23]).reshape(1, -1),
            index=[cyl_num.split('.')[0]])
        return df_pv_row_chg

    def calcu_peak_chgfun(self, df_ewm, cyl_num, file_path):

        """
        1. calcu chg_peak valley
        2. calcu chg_peak valley
        # fst_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] <= 3.415) & (df_p_asm['PEAK_X'] >= 3.4), :].loc[:, 'PEAK']
        # sec_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= 3.365) & (df_p_asm['PEAK_X'] <= 3.385), :].loc[:, 'PEAK']
        # thd_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] <= 3.325), :].loc[:, 'PEAK']
        """
        df_v_asm, df_p_asm = self.minmax_pv_calcu(df_ewm, 'dqdv')
        if self.MODEL_ID.startswith('LFP'):
            df_pv_row_chg = self.chg_dqdv(df_p_asm, df_v_asm, df_ewm, file_path, cyl_num)

            return df_pv_row_chg
        elif self.MODEL_ID.startswith('NC'):
            df_pv_row_chg = self.chg_dqdv_ncx(df_p_asm, df_v_asm, df_ewm, file_path, cyl_num)
            return df_pv_row_chg

    def chg_deltadvdq_lfp(self, df_p_asm, df_v_asm, df_ewm, file_path, cyl_num):
        pass

    def chg_deltadvdq_ncx(self, df_p_asm, df_v_asm, df_ewm, file_path, cyl_num):

        try:

            fst_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DVDQ_LMT['P1'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DVDQ_LMT['P1'][1])] \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            sec_peak = df_p_asm.loc[(df_p_asm['PEAK_X'] >= self.DVDQ_LMT['P2'][0]) &
                                    (df_p_asm['PEAK_X'] <= self.DVDQ_LMT['P2'][1])] \
                           .sort_values(by='PEAK', ascending=False).iloc[0, :]

            v2p_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DVDQ_LMT['V2P'][0]) &
                                      (df_v_asm['VALLEY_X'] <= self.DVDQ_LMT['V2P'][1])] \
                             .sort_values(by='VALLEY', ascending=True).iloc[0, :]

            v2_valley = df_v_asm.loc[(df_v_asm['VALLEY_X'] >= self.DVDQ_LMT['V2'][0]) &
                                     (df_v_asm['VALLEY_X'] <= self.DVDQ_LMT['V2'][1])] \
                            .sort_values(by='VALLEY', ascending=True).iloc[0, :]

            delta_pk1 = fst_peak['PEAK_X'] - v2p_valley['VALLEY_X']
            delta_pk2 = sec_peak['PEAK_X'] - v2_valley['VALLEY_X']
            df_delta_pv_chg = pd.DataFrame(
                columns=['delta_pv1', 'delta_pv2'],
                data=np.array([delta_pk1, delta_pk2]).reshape(1, -1),
                index=[cyl_num.split('.')[0]])

        except Exception as err:
            logger.info('cyl_num:{}, delta_pv calcu errors {} :{}'.format(cyl_num, file_path, err))

        return df_delta_pv_chg

    def calcu_detla_dvdq(self, df_ewm, cyl_num, file_path):
        qlm = self.DVDQ_LMT['QLIM']
        index_q = (df_ewm[self.vars['CAPACITY']] > qlm[0]) & \
                  (df_ewm[self.vars['CAPACITY']] < qlm[1])
        df_ewm = df_ewm[index_q]

        df_v_asm, df_p_asm = self.minmax_pv_calcu(df_ewm, 'dvdq')
        if self.MODEL_ID.startswith('LFP'):
            df_pv_row_chg = self.chg_deltadvdq_lfp(df_p_asm, df_v_asm, df_ewm, file_path, cyl_num)
            return df_pv_row_chg

        elif self.MODEL_ID.startswith('NC'):
            df_pv_row_chg = self.chg_deltadvdq_ncx(df_p_asm, df_v_asm, df_ewm, file_path, cyl_num)
            return df_pv_row_chg
        pass

    def calcu_peak_dischgfun(self, df_ewm, cyl_num, file_path):
        """
        copy above:calcu_peak_chgfun
        1. calcu dischg_peak valley
        2. calcu dischg_peak valley

        """
        df_v_asm, df_p_asm = self.minmax_pv_calcu(df_ewm, 'dqdv')
        if self.MODEL_ID.startswith('LFP'):
            df_pv_row_chg = self.dischg_dqdv(df_p_asm, df_v_asm, df_ewm, file_path, cyl_num)
            return df_pv_row_chg
        elif (self.MODEL_ID.startswith('NC') and self.CC_RAT == '0.5C'):
            df_pv_row_chg = self.dischg_dqdv_ncx(df_p_asm, df_v_asm, df_ewm, file_path, cyl_num)
            return df_pv_row_chg

        elif (self.MODEL_ID.startswith('NC') and self.CC_RAT == '1C'):
            df_pv_row_chg = self.dischg_dqdv_ncx(df_p_asm, df_v_asm, df_ewm, file_path, cyl_num)
            return df_pv_row_chg

    @staticmethod
    def simpson_integrate(func, a, b):
        return (b - a) * (func(a) + func(b) + 4 * func((a + b) / 2)) / 6

    def get_peak_fun(self, sys_fold_path, fold, cyl_f_slice, RUN_MODE):
        """

        1.curve fit methods: fit 拟合 or interpolation 插值
        2.fit defects：smooth but signals trend distortion
        3.interpolation：not smooth but repsents real signal trend
        4.dqdv peak shift errors
        """
        df_icyl_pv = []
        for m, file in enumerate(cyl_f_slice):
            file_path = os.path.join(sys_fold_path, file)
            if not file.endswith('.xlsx'):
                continue
            df = pd.read_excel(file_path, sheet_name='cyl_data')
            df_chg = df[df[self.vars['STATUS']] == self.vars_CD['CC']]
            if df_chg.shape[0] < 10:
                logger.info('{} CC shape <10'.format(RUN_MODE))
                continue
            df_dischg = df[df[self.vars['STATUS']] == self.vars_CD['DC']]
            # logger.info('   {}:dqdv ewm smoothing begins'.format(file))

            df_chg_ewm_dqdv = df_chg['dqdv_roll'].rolling(window=10, min_periods=3).mean()
            df_dischg_ewm_dqdv = df_dischg['dqdv_roll'].rolling(window=10, min_periods=3).mean()

            df_chg_ewm_dqdv_ts = pd.concat([df_chg[self.vars['VOLTAGE']], df_chg_ewm_dqdv], axis=1)
            df_dischg_ewm = pd.concat([df_dischg[self.vars['VOLTAGE']], df_dischg_ewm_dqdv], axis=1)

            df_chg_ewm_dqdv_ts.dropna(how='any', axis=0, inplace=True)
            df_dischg_ewm.dropna(how='any', axis=0, inplace=True)
            try:
                df_pv_row_chg_dqdv = self.calcu_peak_chgfun(df_chg_ewm_dqdv_ts, file, file_path)
            except Exception as err:
                logger.info('file_path:{},dqdv err:{}'.format(fold + file, err))
                continue

            df_chg_ewm_dvdq = df_chg['dvdq_roll'].rolling(window=10, min_periods=3).mean()
            df_chg_ewm_dvdq_ts = pd.concat([df_chg[self.vars['CAPACITY']], df_chg_ewm_dvdq], axis=1)
            df_chg_ewm_dvdq_ts.dropna(how='any', axis=0, inplace=True)

            try:
                df_deltapeak_chg_dvdq = self.calcu_detla_dvdq(df_chg_ewm_dvdq_ts, file, file_path)
            except Exception as err:
                logger.info('file_path:{},dvdq delta_peak err:{}'.format(fold + file, err))
                continue

            df_pv_row_chg = pd.concat([df_pv_row_chg_dqdv, df_deltapeak_chg_dvdq], axis=1)
            # dischg features
            # df_pv_row_dischg = self.calcu_peak_dischgfun(df_dischg_ewm.loc[:, [self.vars['VOLTAGE'], 'dqdv_roll']],
            #                                              file, file_path)
            # df_pv_row = pd.concat([df_pv_row_chg, df_pv_row_dischg], axis=1)
            # df_icyl_pv.append(df_pv_row)
            df_icyl_pv.append(df_pv_row_chg)
            print()
            # logger.info('   {}:peak_valley calcu done'.format(file))

        return df_icyl_pv

    def get_peak_valley_dbg(self, RUN_MODE):
        """
          1. calcu peak
          2. attentions:
             cycle peak shift may cause calcu errors

        """
        mode_path = os.path.join(self.plt_data_path, RUN_MODE)
        for fold in os.listdir(mode_path):
            fold_path = os.path.join(mode_path, fold)
            file_list = os.listdir(fold_path)
            file_list = [int(file_list[k].split('.')[0]) for k in range(0, len(file_list))]
            file_list.sort()
            cyl_filelist = [str(file_list[k]) + '.xlsx' for k in range(0, len(file_list))] \
                [self.dqdv_initcycle:self.dqdv_maxcycle]
            cyl_f_slice = [cyl_filelist[i:i + self.MAX_SLICE_PAREL] for i in
                           range(0, len(cyl_filelist), self.MAX_SLICE_PAREL)]
            logger.info('peak_valley {} calcu begin'.format(fold))
            df_icyl_pv_asyc = []
            for m, slice_file in enumerate(cyl_f_slice):
                res = self.get_peak_fun(fold_path, fold, slice_file, RUN_MODE, )
                df_icyl_pv_asyc.append(res)
                print()

    def get_peak_valley(self, RUN_MODE):
        """
        output: final data by  sys_batch|cellid|*.xlsx

        1. gen  dqdq and dqdv  x-yheight-area count features into data_join_path

        content:
          1. calcu peak
          2. attentions:
             cycle peak shift may cause calcu errors

        """
        mode_path = os.path.join(self.plt_data_path, RUN_MODE)
        for sys_fold in os.listdir(mode_path):
            sys_fold_path = os.path.join(mode_path, sys_fold)
            for cell_fold in os.listdir(sys_fold_path):
                cell_fold_path = os.path.join(sys_fold_path, cell_fold)
                file_list = os.listdir(cell_fold_path)

                file_list = [int(file_list[k].split('.')[0]) for k in range(0, len(file_list))]
                file_list.sort()
                cyl_filelist = [str(file_list[k]) + '.xlsx' for k in range(0, len(file_list))][:self.dqdv_maxcycle]
                cyl_f_slice = [cyl_filelist[i:i + self.MAX_SLICE_PAREL] for i in
                               range(0, len(cyl_filelist), self.MAX_SLICE_PAREL)]
                logger.info('peak_valley {} calcu begin'.format(sys_fold))
                pool = Pool(self.POOL_NUM)
                df_icyl_pv_asyc = []
                for m, slice_file in enumerate(cyl_f_slice):
                    # self.get_peak_fun(fold_path, fold, slice_file, RUN_MODE, )
                    res = pool.apply_async(self.get_peak_fun, args=(cell_fold_path, sys_fold, slice_file, RUN_MODE,))
                    df_icyl_pv_asyc.append(res)
                pool.close()
                pool.join()
                df_icyl_pv = []
                for res in df_icyl_pv_asyc:
                    df_icyl_pv.append(res.get())
                df_icyl_pv = sum(df_icyl_pv, [])
                logger.info('peak_valley {}:{} calcu done'.format(sys_fold, cell_fold))
                df_icyl_pv = pd.concat(df_icyl_pv, axis=0)
                pv_row_path = os.path.join(self.data_join_path, RUN_MODE, sys_fold, cell_fold)
                if not os.path.exists(pv_row_path):
                    os.makedirs(pv_row_path)
                df_icyl_pv.to_excel(os.path.join(pv_row_path, 'count_pv.xlsx'),
                                    sheet_name='cyl_data')
                logger.info('pv_file:{} save done'.format(sys_fold.split('.')[0]))

    def get_peak_dqdv_review(self, RUN_MODE):
        tr_pred_path = os.path.join(self.plt_data_path, RUN_MODE)
        for sys_fold in os.listdir(tr_pred_path):
            sys_fold_path = os.path.join(tr_pred_path, sys_fold)
            for cell_fold in os.listdir(sys_fold_path):
                cell_fold_path = os.path.join(sys_fold_path, cell_fold)

                file_list = os.listdir(cell_fold_path)
                file_list = [int(file_list[k].split('.')[0]) for k in range(0, len(file_list))]
                file_list.sort()
                file_list = file_list[: 800]
                file_list = [str(file_list[k]) + '.xlsx' for k in range(10, len(file_list), 10)]
                plt.figure(figsize=(35, 20), dpi=self.dpi)
                ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=1)
                ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1, rowspan=1)

                color = cm.rainbow(np.linspace(0, 1, len(file_list)))
                for i, file in enumerate(file_list):
                    file_path = os.path.join(cell_fold_path, file)
                    df = pd.read_excel(file_path, sheet_name='cyl_data')
                    df_chg = df[df[self.vars['STATUS']] == self.vars_CD['CC']]
                    df_dischg = df[df[self.vars['STATUS']] == self.vars_CD['DC']]
                    # df_chg = df_chg[(df_chg[self.vars['VOLTAGE']] >= 3.15) &
                    #                 (df_chg[self.vars['VOLTAGE']] <= 3.6)
                    #                 ]
                    # df_dischg = df_dischg[(df_dischg[self.vars['VOLTAGE']] >= 2.95) &
                    #                       (df_dischg[self.vars['VOLTAGE']] <= 3.28)]

                    df_chg_ewm = df_chg.loc[:, [self.vars['VOLTAGE'], 'dqdv_roll']] \
                        .rolling(window=3, min_periods=1).mean()

                    df_dischg_ewm = df_dischg.loc[:, [self.vars['VOLTAGE'], 'dqdv_roll']] \
                        .rolling(window=3, min_periods=1).mean()

                    df_chg_ewm = df_chg_ewm.iloc[:-3, :]
                    df_dischg_ewm = df_dischg_ewm.iloc[:-3, :]
                    df_chg_ewm.dropna(how='any', axis=0, inplace=True)
                    df_dischg_ewm.dropna(how='any', axis=0, inplace=True)

                    df_chg_u, df_dischg_u = df_chg[self.vars['VOLTAGE']], df_dischg[self.vars['VOLTAGE']]
                    df_chg_dqdv, df_dischg_dqdv = df_chg['dqdv_roll'], df_dischg['dqdv_roll']
                    df_chg_u_ewm, df_dischg_u_ewm = df_chg_ewm[self.vars['VOLTAGE']], df_dischg_ewm[
                        self.vars['VOLTAGE']]
                    df_chg_dqdv_ewm, df_dischg_dqdv_ewm = df_chg_ewm['dqdv_roll'], df_dischg_ewm['dqdv_roll']

                    ax1.plot(df_chg_u_ewm, df_chg_dqdv_ewm, c=color[i], label=file.split('.')[0])
                    ax2.plot(df_dischg_u_ewm, df_dischg_dqdv_ewm, c=color[i], label=file.split('.')[0])

                    try:
                        fp = findpeaks(method='topology', whitelist=['peak', 'valley'])
                        try:
                            df_res = fp.fit(df_chg_dqdv_ewm)
                        # # fp.plot()
                        except Exception as err:
                            logger.info(
                                'chg fold:{} cyclenum:{},'
                                'peak-valley fit not enough data:{}'.format(sys_fold+cell_fold, file, err))

                        pv_res = df_res['df']
                        index_peaks = pv_res.loc[pv_res['peak'], 'x'].values
                        index_valleys = pv_res.loc[pv_res['valley'], 'x'].values
                        ax1.plot(df_chg_u_ewm.values[index_peaks], df_chg_dqdv_ewm.values[index_peaks],
                                 'g*', markersize=10)
                        ax1.plot(df_chg_u_ewm.values[index_valleys], df_chg_dqdv_ewm.values[index_valleys],
                                 'co', markersize=10)
                        fp = findpeaks(method='topology', whitelist=['peak', 'valley'])
                        try:
                            df_res = fp.fit(df_dischg_dqdv_ewm)
                        # # fp.plot()
                        except Exception as err:
                            logger.info('dischg fold:{} cyclenum:{},'
                                        'peak-valley fit errors:{}'.format(sys_fold+cell_fold, file, err))
                        pv_res = df_res['df']
                        index_peaks = pv_res.loc[pv_res['peak'], 'x'].values
                        index_valleys = pv_res.loc[pv_res['valley'], 'x'].values
                        ax2.plot(df_dischg_u_ewm.values[index_peaks], df_dischg_dqdv_ewm.values[index_peaks],
                                 'g*', markersize=10)
                        ax2.plot(df_dischg_u_ewm.values[index_valleys], df_dischg_dqdv_ewm.values[index_valleys],
                                 'co', markersize=10)
                    except Exception:
                        logger.info('dischg fold:{} cyclenum:{},'
                                    'not enough data '.format(sys_fold+cell_fold, file))
                        continue
                # range_list = list(np.linspace(3.2, 3.6, 25))
                # ax1.set_xticks(range_list)
                ax1.set_title(sys_fold+cell_fold, fontsize=24)
                ax1.legend(fontsize=18, loc='best')
                ax1.grid(linestyle=":", color="r")
                ax2.legend(fontsize=18, loc='best')
                # ax1.set_ylim((0, 7000))
                # plt.show()
                # plt.tight_layout()
                plt.savefig(os.path.join(self.path_png_plt,
                                         RUN_MODE + '_' + sys_fold+cell_fold + '_pv_dqdv_demo.png'),
                            dpi=800)

    def get_peak_dvdq_review(self, RUN_MODE):
        tr_pred_path = os.path.join(self.plt_data_path, RUN_MODE)
        for fold in os.listdir(tr_pred_path):
            fold_path = os.path.join(tr_pred_path, fold)
            file_list = os.listdir(fold_path)
            file_list = [int(file_list[k].split('.')[0]) for k in range(0, len(file_list))]
            file_list.sort()
            file_list = file_list[: 600]
            file_list = [str(file_list[k]) + '.xlsx' for k in range(10, len(file_list), 25)]
            plt.figure(figsize=(15, 20), dpi=self.dpi)
            ax3 = plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=1)
            ax4 = plt.subplot2grid((2, 1), (1, 0), colspan=1, rowspan=1)
            color = cm.rainbow(np.linspace(0, 1, len(file_list)))

            for i, file in enumerate(file_list):
                file_path = os.path.join(fold_path, file)
                df = pd.read_excel(file_path, sheet_name='cyl_data')
                df_chg = df[df[self.vars['STATUS']] == self.vars_CD['CC']]
                df_dischg = df[df[self.vars['STATUS']] == self.vars_CD['DC']]

                df_chg_dvdq = df_chg['dvdq_roll'].rolling(window=3, min_periods=1).mean()
                df_dischg_dvdq = df_dischg['dvdq_roll'].rolling(window=3, min_periods=1).mean()

                df_chg_dvdq_ewm = pd.concat([df_chg[self.vars['CAPACITY']], df_chg_dvdq], axis=1)
                df_dischg_dvdq_ewm = pd.concat([df_dischg[self.vars['CAPACITY']], df_dischg_dvdq], axis=1)

                ax3.plot(df_chg_dvdq_ewm[self.vars['CAPACITY']], df_chg_dvdq_ewm['dvdq_roll'], c=color[i],
                         label=file.split('.')[0] + ':dvdq_roll')
                ax4.plot(df_dischg_dvdq_ewm[self.vars['CAPACITY']], df_dischg_dvdq_ewm['dvdq_roll'], c=color[i],
                         label=file.split('.')[0] + ':dvdq_roll')
                # chg dvdq
                try:
                    fp = findpeaks(method='topology', whitelist=['peak', 'valley'])
                    df_res = fp.fit(df_chg_dvdq)
                # # fp.plot()
                except Exception as err:
                    logger.info(
                        'chg fold:{} cyclenum:{},peak-valley fit not enough data:{}'.format(fold, file, err))

                pv_res = df_res['df']
                index_peaks = pv_res.loc[pv_res['peak'], 'x'].values
                index_valleys = pv_res.loc[pv_res['valley'], 'x'].values
                ax3.plot(df_chg_dvdq_ewm[self.vars['CAPACITY']].values[index_peaks],
                         df_chg_dvdq_ewm['dvdq_roll'].values[index_peaks],
                         'g*', markersize=10)
                ax3.plot(df_chg_dvdq_ewm[self.vars['CAPACITY']].values[index_valleys],
                         df_chg_dvdq_ewm['dvdq_roll'].values[index_valleys],
                         'co', markersize=10)

                # dischg dvdq
                try:
                    fp = findpeaks(method='topology', whitelist=['peak', 'valley'])
                    df_res = fp.fit(df_dischg_dvdq)
                # # fp.plot()
                except Exception as err:
                    logger.info('dischg fold:{} cyclenum:{},peak-valley fit errors:{}'.format(fold, file, err))
                pv_res = df_res['df']
                index_peaks = pv_res.loc[pv_res['peak'], 'x'].values
                index_valleys = pv_res.loc[pv_res['valley'], 'x'].values
                ax4.plot(df_dischg_dvdq_ewm[self.vars['CAPACITY']].values[index_peaks],
                         df_dischg_dvdq_ewm['dvdq_roll'].values[index_peaks],
                         'g*', markersize=10)
                ax4.plot(df_dischg_dvdq_ewm[self.vars['CAPACITY']].values[index_valleys],
                         df_dischg_dvdq_ewm['dvdq_roll'].values[index_valleys],
                         'co', markersize=10)
                # except Exception:
                #     logger.info('dischg fold:{} cyclenum:{},not enough data '.format(fold, file))

            # range_list = list(np.linspace(3.2, 3.6, 25))
            # ax1.set_xticks(range_list)
            ax3.set_title(fold, fontsize=24)
            ax3.legend(fontsize=18, loc='best')
            ax4.grid(linestyle=":", color="r")
            ax4.legend(fontsize=18, loc='best')
            # ax3.set_ylim((0.0001, 0.0003))
            # ax4.set_ylim((-0.0004, -0.0001))

            ax3.set_ylim((0, 0.005))
            ax4.set_ylim((-0.005, 0))
            # plt.show()
            # plt.tight_layout()
            plt.savefig(os.path.join(self.path_png_plt,
                                     RUN_MODE + '_' + fold.split('.')[0] + '_pv_dvdq_demo.png'),
                        dpi=800)

    def get_peak_review_asm(self, RUN_MODE):
        plt_step = 200
        mode_path = os.path.join(self.plt_data_path, RUN_MODE)
        plt.figure(figsize=(15, 20), dpi=self.dpi)
        ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=1)
        ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1, rowspan=1)
        for sys_fold in os.listdir(mode_path):
            sys_fold_path = os.path.join(mode_path, sys_fold)

            file_list = os.listdir(sys_fold_path)
            file_list = [int(file_list[k].split('.')[0]) for k in range(0, len(file_list))]
            file_list.sort()
            file_list = [str(file_list[k]) + '.xlsx' for k in range(10, len(file_list), plt_step)]
            color_rb = cm.rainbow(np.linspace(0, 1, len(file_list)))
            color_k = ['k'] * len(file_list)
            line_wdth = [2 if sys_fold.endswith('outlier') else 0.5][0]
            color = [color_k if sys_fold.endswith('outlier') else color_rb][0]

            for i, file in enumerate(file_list):
                file_path = os.path.join(sys_fold_path, file)
                df = pd.read_excel(file_path, sheet_name='cyl_data')
                df_chg = df[df[self.vars['STATUS']] == self.vars_CD['CC']]
                df_dischg = df[df[self.vars['STATUS']] == self.vars_CD['DC']]

                df_chg_ewm = df_chg.loc[:, [self.vars['VOLTAGE'], 'dqdv_roll']] \
                    .rolling(window=3, min_periods=1).mean()

                df_dischg_ewm = df_dischg.loc[:, [self.vars['VOLTAGE'], 'dqdv_roll']] \
                    .rolling(window=3, min_periods=1).mean()

                df_chg_ewm = df_chg_ewm.iloc[:-3, :]
                df_dischg_ewm = df_dischg_ewm.iloc[:-3, :]
                df_chg_ewm.dropna(how='any', axis=0, inplace=True)
                df_dischg_ewm.dropna(how='any', axis=0, inplace=True)

                df_chg_u, df_dischg_u = df_chg[self.vars['VOLTAGE']], df_dischg[self.vars['VOLTAGE']]
                df_chg_dqdv, df_dischg_dqdv = df_chg['dqdv_roll'], df_dischg['dqdv_roll']
                df_chg_u_ewm, df_dischg_u_ewm = df_chg_ewm[self.vars['VOLTAGE']], df_dischg_ewm[self.vars['VOLTAGE']]
                df_chg_dqdv_ewm, df_dischg_dqdv_ewm = df_chg_ewm['dqdv_roll'], df_dischg_ewm['dqdv_roll']

                # plot
                # ax1.plot(df_chg_u, df_chg_dqdv, c='r', label=file.split('.')[0])
                # ax2.plot(df_dischg_u, df_dischg_dqdv, c='c', label=file.split('.')[0])

                ax1.plot(df_chg_u_ewm, df_chg_dqdv_ewm, c=color[i], label=file.split('.')[0], linewidth=line_wdth)
                ax2.plot(df_dischg_u_ewm, df_dischg_dqdv_ewm, c=color[i], label=file.split('.')[0], linewidth=line_wdth)
                fp = findpeaks(method='topology', whitelist=['peak', 'valley'])
                df_res = fp.fit(df_chg_dqdv_ewm)
                # # fp.plot()

                pv_res = df_res['df']
                index_peaks = pv_res.loc[pv_res['peak'], 'x'].values
                index_valleys = pv_res.loc[pv_res['valley'], 'x'].values
                ax1.plot(df_chg_u_ewm.values[index_peaks], df_chg_dqdv_ewm.values[index_peaks],
                         'g*', markersize=line_wdth)
                ax1.plot(df_chg_u_ewm.values[index_valleys], df_chg_dqdv_ewm.values[index_valleys],
                         'co', markersize=line_wdth)
                fp = findpeaks(method='topology', whitelist=['peak', 'valley'])
                df_res = fp.fit(df_dischg_dqdv_ewm)
                # fp.plot()
                index_peaks = pv_res.loc[pv_res['peak'], 'x'].values
                index_valleys = pv_res.loc[pv_res['valley'], 'x'].values
                ax2.plot(df_dischg_u_ewm.values[index_peaks], df_dischg_dqdv_ewm.values[index_peaks],
                         'g*', markersize=line_wdth)
                ax2.plot(df_dischg_u_ewm.values[index_valleys], df_dischg_dqdv_ewm.values[index_valleys],
                         'co', markersize=line_wdth)

            # range_list = list(np.linspace(3.2, 3.6, 25))
            # ax1.set_xticks(range_list)
        ax1.set_title(sys_fold, fontsize=24)
        ax1.legend(fontsize=18, loc='best')
        ax1.grid(linestyle=":", color="r")
        # ax2.legend(fontsize=18, loc='best')
        ax1.set_ylim((0, 7000))
        # plt.show()
        # plt.tight_layout()
        plt.savefig(os.path.join(self.path_png_plt, RUN_MODE + '_' + sys_fold.split('.')[0] + '_peak_valley_demo.png'),
                    dpi=800)
        print()
        pass

    def plt_bysys(self):
        for sys_fold in os.listdir(self.data_join_path):
            sys_path = os.path.join(self.data_join_path, sys_fold)
            plt.figure(figsize=(10, 6), dpi=self.dpi)
            ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
            color = cm.rainbow(np.linspace(0, 1, 256))
            k = 1
            for batch_fold in os.listdir(sys_path):
                batch_path = os.path.join(sys_path, batch_fold)
                for file in os.listdir(batch_path):
                    if not file.endswith('_asm.xlsx'):
                        continue
                    file_name = os.path.join(batch_path, file)
                    df_file = pd.read_excel(file_name, sheet_name='cyl_data') \
                        .sort_values(by='CYCLE_INT')['END_CAPACITY']
                    ax1.plot(df_file.values[:-2], c=color[k], label=batch_fold)
                    k = k + 10
            ax1.set_title('CAPACITY(mAH)', fontsize=16)
            ax1.set_ylabel('CAPACITY(mAH)', fontsize=12)
            ax1.legend(loc='best', fontsize=14)
            ax1.set_ylim((300, 4800))
            ax1.set_xlim((0, 2000))
            ax1.grid()
            # plt.show()
            sys_batch_path_png = os.path.join(self.path_png_plt, 'sys_batchid')
            if not os.path.exists(sys_batch_path_png):
                os.makedirs(sys_batch_path_png)
            plt.savefig(os.path.join(sys_batch_path_png, sys_fold + '_capacity.png'), dpi=self.dpi)
            plt.close()

    def plt_byallsys(self):
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
        color = cm.rainbow(np.linspace(0, 4, 256))
        k = 1
        for sys_fold in os.listdir(self.data_join_path):
            sys_path = os.path.join(self.data_join_path, sys_fold)
            for batch_fold in os.listdir(sys_path):
                batch_path = os.path.join(sys_path, batch_fold)
                for file in os.listdir(batch_path):
                    if not file.endswith('_asm.xlsx'):
                        continue
                    file_name = os.path.join(batch_path, file)
                    df_file = pd.read_excel(file_name, sheet_name='cyl_data') \
                        .sort_values(by='CYCLE_INT')['END_CAPACITY']
                    try:
                        if (sys_fold.startswith('SC14') and sys_fold.endswith('OK')):
                            ax1.plot(df_file.values[:-2], c='r', label=sys_fold)
                        elif (sys_fold.startswith('SC18') and sys_fold.endswith('OK')):
                            ax1.plot(df_file.values[:-2], c='k', label=sys_fold)
                        elif sys_fold.startswith('SC17'):
                            ax1.plot(df_file.values[:-2], c='m', label=sys_fold)
                        else:
                            ax1.plot(df_file.values[:-2], c=color[k], label=sys_fold)
                    except Exception:
                        logger.warning('not enough data')
            k = k + 6
        ax1.set_title('CAPACITY(mAH)', fontsize=16)
        ax1.set_ylabel('CAPACITY(mAH)', fontsize=12)
        ax1.legend(loc='best', fontsize=8)
        ax1.set_ylim((300, 4800))
        ax1.set_xlim((0, 2000))
        ax1.grid()
        # plt.show()
        sys_batch_path_png = os.path.join(self.path_png_plt, 'sys_batchid')
        if not os.path.exists(sys_batch_path_png):
            os.makedirs(sys_batch_path_png)
        plt.savefig(os.path.join(sys_batch_path_png, 'sys_capacity.png'), dpi=self.dpi)
        plt.close()

    def lfp_pv_join(self, RUN_MODE):
        """
        join dqdv/dvdq peak features and count-values
        normal scaled_data range:
        1.CHG_INITVOL 3.0-
        3.CHG_AVGVOL:3.75-

        2.DISCHG_INITVOL:4.1-3.8
        4.DISCHG_AVGVOL:3.4-3.6
        5.CHG_INITVOL_DIFF2 :-0.4-0.4
          self.data_join_path=r'E:\code_git\郑州品质_跳水_数据\0.5C\model_data\data_join_PRD'
        """

        mode_path = os.path.join(self.data_join_path, RUN_MODE)
        for sys_fold in os.listdir(mode_path):
            sys_fold_path = os.path.join(mode_path, sys_fold)
            for file_fold in os.listdir(sys_fold_path):
                fold_path = os.path.join(sys_fold_path, file_fold)
                for file in os.listdir(fold_path):
                    if file.endswith('asm.xlsx') or file.endswith('asm.xls'):
                        df_endv = pd.read_excel(os.path.join(fold_path, file), sheet_name='cyl_data', )
                        df_endv = df_endv.rename(columns={'Unnamed: 0': 'SORT_DATE'}).set_index('CYCLE_NUM')
                        df_endv['delta_RV'] = df_endv['RV_VOLTAGE'].diff()
                        df_endv['delta_SV'] = df_endv['SV_VOLTAGE'].diff()

                    elif file.endswith('_pv.xlsx') or file.endswith('_pv.xls'):
                        df_pv = pd.read_excel(os.path.join(fold_path, file), sheet_name='cyl_data')
                        df_pv_rev = df_pv.rename(columns={'Unnamed: 0': 'CYCLE_NUM'}).set_index('CYCLE_NUM')

                #  for debug only
                # df_cycl = df_endv.sort_values(by='SORT_DATE')
                try:
                    df_cycl = df_endv.merge(df_pv_rev, how='left', left_index=True, right_index=True)
                    df_cycl = df_cycl.sort_values(by=['CYCLE_INT'], ascending=True)
                    sys_file_id = '--'.join([sys_fold, file_fold])
                    df_cycl['SN'] = sys_file_id
                except Exception as err:
                    logger.info('{} errors:{}'.format(sys_file_id, err))
                full_size = df_cycl.shape[0]
                df_cycl_f = self.del_cyl_outlier(df_cycl)
                clean_size = df_cycl_f.shape[0]
                drop_size = full_size - clean_size
                if drop_size > 0:
                    logger.info('{} drop size:{}'.format(sys_file_id, drop_size))
                df_cycl_f.to_excel(os.path.join(fold_path, file_fold.split('.')[0] + '_summary.xlsx'),
                                   sheet_name='cyl_data',
                                   index_label='CYCLE_NUM')
        logger.info('summary data merge done')

    def del_cyl_outlier(self, df_ts_src):
        """
        1. cycle data do capacity calibration during in 100 intervals ,starting from 5-6
        2. outlier cycles that is: 5-6-7-8  105-108  205-208
        3. ...

        4. outlier params:
        df_cycl['DIFF1'] = df_cycl['CHG_INITVOL'].diff()
        df_cycl['DIFF2'] = df_cycl['DIFF1'].diff()
        df_cycl = df_cycl.loc[(df_cycl['CHG_INITVOL'] > 3.00)
                              & (df_cycl['CHG_AVGVOL'] > 3.75)
                              & (df_cycl['DISCHG_INITVOL'] < 4.1) & (df_cycl['DISCHG_INITVOL'] > 3.8)
                              & (df_cycl['DISCHG_AVGVOL'] < 3.6) & (df_cycl['DISCHG_INITVOL'] > 3.4)
                              & df_cycl['DIFF2'].abs() < 0.05
                                  , :]
        """

        from sklearn.ensemble import IsolationForest
        features = ['END_CAPACITY',
                    'CHG_ENDCAPACITY', 'CHG_INITVOL', 'CHG_AVGVOL',
                    'STAT_ENDVOL', 'DELTA_STATVOL', 'MEAN_CHGVOL', 'MEAN_DISCHGVOL']

        df_ts = df_ts_src.set_index('CYCLE_INT').loc[:, features]
        # df_ts['CHG_AVGVOL_DF1'] = df_ts['CHG_AVGVOL'].diff()
        # df_ts['CHG_AVGVOL_DF2'] = df_ts['CHG_AVGVOL_DF1'].diff()
        #
        # df_ts['DELTA_STATVOL_DF1'] = df_ts['DELTA_STATVOL'].diff()
        # df_ts['DELTA_STATVOL_DF2'] = df_ts['DELTA_STATVOL_DF1'].diff()

        df_ts.dropna(how='any', axis=0, inplace=True)
        clf = IsolationForest(n_estimators=20,
                              max_samples='auto',
                              contamination=50 / 2000,
                              max_features=1.0)
        df_ts['label'] = clf.fit_predict(df_ts)
        for out_col in features:
            df_ts.loc[df_ts['label'] == -1, out_col] = np.nan
            # df_ts[out_col] = df_ts[out_col].fillna(df_ts[out_col].rolling(5, min_periods=1).mean())
        df_ts_outlf = df_ts.dropna(how='any', axis=0).drop(columns=['label'], axis=1)
        df_ts_src_cut = df_ts_src.drop(columns=features, axis=1)
        df_ts_src = df_ts_src_cut.merge(df_ts_outlf, how='inner', left_index=True, right_index=True)
        return df_ts_src


def run_peak():
    etf = ExtractFeaturesEC()
    # etf.plt_bysys()
    # etf.plt_byallsys()
    # etf.get_peak_review_asm('ASM_NORM')
    # etf.get_peak_review_asm('ASM_OUTLIER')
    # etf.get_peak_valley_dbg('TRAIN')

    etf.get_peak_valley('TRAIN')
    etf.get_peak_valley('PRED')

    etf.lfp_pv_join('TRAIN')
    etf.lfp_pv_join('PRED')


if __name__ == "__main__":
    run_peak()
