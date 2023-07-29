#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     params_config.py
   Description :
   Author :       SV00240663
   date：          2021/10/21
-------------------------------------------------
   Change Activity:
                   2021/10/21:
-------------------------------------------------
"""
__author__ = 'SV00240663'

import os
import platform
from utils.model_conf import MODEL_ID, CC_RAT


# ************************@@@@@***************************#
#   path conf
# ************************@@@@@***************************#
class ConfPath:
    """
    for windows path: path contains ':' probibited

    """

    def __init__(self):
        current_path = os.path.dirname(__file__)
        if platform.system().lower() == 'windows':
            # self.abs_path = r'E:\王先生\bak-note\code_git\LFP_DATA\MODEL_DATA'
            # self.predld_path = r'E:\王先生\bak-note\code_git\LFP_DATA\predld_data'
            # raw_par_path = r'E:\code_git\抚州LFP循环'
            # raw_par_path = r'E:\code_git\抚州LFP循环2'
            # raw_par_path = r'E:\code_git\LFP_CYCLE3000'
            # raw_par_path = r'E:\code_git\抚州_LFP-3-31'
            # raw_par_path = r'E:\code_git\抚州_tmp'
            # raw_par_path = r'E:\code_git\郑州张华-下柜前数据_4颗电芯\3.25AH'
            # raw_par_path = r'E:\code_git\NCM_5.0AH-17颗\count_values'
            # raw_par_path = r'E:\code_git\NCM_5.0AH-17颗\CURVES\4.2v-normal-充放电过程数据'
            raw_par_path = r'E:\code_git\郑州品质_跳水_数据\0.5C'
            # raw_par_path = r'E:\code_git\抚州6月-长循环\LFP产品组_长周期'
            # raw_par_path = r'E:\code_git\抚州6月-长循环\LFP_TEST'
            # raw_par_path = r'E:\code_git\郑州品质_跳水_数据\0.5C_batch'
            # raw_par_path = r'E:\code_git\郑州品质_跳水_数据\NCA_5.0TEST'
            # raw_par_path = r'E:\code_data'


            # self.raw_srcdata_path = os.path.join(raw_par_path, 'data_small_src', '4.2V-25度-0.5C')
            self.raw_srcdata_path = os.path.join(raw_par_path, 'data_small_src')

            self.raw_data_path = os.path.join(raw_par_path, 'data_small')
            self.raw_kpd_data = os.path.join(raw_par_path, 'kpd_data')
            self.abs_path = os.path.join(raw_par_path, 'model_data')
            self.log_path = os.path.join(current_path, 'log')
            self.src_path = os.path.join(self.abs_path, 'src_data')
            self.cycle_data_path = os.path.join(self.abs_path, 'cycle_data')
            self.plt_data_path = os.path.join(self.abs_path, 'plt_data')
            self.data_join_path = os.path.join(self.abs_path, 'data_join')
            self.ts_avg_fatures_path = os.path.join(self.abs_path, 'ts_avg_features')

            self.pkl_path = os.path.join(self.abs_path, 'pkl_fold')
            self.pkl_path_scale = os.path.join(self.pkl_path, 'scale')
            self.pkl_path_mdl = os.path.join(self.pkl_path, 'mdl')
            self.pkl_path_kpd = os.path.join(self.pkl_path, 'kpd')

            self.pred_path = os.path.join(self.abs_path, 'pred_allmdl')
            self.path_png_plt = os.path.join(self.abs_path, 'png_plt')

            self.diff_dqdv_path = os.path.join(self.abs_path, 'diff_dqdv')
            self.diff_dvdq_path = os.path.join(self.abs_path, 'diff_dvdq')

            if not os.path.exists(self.cycle_data_path):
                os.makedirs(self.cycle_data_path)
            if not os.path.exists(self.plt_data_path):
                os.makedirs(self.plt_data_path)
            if not os.path.exists(self.data_join_path):
                os.makedirs(self.data_join_path)
            if not os.path.exists(self.ts_avg_fatures_path):
                os.makedirs(self.ts_avg_fatures_path)

            if not os.path.exists(self.pkl_path_scale):
                os.makedirs(self.pkl_path_scale)
            if not os.path.exists(self.pkl_path_mdl):
                os.makedirs(self.pkl_path_mdl)
            if not os.path.exists(self.pkl_path_kpd):
                os.makedirs(self.pkl_path_kpd)

            if not os.path.exists(self.pred_path):
                os.makedirs(self.pred_path)
            if not os.path.exists(self.path_png_plt):
                os.makedirs(self.path_png_plt)

            if not os.path.exists(self.diff_dqdv_path):
                os.makedirs(self.diff_dqdv_path)

            if not os.path.exists(self.diff_dvdq_path):
                os.makedirs(self.diff_dvdq_path)

        elif platform.system().lower() == 'linux':
            pass


# ************************@@@@@***************************#
#   vars conf
# ************************@@@@@***************************#

class ConfVars:

    def __init__(self):
        """
         columns vars：data columns
         const vars：
        :param run_mode:
        """
        self.MODEL_ID = MODEL_ID
        self.CC_RAT = CC_RAT
        if MODEL_ID == 'LFPXY_ONGOING':
            # self.sheet_name = 'cycles_info'
            self.sheet_name = 'cyl_data'
            self.y_label = 'END_CAPACITY'
            self.date_col = 'SORT_DATE'
            self.drop_cols = ['SN', 'SORT_DATE', 'CYCLE_NUM']
            self.timevar_features = ['DISCHG_ENDENERGY', 'CHG_ENDCAPACITY', 'CHG_ENDENERGY']
            self.x_columns = ['循环', '测试时间', '步骤时间', '电压/V', '电流/mA', '容量/mAh', '能量/mWh']

            # self.vars = {
            #     'CYCLESID': '循环号',
            #     'STEPTIME': '工步时间',
            #     'STEP': '工步号',
            #     'VOLTAGE': '电压(V)',
            #     'CURRENCY': '电流(A)',
            #     'INCAPACITY': '充电容量(Ah)',
            #     'OUTCAPACITY': '放电容量(Ah)',
            # }
            self.vars = {
                'CYCLESID': '循环',
                'STEPTIME': '绝对时间',
                'STEP': '步次',
                'STATUS': '状态',
                'VOLTAGE': '实际电压(V)',
                'CURRENCY': '实际电流(A)',
                'CAPACITY': '容量(Ah)',
                'ENERGY': '能量(Wh)',
                'INCAPACITY': '充电容量(Ah)',
                'OUTCAPACITY': '放电容量(Ah)'
            }

            self.vars_CD = {'CC': '充电 CC',
                            'CV': '充电 CV',
                            'DC': '放电 DC',
                            'REST': '静置'
                            }

            self.proc_parms = {'CC_CURRENCY': 2500,
                               'CV_VOLTAGE': 4.2 - 0.01
                               }

        elif MODEL_ID == 'LFPHYN_CYCLE3000':
            self.sheet_name = 'cyl_data'
            self.sheet_name = 'cyl_data'
            self.y_label = 'END_CAPACITY'
            self.date_col = 'SORT_DATE'
            self.drop_cols = ['SN', 'SORT_DATE', 'CYCLE_NUM']
            self.vars = {
                # 'CYCLESID': '循环号',
                'CYCLESID': '循环次数',
                'STEPTIME': '真实时间',
                'STEP': '工步号',
                'STATUS': '工步类型',
                'VOLTAGE': '采样电压(V)',
                'CURRENCY': '采样电流(A)',
                'CAPACITY': '容量(Ah)',
                'ENERGY': '能量(Wh)',
                # 'CAPACITY': '容量(mAh)',
                # 'ENERGY': '能量(mWh)',
            }

            self.vars_CD = {'CC': '恒流充电',
                            'CV': '恒压充电',
                            'DC': '恒流放电',
                            'REST': '静置'
                            }

            self.x_columns = [
                '恒流充电容量(mAh)',
                '恒压充电容量(mAh)',
                '总充电容量(mAh)',
                '恒流放电容量(mAh)',
                '总放电容量(mAh)',
                '恒流充电时间(Sec)',
                '恒压充电时间(Sec)',
                '恒流放电时间(Sec)',
                '总放电时间(Sec)',
                '放电均压(V)',
                '中值电压(V)',
                '循环号']
            self.cycle_num = '循环号'
            self.date_columns = ['恒流充电时间(Sec)', '恒压充电时间(Sec)', '恒流放电时间(Sec)', '总放电时间(Sec)']
            # self.y_label = '容量(Ah)'
            self.date_col = 'SORT_DATE'
            self.sheet_src = '测试数据'

        elif MODEL_ID == 'NCM_3.25':
            self.sheet_name = 'cyl_data'
            self.vars = {
                'CYCLESID': '循环号',
                'STEPTIME': '真实时间',
                'STEP': '工步号',
                'STATUS': '工步类型',
                'VOLTAGE': '采样电压(V)',
                'CURRENCY': '采样电流(A)',
                'CAPACITY': '容量(mAh)',
                'ENERGY': '能量(mWh)',
                'INCAPACITY': '充电容量(Ah)',
                'OUTCAPACITY': '放电容量(Ah)',

            }
            self.y_label = 'END_CAPACITY'
            self.vars_CD = {'CC': 'C_CC',
                            'CV': 'C_CV',
                            'DC': 'D_DC',
                            'REST': 'R'
                            }
            self.x_columns = ['循环序号', '充电容量/mAh', '充电能量/mWh', '放电能量/mWh', '放电中压/V',
                              '恒流充入容量/mAh', '放电终压/V']
            self.cycle_num = '循环序号'
            self.y_label_simple = '放电容量/mAh'

        elif MODEL_ID == 'NCM_5.0':
            self.sheet_name = 'cyl_data'
            # self.vars_SRC = {
            #     'CYCLESID': '循环号',
            #     'TESTTIME': '测试时间',
            #     'STEPTIME': '步骤时间',
            #     'STATUS': '状态',
            #     'VOLTAGE': '电压/V',
            #     'CURRENCY': '电流/mA',
            #     'CAPACITY': '容量/mAh',
            #     'ENERGY': '能量/mWh',
            # }
            self.vars = {
                'CYCLESID': '循环序号',
                'STEPTIME': '测试时间',
                'STATUS': '状态',
                'VOLTAGE': '电压/V',
                'CURRENCY': '电流/mA',
                'CAPACITY': '容量/mAh',
                'ENERGY': '能量/mWh',
            }
            self.y_label = 'END_CAPACITY'
            self.vars_CD = {'CC': 'C_CC',
                            'CV': 'C_CV',
                            'DC': 'D_CC',
                            'REST': 'R'
                            }
            self.x_columns = ['循环序号', '充电容量/mAh', '充电能量/mWh', '放电能量/mWh', '放电中压/V',
                              '恒流充入容量/mAh', '放电终压/V']
            self.cycle_num = '循环序号'
            self.cycle_num = 'CYCLE_NUM'
            self.y_label_simple = '放电容量/mAh'
            self.date_col = 'SORT_DATE'
            self.drop_cols = ['SN', 'SORT_DATE', 'CYCLE_NUM']
            self.sheet_src = '记录'


# ************************@@@@@***************************#
#   vars conf
# ************************@@@@@***************************#


class ConstVars:
    def __init__(self):
        self.POOL_NUM = 36
        self.MAX_SLICE_PAREL = 50
        self.CYLPLT_STEP = 10
        self.MAX_PRED_SLICE = 10
        self.DQDV_EWM_WINDOWS = 10
        self.DQDV_EWM_STEP = 5
        self.dpi = 300
        self.DISCHGCURRENT = 80
        self.dqdv_initcycle = 5
        self.CC_RAT = CC_RAT
        """
        V1:max valley
        V2:sec valley
        V3: min valley
        
        P1:max peak
        P2:sec peak
        P3: min peak
        
        """
        self.dqdv_window = 10
        self.dqdv_maxcycle = 520
        if MODEL_ID.startswith('LFP'):
            # self.DQDV_LMT = {'P3': (3.31, 3.35),
            #                  'P2': (3.367, 3.4),
            #                  'P1': (3.4, 3.43),
            #                  'V3': (3.0, 3.2),
            #                  'V2': (3.33, 3.37),
            #                  'V2P': (3.37, 3.42),
            #                  'V1': (3.45, 3.6)
            #                  }
            self.DQDV_LMT = {'P3': (3.31, 3.35),
                             'P2': (3.367, 3.42),
                             'P1': (3.42, 3.45),
                             'V3': (3.0, 3.2),
                             'V2': (3.33, 3.37),
                             'V2P': (3.37, 3.43),
                             'V1': (3.43, 3.6)
                             }
            self.DVDQ_LMT = {
                            'QLIM': (40, 90),
                            'V2P': (2500, 3800),
                            'P2': (2000, 3500),
                            'V2': (1500, 2500)
                             }
            # *****************
            self.DQDV_LMT_dischg = {'P3': (3.31, 3.35),
                                    'P2': (3.367, 3.4),
                                    'P1': (3.4, 3.43),
                                    'V3': (3.0, 3.2),
                                    'V2': (3.33, 3.37),
                                    'V2P': (3.37, 3.42),
                                    'V1': (3.45, 3.6)
                                    }
        elif MODEL_ID == 'NCM_3.25':
            self.DQDV_LMT = {'P3': (3.4, 3.7),
                             'P2': (3.7, 3.9),
                             'P1': (3.9, 4.1),
                             'V3': (3.0, 3.4),
                             'V2': (3.6, 3.7),
                             'V2P': (3.8, 4.0),
                             'V1': (4.1, 4.2)
                             }
            # *****************
            self.DQDV_LMT_dischg = {}


        elif MODEL_ID == 'NCM_5.0' and self.CC_RAT == '0.5C':
            self.DQDV_LMT = {'P3': (3.4, 3.7),
                             'P2': (3.7, 3.9),
                             'P1': (3.9, 4.25),
                             'V3': (3.0, 3.45),
                             'V2': (3.6, 3.7),
                             'V2P': (3.8, 4.0),
                             'V1': (4.1, 4.2)
                             }
            # 7.14
            # V1: (4.1, 4.2)
            # 'P1': (3.9, 4.25),

            self.DVDQ_LMT = {
                            'QLIM': (1200, 4500),
                            'P1': (2900, 4250),
                            'V2P': (2500, 3800),
                            'P2': (2000, 3500),
                            'V2': (1500, 2500)
                             }

            # *****************
            self.DQDV_LMT_dischg = {}

            pass

        elif MODEL_ID == 'NCM_5.0' and self.CC_RAT == '1C':
            self.DQDV_LMT = {
                'P2': (3.7, 4.05),
                'P1': (4.0, 4.25),
                'V3': (3.0, 3.6),
                'V2P': (3.8, 4.1),
                'V1': (4.1, 4.2)
            }


            # *****************
            self.DQDV_LMT_dischg = {}

            pass
