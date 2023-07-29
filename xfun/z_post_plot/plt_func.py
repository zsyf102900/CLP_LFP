# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     plt_func.py
   Description :
   Author :       ASCEND
   date：          2023/1/17
-------------------------------------------------
   Change Activity:
                   2023/1/17:
-------------------------------------------------
"""
__author__ = 'ASCEND'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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
#   PlotVisual
# ************************@@@@@***************************#
class PlotVisual(ConfParams):

    def __init__(self):

        ConfParams.__init__(self)
        self.dqdv_path = self.plt_data_path

    def plt_dqdv_ewm(self, chg_type):
        save_cmp_path = os.path.join(self.path_png_plt, chg_type + '_cmp')
        if not os.path.exists(save_cmp_path):
            os.makedirs(save_cmp_path)

        for fold_status in os.listdir(self.dqdv_path):
            # if not fold_status == 'PRED':
            #     continue
            if not fold_status == 'TRAIN':
                continue
            fold_sts_path = os.path.join(self.dqdv_path, fold_status)
            for cellid in os.listdir(fold_sts_path):
                if not cellid.startswith('E1.xlsx'):
                    continue
                cylcell_path = os.path.join(fold_sts_path, cellid)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(25, 30))
                cycle_names = os.listdir(cylcell_path)
                # cycle_names = [int(id.split('.')[0]) for id in cycle_names ]
                cycle_names.sort(reverse=False)
                cycle_names = [i for i in cycle_names[1:20] if len(i.split('.')[0]) == 4]
                color = cm.rainbow(np.linspace(0, 1, len(cycle_names)))
                for k, file in enumerate(cycle_names):
                    if not file.endswith('.xlsx'):
                        continue

                    cycleid = file.split('.')[0]
                    # if not file.endswith('.csv'):
                    #     continue
                    # if not int(cycleid) in plt_range:
                    #     continue
                    file_path = os.path.join(cylcell_path, file)
                    df_cycl = pd.read_excel(file_path, sheet_name=self.sheet_name)
                    if chg_type in ('charge'):
                        df_cycl = df_cycl[df_cycl[self.vars['STATUS']] == '充电 CC']
                    # df_cycl_p = df_cycl.loc[:, ['dv', 'dq', 'dqdv']].rolling(window=20, min_periods=3).mean()
                    # df_cycl_p['dqdv_roll'] = df_cycl_p['dq'] / df_cycl_p['dv']
                    # df_cycl_p['dvdq_roll'] = df_cycl_p['dv'] / df_cycl_p['dq']
                    # df_cycl_p.dropna(how='any', inplace=True)
                    try:
                        ax1.plot(df_cycl[self.vars['VOLTAGE']], df_cycl['dqdv_roll'], c=color[k],
                                 label=cycleid + ':dqdv_roll')
                        # ax1.plot(df_cycl[self.vars['VOLTAGE']][2:], df_cycl_p['dqdv'], label=cycleid + ':src')
                        ax2.plot(df_cycl[self.vars['VOLTAGE']], df_cycl[self.vars['CAPACITY']], c=color[k],
                                 label=cycleid + ':Q-U')
                        ax3.plot(df_cycl[self.vars['VOLTAGE']], df_cycl['dvdq_roll'], c=color[k],
                                 label=cycleid + ':dvdq_roll')
                        # save_file = os.path.join(save_cmp_path, file[:-4] + '.png')
                        # plt.savefig(save_file, dpi=self.dpi)
                        # plt.show()
                        # print()
                    except Exception as err:
                        logger.info('cycleID:{} data exception:{}'.format(cycleid, err))
                ax1.legend(fontsize=18)
                ax1.set_xlabel('voltage U:(V)', fontsize=16)
                ax1.set_ylabel('dqdv', fontsize=16)
                ax1.set_title(chg_type + '_U-DQDV curves', fontsize=16)

                ax2.legend(fontsize=18)
                ax2.set_xlabel('voltage U:(V)', fontsize=16)
                ax2.set_ylabel('capacity Q:(AH)', fontsize=16)
                ax2.set_title(chg_type + '_U-Q curves', fontsize=16)

                ax3.legend(fontsize=18)
                ax3.set_xlabel('voltage U:(V)', fontsize=16)
                ax3.set_ylabel('dvdq', fontsize=16)
                ax3.set_title(chg_type + '_U-DVDQ curves', fontsize=16)

                save_file = os.path.join(save_cmp_path, cellid + '.png')
                plt.savefig(save_file, dpi=self.dpi)
                plt.close()

    def plt_dqdv_bycellid(self, chg_type):
        for fold_chg_status in os.listdir(self.dqdv_path):
            fold_chgsts_path = os.path.join(self.dqdv_path, fold_chg_status)
            for cellid in os.listdir(fold_chgsts_path):
                cylcell_path = os.path.join(fold_chgsts_path, cellid)
                save_cyl_path = os.path.join(self.path_png_plt, chg_type + '_bycycle', cellid)
                if not os.path.exists(save_cyl_path):
                    os.makedirs(save_cyl_path)

                cycle_names = os.listdir(cylcell_path)
                for k, cycl_file in enumerate(cycle_names):
                    cycleid = cycl_file.split('.')[0]
                    if not cycl_file.endswith('.csv'):
                        continue
                    file_path = os.path.join(cylcell_path, cycl_file)
                    df_cycl = pd.read_csv(file_path, sep=',')
                    df_cycl = df_cycl.iloc[:-5, :]
                    if chg_type in ('charge'):
                        df_cycl = df_cycl[df_cycl[self.vars['STEP']] == 'CC']
                    #
                    # df_cycl_p = df_cycl.loc[:, ['dv', 'dq', 'dqdv']].rolling(window=20, min_periods=3).mean()
                    # df_cycl_p['dqdv_roll'] = df_cycl_p['dq'] / df_cycl_p['dv']
                    # df_cycl_p['dvdq_roll'] = df_cycl_p['dv'] / df_cycl_p['dq']
                    # df_cycl_p.dropna(how='any', inplace=True)

                    if cycl_file.startswith('1058.0.csv'):
                        continue
                    else:
                        pass

                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 30))
                    try:
                        ax1.plot(df_cycl[self.vars['VOLTAGE']], df_cycl['dqdv_roll'], 'r', label=cycleid + ':dqdv_roll')

                        ax2.plot(df_cycl[self.vars['VOLTAGE']], df_cycl[self.vars['CAPACITY']], 'c',
                                 label=cycleid + ':Q-U')

                        ax3.plot(df_cycl[self.vars['VOLTAGE']], df_cycl['dvdq_roll'], 'm', label=cycleid + ':dvdq_roll')


                    except Exception as err:
                        logger.info('cycleID:{} data exception:{}'.format(cycleid, err))
                    ax1.legend(fontsize=18)
                    ax1.set_xlabel('voltage U:(V)', fontsize=16)
                    ax1.set_ylabel('dqdv', fontsize=16)
                    ax1.set_title(chg_type + '_U-DQDV curves', fontsize=16)

                    ax2.legend(fontsize=18)
                    ax2.set_xlabel('voltage U:(V)', fontsize=16)
                    ax2.set_ylabel('capacity Q:(AH)', fontsize=16)
                    ax2.set_title(chg_type + '_U-Q curves', fontsize=16)

                    ax3.legend(fontsize=18)
                    ax3.set_xlabel('voltage U:(V)', fontsize=16)
                    ax3.set_ylabel('dvdq', fontsize=16)
                    ax3.set_title(chg_type + '_U-DVDQ curves', fontsize=16)

                    save_file = os.path.join(save_cyl_path, cycleid + '.png')
                    plt.savefig(save_file, dpi=300)
                    plt.close()

    def lfp_plt(self):
        """
        1.cmap  'viridis'  rainbow
        """

        for fold in os.listdir(self.dqdv_path):
            fold_path = os.path.join(self.dqdv_path, fold)
            file_list = os.listdir(fold_path)
            file_list.sort()
            file_list = [file for i, file in enumerate(file_list) if (i % self.CYLPLT_STEP) == 0]
            color = cm.rainbow(np.linspace(0, 1, len(file_list)))
            plt.figure(figsize=(20, 45), dpi=self.dpi)
            ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=1, rowspan=1)
            ax2 = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=1)
            ax3 = plt.subplot2grid((4, 2), (1, 0), colspan=1, rowspan=1)
            ax4 = plt.subplot2grid((4, 2), (1, 1), colspan=1, rowspan=1)
            ax5 = plt.subplot2grid((4, 2), (2, 0), colspan=1, rowspan=1)
            ax6 = plt.subplot2grid((4, 2), (2, 1), colspan=1, rowspan=1)
            axins5 = ax5.inset_axes((0.20, 0.15, 0.45, 0.40))
            axins6 = ax6.inset_axes((0.15, 0.15, 0.45, 0.40))
            ax78 = plt.subplot2grid((4, 2), (3, 0), colspan=2, rowspan=1)

            for k, file in enumerate(file_list):
                file_name = os.path.join(fold_path, file)
                df = pd.read_excel(file_name, sheet_name='cyl_data')
                if file.startswith('endq'):
                    ax78.plot(df['EndCapacity'].abs())
                else:
                    df_chg = df[df[self.vars['STATUS']] == '充电 CC']
                    df_dischg = df[df[self.vars['STATUS']] == '放电 DC']
                    # df_dischg.loc[:, self.vars['CAPACITY']] = df_dischg[self.vars['CAPACITY']].apply(
                    #     lambda x: np.abs(x))

                    df_chg = df_chg[(df_chg[self.vars['VOLTAGE']] >= 3.2) &
                                    (df_chg[self.vars['VOLTAGE']] <= 3.6)
                                    ]
                    df_dischg = df_dischg[(df_dischg[self.vars['VOLTAGE']] >= 2.95) &
                                          (df_dischg[self.vars['VOLTAGE']] <= 3.28)]
                    df_chg_u, df_dischg_u = df_chg[self.vars['VOLTAGE']], df_dischg[self.vars['VOLTAGE']]
                    df_chg_dqdv, df_dischg_dqdv = df_chg['dqdv_roll'], df_dischg['dqdv_roll']

                    df_chg_u_ewm = df_chg_u.rolling(window=20,
                                                    min_periods=3).mean().rolling(window=5, min_periods=3).mean()
                    df_dischg_u_ewm = df_dischg_u.rolling(window=20,
                                                          min_periods=3).mean().rolling(window=5, min_periods=3).mean()
                    df_chg_dqdv_ewm = df_chg_dqdv. \
                        rolling(window=20, min_periods=3).mean().rolling(window=5, min_periods=3).mean()
                    df_dischg_dqdv_ewm = df_dischg_dqdv. \
                        rolling(window=20, min_periods=3).mean().rolling(window=5, min_periods=3).mean()

                    # ax1.plot(df_chg[self.vars['VOLTAGE']], df_chg['dqdv_roll'],
                    #          c=color[k], label=file.split('.')[0])
                    # ax2.plot(df_dischg[self.vars['VOLTAGE']], df_dischg['dqdv_roll'],
                    #          c=color[k], label=file.split('.')[0])

                    ax1.plot(df_chg_u_ewm, df_chg_dqdv_ewm, c=color[k], label=file.split('.')[0] + '_ewm')
                    ax2.plot(df_dischg_u_ewm, df_dischg_dqdv_ewm, c=color[k], label=file.split('.')[0] + '_ewm')

                    # ax3.plot(df_chg[self.vars['CAPACITY']],
                    #          df_chg['dvdq_roll'],
                    #          c=color[k], label=file.split('.')[0])
                    # ax4.plot(df_dischg[self.vars['CAPACITY']],
                    #          df_dischg['dvdq_roll'],
                    #          c=color[k], label=file.split('.')[0])

                    ax3.plot(df_chg[self.vars['CAPACITY']].rolling(window=5, min_periods=3).mean(),
                             df_chg['dvdq_roll'].rolling(window=5, min_periods=3).mean(),
                             c=color[k], label=file.split('.')[0] + '_ewm')
                    ax4.plot(df_dischg[self.vars['CAPACITY']].rolling(window=5, min_periods=3).mean(),
                             df_dischg['dvdq_roll'].rolling(window=5, min_periods=3).mean(),
                             c=color[k], label=file.split('.')[0] + '_ewm')

                    ax5.plot(df_chg[self.vars['CAPACITY']], df_chg[self.vars['VOLTAGE']], c=color[k],
                             label=file.split('.')[0])

                    ax6.plot(df_dischg[self.vars['CAPACITY']], df_dischg[self.vars['VOLTAGE']], c=color[k],
                             label=file.split('.')[0])

                    # insert subplot
                    axins5.plot(df_chg[self.vars['CAPACITY']], df_chg[self.vars['VOLTAGE']], c=color[k],
                                label=file.split('.')[0])
                    df_dischg_cut = df_dischg[df_dischg[self.vars['CAPACITY']] > -10]
                    axins6.plot(df_dischg_cut[self.vars['CAPACITY']], df_dischg_cut[self.vars['VOLTAGE']], c=color[k],
                                label=file.split('.')[0])

            ax1.set_xlabel('voltage U:(V)', fontsize=16)
            ax1.set_ylabel('dqdv', fontsize=16)
            ax1.set_title('chg_voltage-DQDV', fontsize=16)
            ax1.set_xlim((3.2, 3.6))
            ax1.legend(fontsize=18)

            ax2.set_xlabel('voltage U:(V)', fontsize=16)
            ax2.set_ylabel('dqdv', fontsize=16)
            ax2.set_title('dischg_voltage-DQDV', fontsize=16)
            ax2.set_xlim((3.0, 3.3))
            ax2.legend(fontsize=18)

            ax3.set_xlabel('capacity Q:(AH)', fontsize=16)
            ax3.set_ylabel('dvdq', fontsize=16)
            ax3.set_title('chg dvdq', fontsize=16)
            ax3.set_xlim((5, 110))
            ax3.set_ylim((0, 0.008))
            ax3.legend(fontsize=18)

            ax4.set_xlabel('capacity Q:(AH)', fontsize=16)
            ax4.set_ylabel('dvdq', fontsize=16)
            ax4.set_title('dischg dvdq', fontsize=16)
            ax4.set_xlim((-110, -5))
            ax4.set_ylim((0, 0.008))
            ax4.legend(fontsize=18)

            ax5.set_xlabel('capacity Q:(AH)', fontsize=16)
            ax5.set_ylabel('voltage U:(V)', fontsize=16)
            ax5.set_title('chg_Q-U', fontsize=16)
            axins5.set_xlim((0, 10))
            axins5.set_ylim((2.8, 3.3))
            ax5.legend(fontsize=18)

            ax6.set_xlabel('capacity Q:(AH)', fontsize=16)
            ax6.set_ylabel('voltage U:(V)', fontsize=16)
            ax6.set_title('dischg_Q-U', fontsize=16)
            # axins6.set_xlim((-20, 0))
            axins6.set_ylim((3.25, 3.335))
            ax6.legend(fontsize=18, loc='right')

            save_cyl = os.path.join(self.path_png_plt, fold[:-5] + '_cyl.png')
            plt.tight_layout()
            plt.savefig(save_cyl)

    def lfp_endv_plt(self):
        for fold in os.listdir(self.dqdv_path):
            path_fold = os.path.join(self.dqdv_path, fold)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 30))
            # ax1_twin = ax1.twins()
            # ax2_twin = ax2.twins()
            for file in os.listdir(path_fold):
                if not file.startswith('endq'):
                    continue
                file_end = os.path.join(path_fold, file)
                df = pd.read_excel(file_end, sheet_name='cyl_data')
                df['EndCapacity'] = df['EndCapacity'].apply(lambda x: np.abs(x))
                df = df[df['StaticEndVol'] >= 3.48]
                df.sort_index(ascending=False, inplace=True)
                # EndCapacity	ChgInitVol	DischgInitVol	StaticEndVol	DeltaStatU	CycleNum
                ax1.plot(df['ChgInitVol'], c='r', label='ChgInitVol:V')
                ax2.plot(df['DischgInitVol'], c='m', label='DischgInitVol:V')
                ax2.plot(df['StaticEndVol'], c='c', label='StaticEndVol:V')

                ax3.plot(df['DeltaStatU'] * 1000, c='c', label='DeltaStatU:mv')
                ax1.set_xlabel('cyl_NUM', fontsize=18)
                ax1.set_ylabel('chg Voltage U:(V)', fontsize=18)
                ax1.set_title('chg Voltage ', fontsize=24)
                ax1.legend(fontsize=22, loc='best')

                ax2.set_xlabel('cyl_NUM', fontsize=18)
                ax2.set_ylabel('Dischg-StaticEndVol U:(V)', fontsize=18)
                ax2.set_title('Dischg-StaticEndVol', fontsize=24)
                ax2.legend(fontsize=22, loc='best')

                ax3.set_xlabel('cyl_NUM', fontsize=18)
                ax3.set_ylabel('DeltaStatU:mv', fontsize=24)
                ax3.set_title('DeltaStatU', fontsize=24)
                ax3.legend(fontsize=22, loc='best')

                xtick = list(range(0, df.shape[0], 10))
                xtick_label = df['CycleNum'].values[xtick]
                ax1.set_xticks([])
                ax2.set_xticks(xtick)
                ax2.set_xticklabels(xtick_label, fontsize=22, rotation=45)
                ax3.set_xticks(xtick)
                ax3.set_xticklabels(xtick_label, fontsize=22, rotation=45)
            plt.tight_layout()
            save_cyl = os.path.join(self.path_png_plt, fold[:-5] + '_cyl_endval.png')
            plt.savefig(save_cyl, dpi=self.dpi)


if __name__ == "__main__":
    # 5.plot postvisual
    pv = PlotVisual()
    # pv.plt_dqdv_bycellid('discharge')
    # pv.plt_dqdv_bycellid('charge')
    # pv.plt_dqdv_ewm('discharge')
    pv.plt_dqdv_ewm('charge')
    # pv.lfp_plt()
    # pv.lfp_pv_plt()
    # pv.lfp_endv_plt()
