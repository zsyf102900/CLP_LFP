# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     knee_point.py
   Description :
   Author :       ASUS
   date：          2023/4/13
-------------------------------------------------
   Change Activity:
                   2023/4/13:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from log_conf.logger import logger
from utils.params_config import ConfPath, ConfVars, ConstVars

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

kpc_cols = ['fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23',
            'CYCLE_INT',
            'END_CAPACITY']


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)


# ************************@@@@@***************************#
#   KneeePointCalcu
# ************************@@@@@***************************#


class BaseData(ConfParams):
    def __init__(self):
        ConfParams.__init__(self)
        self.MAX_LEN = 1390
        self.y_lim = (3000, 5000)
        self.plt_range = range(100, 1500, 120)
        self.cycle_int = 'CYCLE_INT'
        self.df_norm = self.gen_data_y('NORMAL', self.MAX_LEN)
        self.df_outlier = self.gen_data_y('OUTLIER', self.MAX_LEN)
        print()
        # self.MAX_LEN = 215
        # self.y_lim = (2800, 3250)
        # self.plt_range = range(50, 217, 25)

    def gen_data_y(self, RUN_MODE, MAX_LEN):
        fold_path = os.path.join(self.raw_kpd_data, RUN_MODE)
        data_dict = {}
        for sub_sysbatch in os.listdir(fold_path):
            sub_path = os.path.join(fold_path, sub_sysbatch)
            for fold_ID in os.listdir(sub_path):
                fold_ID_path = os.path.join(sub_path, fold_ID)
                for file in os.listdir(fold_ID_path):
                    if not file.endswith('summary.xlsx'):
                        continue
                    file_path = os.path.join(fold_ID_path, file)
                    df_file = pd.read_excel(file_path,
                                            sheet_name=self.sheet_name,
                                            index_col=self.cycle_num)
                    df_file = df_file.drop(columns=['SN', 'SORT_DATE'], axis=1)
                    df_file_cylint = df_file['CYCLE_INT']
                    df_file = df_file.iloc[5:MAX_LEN, :].rolling(window=10, min_periods=3).mean().loc[:, kpc_cols]
                    df_file['CYCLE_INT'] = df_file_cylint
                    df_file.dropna(how='any', axis=0, inplace=True)
                    # ms = MaxAbsScaler()
                    # scal_ary = ms.fit_transform(df_file)
                    # df_scal = pd.DataFrame(index=df_file.index, columns=kpc_cols)
                    for col in kpc_cols:
                        if col in ('CYCLE_INT'):
                            continue
                        df_cols = df_file[col]
                        df_file[col] = df_cols / df_cols.values[0]

                    # df_file = df_file.iloc[3:, :]
                    # df_file = df_file[df_file[col] > 0]
                    data_dict[file.split(' ')[0]] = df_file
        return data_dict

    def gen_data_y_bak(self, RUN_MODE, MAX_LEN, col):
        fold_path = os.path.join(self.raw_kpd_data, RUN_MODE)
        data_dict = {}
        for file in os.listdir(fold_path):
            file_path = os.path.join(fold_path, file)
            df_file = pd.read_excel(file_path,
                                    sheet_name=self.sheet_name,
                                    index_col=self.cycle_num)
            df_file = df_file.iloc[5:MAX_LEN, :]
            # df_file = df_file.iloc[3:, :]
            df_file = df_file[df_file[col] > 0]
            data_dict[file.split(' ')[0]] = df_file
        return data_dict


class FitKPD(BaseData):

    def __init__(self):
        BaseData.__init__(self)

    def knee_point_fit(self, col):
        """
        1. donnot dropna in gen_data_y ,peak calcu (cycle800-1500) contains too many nan
        2. may cause  capacity calcu not enough data (cycle800-1500)
        """
        df_normal_tmp = self.df_norm
        df_normal = pd.concat(df_normal_tmp, axis=0).dropna(how='any', axis=0)
        regr = LinearRegression()
        regr.fit(df_normal[self.cycle_int].values.reshape(-1, 1), df_normal[col].values.reshape(-1, 1))
        logger.info('residual trend fit done')
        normal_data = df_normal_tmp
        y_pred = []
        y_pred_up = []
        y_pred_low = []
        for _, idf in normal_data.items():
            idf = idf.dropna(how='any', axis=0)
            y_predict = regr.predict(idf[self.cycle_int].values.reshape(-1, 1))
            kpc_err = (y_predict.ravel() - idf[col].values.ravel())
            resid_upper = y_predict + kpc_err.max()
            resid_lower = y_predict - 0.5 * abs(kpc_err.min())
            y_pred.append(y_predict.ravel())
            y_pred_up.append(resid_upper.ravel())
            y_pred_low.append(resid_lower.ravel())
        min_len = min([ary.shape[0] for ary in y_pred])
        y_pred = [ary[:min_len] for ary in y_pred]
        y_pred_up = [ary[:min_len] for ary in y_pred_up]
        y_pred_low = [ary[:min_len] for ary in y_pred_low]
        self.y_pred = np.average(y_pred, axis=0)
        self.y_pred_up = np.average(y_pred_up, axis=0)
        self.y_pred_low = np.average(y_pred_low, axis=0)

        kp_df_bound = pd.DataFrame()
        kp_df_bound['UPPER'] = self.y_pred_up.ravel()
        kp_df_bound['LOWER'] = self.y_pred_low.ravel()
        kp_df_bound['MID'] = self.y_pred.ravel()
        kp_df_bound['index_cylnum'] = np.arange(1, self.y_pred.shape[0] + 1)
        kp_df_bound = kp_df_bound.set_index('index_cylnum')
        kp_df_bound.to_excel(os.path.join(self.pkl_path_kpd, 'kpd_bound.xlsx'), sheet_name='kpd')
        with open(os.path.join(self.pkl_path_kpd, 'kpd_bound.pkl'), 'wb') as bkpd:
            pickle.dump(kp_df_bound, bkpd)
        with open(os.path.join(self.pkl_path_kpd, 'mdl_kpc.pkl'), 'wb') as mkpd:
            pickle.dump(regr, mkpd)
        logger.info('residual bounds done')


class PredKPD(BaseData):

    def __init__(self):
        BaseData.__init__(self)
        with open(os.path.join(self.pkl_path_kpd, 'kpd_bound.pkl'), 'rb') as bkpd:
            self.kp_df_bound = pickle.load(bkpd)
        with open(os.path.join(self.pkl_path_kpd, 'mdl_kpc.pkl'), 'rb') as mkpd:
            self.mdl_kpd = pickle.load(mkpd)

    def knee_crosspoint(self, col):
        cyl_kpc_dict = {}
        df_outlier = self.df_norm
        for filename, idf_outl in df_outlier.items():
            idf_outl = idf_outl.dropna(how='any', axis=0)
            eol_capacity = idf_outl[self.y_label].values[0] * 0.8
            eol_cylnum = idf_outl[idf_outl[self.y_label] <= eol_capacity][self.cycle_int].values[-1]

            df_lower = self.kp_df_bound['LOWER']
            # cross_ary = np.argwhere(np.isclose(df_lower.values, idf_outl[col].values)).reshape(-1)
            cross_ary = np.array([1000, 1001])
            if cross_ary.shape[0] > 0:
                cross_cylnum = cross_ary[0]
            else:
                cross_cylnum = 0
            cyl_kpc_dict[filename] = [cross_cylnum, eol_cylnum]
        df_cyl_kpc = pd.DataFrame.from_dict(data=cyl_kpc_dict, columns=['crosss_num', 'eol_num'], orient='index')
        return df_cyl_kpc.to_json(orient="columns")

    def knee_point_pred(self, col):
        res_pred_dict = {}
        df_outlier = self.df_outlier
        for filename, idf_outl in df_outlier.items():
            idf_outl = idf_outl.reset_index()
            y_predict = self.mdl_kpd.predict(idf_outl[self.cycle_num].values.reshape(-1, 1)).ravel()
            idf_outl['PRED'] = y_predict
            res_pred_dict[filename] = idf_outl
        return res_pred_dict


class PostVisualKp(PredKPD):

    def __init__(self):
        PredKPD.__init__(self)
        with open(os.path.join(self.pkl_path_kpd, 'kpd_bound.pkl'), 'rb') as bkpd:
            self.kp_df_bound = pickle.load(bkpd)
        self.y_pred_up = self.kp_df_bound['UPPER']
        self.y_pred_low = self.kp_df_bound['LOWER']
        self.y_pred = self.kp_df_bound['MID']

    def plt_cmp_run(self, col):
        df_norm = self.df_norm
        df_outlier = self.df_outlier
        fig, (ax) = plt.subplots(1, 1, figsize=(8, 10))
        # fig, (ax, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        # marker = ['*', 'o', 's', 'v', '^', 'p', '^', 'v', 'p', 'd', 'h', '2', '8', '6']
        # color_rbow = plt.cm.rainbow(np.linspace(0, 50, 256))
        for file_name, i_df in df_norm.items():
            file_name = file_name.split('_summary')[0]
            y_cap = i_df.loc[:, [col]]
            # ax.plot(y_cap[y_cap > 500], color='c', label=file_name.split('_summary')[0] + '_ok')
            ax.plot(y_cap, color='c', label=file_name + '_ok')
        for file_name, i_df in df_outlier.items():
            file_name = file_name.split('_summary')[0]
            y_cap = i_df.loc[:, [col]]

            if file_name.startswith('G25'):
                ax.plot(y_cap.values[:-2], c='k', label=file_name + '_kp_outlier', linewidth=4)
            elif file_name.startswith('G24'):
                ax.plot(y_cap.values[:-2], c='dimgrey', label=file_name + '_kp_outlier', linewidth=4)
            elif file_name.startswith('I26'):
                ax.plot(y_cap.values[:-2], c='forestgreen', label=file_name + '_kp_outlier', linewidth=4)
            else:
                ax.plot(y_cap.values[:-2], c='blue', label=file_name + '_kp_outlier', linewidth=4)

        ax.plot(self.y_pred_up, color='sienna', label='upper_bound')
        ax.plot(self.y_pred_low, color='sienna', label='lower_bound')
        ax.plot(self.y_pred, color='k', label='fade_trend', linewidth=4)
        # ax.set_ylim(self.y_lim)
        ax.legend(loc='best', fontsize=8)
        ax.set_title('{} CMP'.format(col))
        ax.set_xlabel('cycle NUM', fontsize=12)
        ax.set_ylabel('{}'.format(col), fontsize=12)
        save_path = os.path.join(self.path_png_plt, col + '_trend.png')
        plt.savefig(save_path, dpi=self.dpi)

        # @@@@@@@@@@@@@@@********* "gif and mp4" **********@@@@@@@@@@@@@@@@@
        # ims = []
        # ax2.set_ylim(self.y_lim)
        # ax2.set_xlim((0, self.MAX_LEN))
        # im_up = ax2.plot(self.y_pred_up, color='sienna', label='upper_bound', linewidth=4)
        # im_low = ax2.plot(self.y_pred_low, color='sienna', label='lower_bound', linewidth=4)
        # im_trend = ax2.plot(self.y_pred, color='k', label='fade_trend', linewidth=4)
        # res_pred_dict = self.knee_point_pred()
        #
        # for cyl_num in self.plt_range:
        #     for filename, idf_all in res_pred_dict.items():
        #         idf = idf_all.iloc[:cyl_num, :]
        #         im_p = ax2.plot(idf['PRED'], color='r', label='RealTime_PRED', linewidth=6)
        #         im_t = ax2.plot(idf[self.y_label], color='g', label='TRUE', linewidth=6)
        #         ims.append(im_p)
        #         ims.append(im_t)
        # leng_label = ['upper_bound', 'lower_bound', 'fade_trend', 'RealTime_PRED', 'TRUE', ]
        # ax2.legend(leng_label, loc='best', fontsize=8)
        # ax2.set_title('{} CMP'.format(col))
        # ax2.set_xlabel('cycle NUM', fontsize=12)
        # ax2.set_ylabel('{}'.format(col), fontsize=12)

        # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=0)
        # wrt_obj = animation.writers['ffmpeg']
        # writer = wrt_obj(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save(os.path.join(self.path_png_plt, "kneepoint_movie.mp4"), writer=writer)
        # ani.save(os.path.join(self.path_png_plt, "kneepoint.gif"))
        # plt.close()
        # logger.info('knee_point plot done')

    def diff_cmp(self, col):
        """
        function also show in postVisual
        COMPARE OK-NG datasets: all features
        """
        df_norm = self.df_norm[col]
        df_outlier = self.df_outlier[col]
        with open(os.path.join(self.pkl_path_mdl, 'lgbm_selected_features.pkl'), 'rb') as sf:
            selected_features = pickle.load(sf)

        for col_f in selected_features + [self.y_label]:
            fig, (ax) = plt.subplots(1, 1, figsize=(12, 8))
            if col_f in ("循环序号"):
                continue
            for key, i_df in df_norm.items():
                i_df = i_df.iloc[:-3, :]
                ax.plot(i_df[col_f], color='c', label=key + '_norm')
            for key, i_df in df_outlier.items():
                i_df = i_df.iloc[:-3, :]
                ax.plot(i_df[col_f], color='m', label=key + '_NG')

            ax.legend(loc='best', fontsize=6)
            ax.set_title(col_f + '_cmp')
            ax.set_xlabel('cycle NUM', fontsize=12)
            if '/' in col_f:
                col_f = col_f.replace('/', '-')
            NG_path = os.path.join(self.path_png_plt, 'kneepoint_norm_NG')
            if not os.path.exists(NG_path):
                os.makedirs(NG_path)
            save_path = os.path.join(NG_path, col_f + '_cmp_f.png')
            plt.savefig(save_path, dpi=self.dpi)


if __name__ == "__main__":
    kpc_cols = ['CYCLE_INT', 'fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23', 'END_CAPACITY']
    for col in kpc_cols:
        kpc = FitKPD()
        kpc.knee_point_fit(col)
        # pkp = PredKPD()
        # pkp.knee_crosspoint()
        pvk = PostVisualKp()
        pvk.plt_cmp_run(col)
