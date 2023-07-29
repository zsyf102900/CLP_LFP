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
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
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
#   KneeePointCalcu
# ************************@@@@@***************************#

class KneeePointCalcu(ConfParams):

    def __init__(self):
        ConfParams.__init__(self)
        self.sheet_name = '循环'
        # self.MAX_LEN = 1470
        # self.y_lim=(3000,5000)
        # self.plt_range = range(100, 1500, 120)
        self.MAX_LEN = 215
        self.y_lim = (2800, 3250)
        self.plt_range = range(50, 217, 25)

    def plot_cmp_y(self, RUN_MODE, MAX_LEN):
        fold_path = os.path.join(self.raw_data_path, RUN_MODE)
        data_dict = {}
        for file in os.listdir(fold_path):
            file_path = os.path.join(fold_path, file)
            df_file = pd.read_excel(file_path, sheet_name=self.sheet_name,
                                    index_col=self.cycle_num)
            # df_file = df_file.iloc[5:MAX_LEN, :][[self.y_label]]
            df_file = df_file.iloc[3:MAX_LEN, :]
            df_file = df_file[df_file[self.y_label] > 0]
            data_dict[file.split(' ')[0]] = df_file
        return data_dict

    def plt_cmp_run(self):
        df_tr = self.plot_cmp_y('TRAIN', self.MAX_LEN)
        df_pred = self.plot_cmp_y('PRED', self.MAX_LEN)
        df_outlier = self.plot_cmp_y('OUTLIER', self.MAX_LEN)
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        marker = ['*', 'o', 's', 'v', '^', 'p', '^', 'v', 'p', 'd', 'h', '2', '8', '6']
        color_rbow = plt.cm.rainbow(np.linspace(0, 50, 256))
        for file_name, i_df in df_tr.items():
            y_cap = i_df.loc[:, [self.y_label]]
            ax.plot(y_cap[y_cap > 2000], color='c', label=file_name)
        for file_name, i_df in df_pred.items():
            y_cap = i_df.loc[:, [self.y_label]]
            ax.plot(y_cap[y_cap > 2000], color='m', label=file_name)
        for file_name, i_df in df_outlier.items():
            y_cap = i_df.loc[:, [self.y_label]]
            ax.plot(y_cap[y_cap > 2000],
                    color='g',
                    label=file_name + '_knee_point_outlier')
        ax.plot(self.y_pred_up, color='sienna', label='upper_bound')
        ax.plot(self.y_pred_low, color='sienna', label='lower_bound')
        ax.plot(self.y_pred, color='k', label='fade_trend', linewidth=4)

        ax.set_ylim(self.y_lim)
        ax.legend(loc='best', fontsize=8)
        ax.set_title('Capacity CMP')
        ax.set_xlabel('cycle NUM', fontsize=12)
        ax.set_ylabel('CAPACITY Reten mAH', fontsize=12)

        ims = []
        ax2.set_ylim(self.y_lim)
        ax2.set_xlim((0, self.MAX_LEN))
        im_up = ax2.plot(self.y_pred_up, color='sienna', label='upper_bound', linewidth=4)
        im_low = ax2.plot(self.y_pred_low, color='sienna', label='lower_bound', linewidth=4)
        im_trend = ax2.plot(self.y_pred, color='k', label='fade_trend', linewidth=4)

        for cyl_num in self.plt_range:
            res_pred_dict = self.knee_point_pred(cyl_num)
            for filename, idf in res_pred_dict.items():
                im_p = ax2.plot(idf['PRED'], color='r', label='RealTime_PRED', linewidth=6)
                im_t = ax2.plot(idf[self.y_label], color='g', label='TRUE', linewidth=6)
                ims.append(im_p)
                ims.append(im_t)
        leng_label = ['upper_bound', 'lower_bound', 'fade_trend', 'RealTime_PRED', 'TRUE', ]
        ax2.legend(leng_label, loc='best', fontsize=8)
        ax2.set_title('Capacity CMP')
        ax2.set_xlabel('cycle NUM', fontsize=12)
        ax2.set_ylabel('CAPACITY Reten mAH', fontsize=12)
        save_path = os.path.join(self.path_png_plt, 'capacity_train_pred.png')
        plt.savefig(save_path, dpi=self.dpi)
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=0)
        wrt_obj = animation.writers['ffmpeg']
        writer = wrt_obj(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(os.path.join(self.path_png_plt, "kneepoint_movie.mp4"), writer=writer)
        ani.save(os.path.join(self.path_png_plt, "kneepoint.gif"))
        plt.close()
        logger.info('knee_point plot done')

    def diff_cmp(self):
        df_tr = self.plot_cmp_y('TRAIN', self.MAX_LEN)
        df_pred = self.plot_cmp_y('PRED', self.MAX_LEN)
        tr_data = dict(df_tr, **df_pred)
        df_outlier = self.plot_cmp_y('OUTLIER', self.MAX_LEN)
        with open(os.path.join(self.pkl_path_mdl, 'lgbm_selected_features.pkl'), 'rb') as sf:
            selected_features = pickle.load(sf)

        for col_f in selected_features + [self.y_label]:
            fig, (ax) = plt.subplots(1, 1, figsize=(12, 8))
            if col_f in ("循环序号"):
                continue
            for key, i_df in tr_data.items():
                i_df = i_df.iloc[:-3, :]
                ax.plot(i_df[col_f], color='c', label=key + '_norm')
            for key, i_df in df_outlier.items():
                i_df = i_df.iloc[:-3, :]
                ax.plot(i_df[col_f], color='m', label=key + '_NG')

            ax.legend(loc='best', fontsize=8)
            ax.set_title(col_f + '_cmp')
            ax.set_xlabel('cycle NUM', fontsize=12)
            if '/' in col_f:
                col_f = col_f.replace('/', '-')
            NG_path = os.path.join(self.path_png_plt, 'kneepoint_norm_NG')
            if not os.path.exists(NG_path):
                os.makedirs(NG_path)
            save_path = os.path.join(NG_path, col_f + '_cmp_f.png')
            plt.savefig(save_path, dpi=self.dpi)

    def knee_point_fit(self):
        from collections import Counter
        df_tr_tmp = self.plot_cmp_y('TRAIN', self.MAX_LEN)
        df_pred_tmp = self.plot_cmp_y('PRED', self.MAX_LEN)
        df_tr = pd.concat(df_tr_tmp).reset_index()
        df_pred = pd.concat(df_pred_tmp).reset_index()
        df_union = pd.concat([df_tr, df_pred], axis=0)
        regr = LinearRegression()
        regr.fit(df_union[self.cycle_num].values.reshape(-1, 1), df_union[self.y_label].values.reshape(-1, 1))
        self.mdl_kpc = regr
        logger.info('residual trend fit done')
        normal_data = dict(df_tr_tmp, **df_pred_tmp)
        y_pred = []
        y_pred_up = []
        y_pred_low = []
        for _, idf in normal_data.items():
            y_predict = regr.predict(idf.reset_index()[self.cycle_num].values.reshape(-1, 1))
            kpc_err = (y_predict.ravel() - idf[self.y_label].values.ravel())
            resid_upper = y_predict + kpc_err.max()
            resid_lower = y_predict - 0.5 * abs(kpc_err.min())
            y_pred.append(y_predict)
            y_pred_up.append(resid_upper)
            y_pred_low.append(resid_lower)
        self.y_pred = np.average(y_pred, axis=0)
        self.y_pred_up = np.average(y_pred_up, axis=0)
        self.y_pred_low = np.average(y_pred_low, axis=0)
        logger.info('residual bounds done')

    def knee_point_pred(self, cyl_num):
        res_pred_dict = {}
        df_outlier = self.plot_cmp_y('OUTLIER', cyl_num)
        for filename, idf_outl in df_outlier.items():
            idf_outl = idf_outl.reset_index()
            y_predict = self.mdl_kpc.predict(idf_outl[self.cycle_num].values.reshape(-1, 1)).ravel()
            idf_outl['PRED'] = y_predict
            res_pred_dict[filename] = idf_outl
        return res_pred_dict


if __name__ == "__main__":
    kpc = KneeePointCalcu()
    kpc.knee_point_fit()
    kpc.plt_cmp_run()
    kpc.diff_cmp()
