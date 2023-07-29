# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     outlier_kpc
   Description :
   Author :       ASUS
   date：          2023-05-24
-------------------------------------------------
   Change Activity:
                   2023-05-24:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
from collections import Counter
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils.params_config import ConfPath, ConfVars, ConstVars


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)


class OutlierModelSets:
    def __init__(self):
        self.out_frac = 0.1

    def iso_tree(self):
        clf = IsolationForest(n_estimators=10,
                              max_samples='auto',
                              contamination=self.out_frac,
                              max_features=1.0)
        return clf

    def lof(self):
        clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
        return clf

    def outlier_detect(self, mdl_name):
        mdl_set = {'lof': self.lof(),
                   'iso': self.iso_tree()
                   }
        return mdl_set[mdl_name]

    def lgbm_mdl(self):
        params = {
            'learning_rate': 0.1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 3,
            'objective': 'binary',
            'class_weight': {}
        }
        mdl = lgb.LGBMClassifier(**params, )
        return mdl


class BaseLoad(ConfParams):
    def __init__(self):
        ConfParams.__init__(self)
        self.cols = [
            # 'area1_q23',
            'sec_peak_x', 'STAT_ENDVOL', 'MEAN_CHGVOL']
        # self.cols = ['CHG_INITVOL', 'CHG_AVGVOL', 'STAT_ENDVOL', 'MEAN_CHGVOL', 'RV_VOLTAGE', 'CYCLE_INT']
        # self.add_cols = ['SV_VOLTAGE', 'Q_EPSILON', 'CC_EPSILON', 'DISCHG_INITVOL', 'DELTA_STATVOL',
        #                  'fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23']

    def load_data(self, RUN_MODE, cols_set):
        """
        1.smooth data
        2.drop nan avg_features(1500cycles) + peak (800cycles)
          may drop 800:1500cycles
        """
        data_dict = {}
        fold_path = os.path.join(self.data_join_path, RUN_MODE)
        for cellid_fold in os.listdir(fold_path):
            ifold_path = os.path.join(fold_path, cellid_fold)
            for file in os.listdir(ifold_path):
                if not file.endswith('summary.xlsx'):
                    continue
                df_asm = pd.read_excel(os.path.join(ifold_path, file), sheet_name='cyl_data')
                df_asm = df_asm.set_index(self.drop_cols).rolling(window=10, min_periods=3).mean()
                df_asm.dropna(how='any', axis=0, inplace=True)
                for col in cols_set:
                    df_cols = df_asm[[col]].dropna(how='any', axis=0)
                    if col.startswith('Q'):
                        df_col_scal = df_cols
                        df_col_scal = df_col_scal[df_col_scal.abs() < 1.5]
                    elif (col.endswith('RV') or col.endswith('SV')):
                        df_col_scal = df_cols
                        df_col_scal = df_col_scal[df_col_scal.abs() < 0.05]
                    else:
                        # SACLED DATA BY VALUES[0]
                        df_col_scal = df_cols / df_cols.values[0]
                    df_asm[col] = df_col_scal
                cell_index = cellid_fold + ':' + file.split('.')[0]
                df_avg_cell = self.extract_avg_features(df_asm)

                data_dict[cell_index] = df_avg_cell
        # df_cell = pd.concat(data_dict, axis=0)
        return data_dict

    @staticmethod
    def extract_avg_features(df):
        diff_area_q23_0t400 = df['area1_q23'].values[0] - df['area1_q23'].values[400]
        diff_sec_peak_x_0t400 = df['sec_peak_x'].values[0] - df['sec_peak_x'].values[400]
        diff_STAT_ENDVOL_0t400 = df['STAT_ENDVOL'].values[0] - df['STAT_ENDVOL'].values[400]
        avg_STAT_ENDVOL_0t400 = df['STAT_ENDVOL'].values[0:400].mean()
        diff_MEAN_CHGVOL_0t400 = df['MEAN_CHGVOL'].values[0] - df['MEAN_CHGVOL'].values[400]
        avg_MEAN_CHGVOL_0t400 = df['MEAN_CHGVOL'].values[0:400].mean()

        diff_area_q23_0t500 = df['area1_q23'].values[0] - df['area1_q23'].values[500]
        diff_sec_peak_x_0t500 = df['sec_peak_x'].values[0] - df['sec_peak_x'].values[500]
        diff_STAT_ENDVOL_0t500 = df['STAT_ENDVOL'].values[0] - df['STAT_ENDVOL'].values[500]
        avg_STAT_ENDVOL_0t500 = df['STAT_ENDVOL'].values[0:500].mean()
        diff_MEAN_CHGVOL_0t500 = df['MEAN_CHGVOL'].values[0] - df['MEAN_CHGVOL'].values[500]
        avg_MEAN_CHGVOL_0t500 = df['MEAN_CHGVOL'].values[0:500].mean()

        columns = ['diff_area_q23_0t400', 'diff_sec_peak_x_0t400', 'diff_STAT_ENDVOL_0t400',
                   'avg_STAT_ENDVOL_0t400', 'diff_MEAN_CHGVOL_0t400', 'avg_MEAN_CHGVOL_0t400']
        data = [diff_area_q23_0t400, diff_sec_peak_x_0t400, diff_STAT_ENDVOL_0t400,
                avg_STAT_ENDVOL_0t400, diff_MEAN_CHGVOL_0t400, avg_MEAN_CHGVOL_0t400,
                diff_area_q23_0t500, diff_sec_peak_x_0t500, diff_STAT_ENDVOL_0t500,
                avg_STAT_ENDVOL_0t500, diff_MEAN_CHGVOL_0t500, avg_MEAN_CHGVOL_0t500
                ]
        df_avg = pd.DataFrame(columns=columns, data=np.array(data).reshape(1, 16))
        return df_avg

    def get_data(self):
        normal_dict = {}
        pred_normal_dict = {}
        outlier_dict = {}

        for fold_sys in os.listdir(self.data_join_path):
            tmp_dict = self.load_data(fold_sys, self.cols)
            if fold_sys.endswith('outlier'):
                outlier_dict.update(tmp_dict)
            elif fold_sys.endswith('pred'):
                pred_normal_dict.update(tmp_dict)
            else:
                normal_dict.update(tmp_dict)

        df_normal = pd.concat(normal_dict, axis=0)
        df_pred_normal = pd.concat(normal_dict, axis=0)
        df_outlier = pd.concat(outlier_dict, axis=0)
        df_ok = pd.concat([df_normal, df_pred_normal], axis=0)

        df_outlier.to_excel(os.path.join(self.diff_dqdv_path, 'diff_dqdv_outlier.xlsx'), sheet_name='outlier')
        df_ok.to_excel(os.path.join(self.diff_dqdv_path, 'diff_dqdv_ok.xlsx'), sheet_name='outlier')
        return normal_dict, pred_normal_dict, outlier_dict


class OutlierDetectKPC(BaseLoad, OutlierModelSets):
    def __init__(self):
        BaseLoad.__init__(self)
        OutlierModelSets.__init__(self)
        self.normal_dict, self.pred_normal_dict, self.outlier_dict = self.get_data()
        print()

    def outlier_fit(self, mdl_name='lof'):
        # df_normal = self.filter_bycycle(self.normal_dict, cycle_step)
        df_normal = pd.concat(self.normal_dict, axis=0)
        self.outl_mdl = self.outlier_detect(mdl_name)
        self.outl_mdl.fit(df_normal)
        return

    @staticmethod
    def filter_bycycle(dict_data, cycle_step):
        new_noutlier_dict = {}
        for key, idf in dict_data.items():
            # new_noutlier_dict[key] = idf[idf['CYCLE_INT'] <= cycle_step]
            new_noutlier_dict[key] = idf[(idf['CYCLE_INT'] <= cycle_step[1]) &
                                         (idf['CYCLE_INT'] >= cycle_step[0])
                                         ]
        df_outlier = pd.concat(new_noutlier_dict, axis=0)
        return df_outlier

    def outlier_transform(self):
        # df_pred_normal = self.filter_bycycle(self.pred_normal_dict, cycle_step)
        # df_outlier = self.filter_bycycle(self.outlier_dict, cycle_step)

        df_outlier = pd.concat(self.outlier_dict, axis=0)
        df_pred_normal = pd.concat(self.pred_normal_dict, axis=0)
        df_pred_asm = pd.concat([df_outlier, df_pred_normal], axis=0)
        pred_label = self.outl_mdl.predict(df_pred_asm)
        df_pred_label = pd.DataFrame(columns=['TRUE', 'PRED'], index=df_pred_asm.index)
        df_pred_label['PRED'] = pred_label
        df_pred_label['TRUE'] = [-1] * df_outlier.shape[0] + [1] * df_pred_normal.shape[0]
        return df_pred_label

    def conf_matrix(self, df_pred_res, cyl_str):
        plt_path_cmt = os.path.join(self.path_png_plt, 'confusion_matrix')
        if not os.path.exists(plt_path_cmt):
            os.makedirs(plt_path_cmt)
        # df_pred_res = pd.concat(df_label_dict, axis=0)
        # df_pred_res = df_pred_res.reset_index().drop(columns=['SORT_DATE', 'CYCLE_NUM'], axis=1)
        # sn_list = df_pred_res.index.unique().tolist()
        # df_label_summary = pd.DataFrame(columns=['TRUE', 'PRED'])
        # for sn in sn_list:
        #     idf = df_pred_res[df_pred_res['SN'] == sn]
        #     if (idf['PRED'] == -1).any():
        #         y_pred = -1
        #     else:
        #         y_pred = 1
        #     df_label_summary.loc[sn, :] = [idf['TRUE'].unique()[0], y_pred]

        f, ax = plt.subplots()
        cmatrix = confusion_matrix(df_pred_res['TRUE'].values.tolist(), df_pred_res['PRED'].values.tolist(),
                                   labels=[-1, 1])
        sns.heatmap(cmatrix, annot=True, ax=ax)
        ax.set_title('confusion matrix')
        ax.set_xlabel('predict')
        ax.set_ylabel('true')
        # writer_label = pd.ExcelWriter(os.path.join(plt_path_cmt, 'df_outlier_pred.xlsx'))
        # writer_label_summary = pd.ExcelWriter(os.path.join(plt_path_cmt, 'label_summary.xlsx'))
        # df_pred_res.to_excel(writer_label, sheet_name=cyl_str)
        # df_label_summary.to_excel(writer_label_summary, sheet_name=cyl_str)
        df_pred_res.to_excel(os.path.join(plt_path_cmt, cyl_str + 'df_outlier_pred.xlsx'), sheet_name=cyl_str)
        # df_label_summary.to_excel(os.path.join(plt_path_cmt, cyl_str + 'label_summary.xlsx'), sheet_name=cyl_str)
        plt.savefig(os.path.join(plt_path_cmt, cyl_str + '.png'), dpi=self.dpi)
        plt.close()

    def clf(self):
        df_normal = pd.concat(self.normal_dict, axis=0)
        df_outlier = pd.concat(self.outlier_dict, axis=0)
        df_pred_normal = pd.concat(self.pred_normal_dict, axis=0)
        df_outlier_tr = df_outlier.iloc[:4, :]
        df_outlier_pred = df_outlier.iloc[4:, :]
        df_tr = pd.concat([df_normal, df_outlier_tr], axis=0)
        df_pred = pd.concat([df_pred_normal, df_outlier_pred], axis=0)

        y_t = [1] * df_normal.shape[0] + [-1] * df_outlier_tr.shape[0]
        y_ptrue = [1] * df_pred_normal.shape[0] + [-1] * df_outlier_pred.shape[0]

        params = {
            'learning_rate': 0.1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 3,
            'objective': 'binary',
            # 'class_weight': {1: 0.5, -1: 0.5}
            'class_weight': 'balanced'

        }
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(df_tr, y_t)
        y_preds = mdl.predict(df_pred)

        df_pred_label = pd.DataFrame(columns=['TRUE', 'PRED'], index=df_pred.index)
        df_pred_label['PRED'] = y_preds
        df_pred_label['TRUE'] = y_ptrue
        self.conf_matrix(df_pred_label, 'lgbm_tree')


def run_kpc2():
    step, max_cycle = 100, 601
    cyl_list = list(range(0, max_cycle, step))
    cyl_slice_list = [cyl_list[i:i + 2] for i in range(0, len(cyl_list) - 1)]
    odk = OutlierDetectKPC()
    for cyl_num in cyl_slice_list:
        df_label = {}
        odk.outlier_fit(cyl_num, 'lof')
        df_pred_res = odk.outlier_transform(cyl_num)
        df_label[cyl_num[0]] = df_pred_res
        cyl_str = '-'.join(['cyl', str(cyl_num[0]) + '-' + str(cyl_num[1]) + '-'])
        odk.conf_matrix(df_label, cyl_str)
        print()


def run_kpc1():
    odk = OutlierDetectKPC()
    odk.outlier_fit('iso')
    df_pred_res = odk.outlier_transform()
    odk.conf_matrix(df_pred_res, 'cyl_all')
    print()


def run_kpc3():
    odk = OutlierDetectKPC()
    odk.clf()


if __name__ == "__main__":
    # run_kpc1()
    run_kpc3()
