# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     demo-test
   Description :
   Author :       ASUS
   date：          2023/3/22
-------------------------------------------------
   Change Activity:
                   2023/3/22:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import numpy as np
import pandas as pd
import lightgbm as lgb


class DemoValid:

    def __init__(self):

        self.src_path = r'E:\code_git\抚州LFP循环2\model_data\data_join\TRAIN'
        self.prd_path = r'E:\code_git\抚州LFP循环2\model_data\png_plt'
        self.tr = None
        self.pred = None
        self.mdl = None

    def split_data(self):
        self.src_path = r'E:\code_git\抚州LFP循环2\model_data\data_join\TRAIN'
        for fold_file in os.listdir(self.src_path):
            path_fold = os.path.join(self.src_path, fold_file)
            for file in os.listdir(path_fold):
                if not file.endswith('summary'):
                    continue
                if not file.startswith('A1'):
                    continue
                file_path = os.path.join(path_fold, file)
                df_cyl = pd.read_excel(file_path, sheet_name='cyl_data')
                cyl_lambda = lambda x: (int(x.split('-')[0]) - 1) * 100 + int(x.split('-')[1])
                df_cyl['CYCLE_INT'] = df_cyl['CYCLE_NUM'].apply(cyl_lambda)
                self.tr = df_cyl[df_cyl['CYCLE_INT'] <= 100].set_index(['CYCLE_NUM', 'SORT_DATE', 'SN'])
                self.pred = df_cyl[(df_cyl['CYCLE_INT'] >= 300) & (df_cyl['CYCLE_INT'] <= 320)].set_index(
                    ['CYCLE_NUM', 'SORT_DATE', 'SN'])

        return

    def train_data(self):
        params = {
            'learning_rate': 0.1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 6,
            'objective': 'regression'
        }
        self.mdl = lgb.LGBMRegressor(**params)
        self.mdl.fit(self.tr.drop(columns='END_CAPACITY', axis=1), self.tr['END_CAPACITY'])

    def pred_data(self):
        self.prd_path = r'E:\code_git\抚州LFP循环2\model_data\png_plt'
        y_pred = self.mdl.predict(self.pred.drop(columns='END_CAPACITY', axis=1))
        df_pred = self.pred['END_CAPACITY']
        df_pred['PRED'] = y_pred
        save_file = os.path.join(self.prd_path, 'pred.xlsx')
        df_pred.to_excel(save_file, sheet_name='cyl_data')
        pass

    def run(self):
        self.src_path = r'E:\code_git\抚州LFP循环2\model_data\data_join\PRED'

        file_path = os.path.join(self.src_path, 'F2.xlsx', 'F2_summary.xls')
        df_cyl = pd.read_excel(file_path, sheet_name='cyl_data')
        cyl_lambda = lambda x: (int(x.split('-')[0]) - 1) * 100 + int(x.split('-')[1])
        df_cyl['CYCLE_INT'] = df_cyl['CYCLE_NUM'].apply(cyl_lambda)
        tr = df_cyl[df_cyl['CYCLE_INT'] <= 20].set_index(['CYCLE_NUM', 'SORT_DATE', 'SN'])
        pred = df_cyl[(df_cyl['CYCLE_INT'] >= 300) & (df_cyl['CYCLE_INT'] <= 320)].set_index(
            ['CYCLE_NUM', 'SORT_DATE', 'SN'])

        params = {
            'learning_rate': 0.1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 6,
            'objective': 'regression'
        }
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(tr.drop(columns='END_CAPACITY', axis=1), tr['END_CAPACITY'])

        self.prd_path = r'E:\code_git\抚州LFP循环2\model_data\png_plt'
        y_pred = mdl.predict(pred.drop(columns='END_CAPACITY', axis=1))

        df_true = pred['END_CAPACITY']
        df_pred = pd.DataFrame(columns=['TRUE', 'PRED', 'MAPE'])
        df_pred['TRUE'] = pred['END_CAPACITY']
        df_pred['PRED'] = y_pred
        df_pred['MAPE'] = (df_pred['TRUE'] - df_pred['PRED']) / df_pred['TRUE'] + 1
        save_file = os.path.join(self.prd_path, 'pred.xlsx')
        df_pred.to_excel(save_file, sheet_name='cyl_data')


if __name__ == "__main__":
    dtp = DemoValid()
    # dtp.split_data()
    # dtp.train_data()
    # dtp.pred_data()
    dtp.run()
