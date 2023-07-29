# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     output_api
   Description :
   Author :       ASUS
   date：          2023-05-23
-------------------------------------------------
   Change Activity:
                   2023-05-23:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import json
import pandas as pd
from utils.params_config import ConfPath, ConfVars, ConstVars


class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)


class OutPutParms(ConfParams):
    def __init__(self):
        ConfParams.__init__(self)
        self.index_col_dqdv = ['SN']

    def output_dqdv(self, cols):
        json_dqdv = {}
        json_dqdv_delta = {}
        for fold in os.listdir(self.data_join_path):
            fold_path = os.path.join(self.data_join_path, fold)
            for ifold in os.listdir(fold_path):
                ifold_path = os.path.join(fold_path, ifold)
                for file in os.listdir(ifold_path):
                    if not file.endswith('summary.xlsx'):
                        continue
                    df_asm = pd.read_excel(os.path.join(ifold_path, file), sheet_name='cyl_data')
                    df_asm = df_asm.set_index(self.drop_cols).iloc[self.dqdv_initcycle:self.dqdv_maxcycle, :]
                    df_asm = df_asm.dropna(how='any', axis=0).loc[:, cols]
                    key = ':'.join([fold, ifold])
                    json_dqdv[key] = df_asm.to_json(orient="columns")

                    delta_dqdv_data = (df_asm.iloc[[-1], :].values - df_asm.iloc[[1], :].values).tolist()
                    df_delta_dqdv = pd.DataFrame(data=delta_dqdv_data, columns=cols, index=[key])
                    json_dqdv[key] = df_delta_dqdv.to_json(orient="columns")

        return json.dumps(json_dqdv), json.dumps(json_dqdv_delta)

    def output_capacity(self, x_range, y_range):
        json_true = {}
        json_pred = {}
        plt_cyl_path = os.path.join(self.pred_path, 'lgbm')
        for scellid in os.listdir(plt_cyl_path):
            file_cyl_path = os.path.join(plt_cyl_path, scellid)
            if not os.path.isdir(file_cyl_path):
                continue
            columns = ['slice_' + str(i) for i in x_range]
            df_cyl_pred = pd.DataFrame(columns=columns, index=y_range)
            df_true = pd.DataFrame(columns=columns + ['TRUE'], index=y_range)
            for cond_file in os.listdir(file_cyl_path):
                if not cond_file.endswith('.xlsx'):
                    continue
                df_res = pd.read_excel(os.path.join(file_cyl_path, cond_file), sheet_name=self.sheet_name)
                (row, col) = int(cond_file.split('-')[1][:-5]), 'slice_' + cond_file.split('-')[0]
                # i_df = df_res[df_res['SN'] == scellid].sort_values(by='SORT_DATE').iloc[-1:, :]
                i_df = df_res[df_res['SN'] == scellid].sort_values(by='CYCLE_NUM').iloc[-1:, :]
                df_cyl_pred.loc[row, col] = round(i_df['MAPE'][0], 3)
                df_true.loc[row, col] = round(i_df['PRED'][0], 4)
                df_true.loc[row, 'TRUE'] = round(i_df['TRUE'][0], 4)
            json_pred[scellid] = df_cyl_pred.to_json(orient="columns")
            json_true[scellid] = df_true.to_json(orient="columns")
        return json.dumps(json_true), json.dumps(json_pred)





if __name__ == "__main__":
    opm = OutPutParms()
    pv_cols = ['fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23']
    opm.output_dqdv(pv_cols)
    # opm.output_capacity(0, 0)
