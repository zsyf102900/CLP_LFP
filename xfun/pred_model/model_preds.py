# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_preds
   Description :
   Author :       ASUS
   date：          2023/3/6
-------------------------------------------------
   Change Activity:
                   2023/3/6:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from multiprocessing import Pool
from xfun.train_model.models_set import ModelsAsm
from utils.params_config import ConfPath, ConfVars, ConstVars
from log_conf.logger import logger


class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)
        ModelsAsm.__init__(self)


# ************************@@@@@***************************#
#  model pred
#  model ecaluate
# ************************@@@@@***************************#

class ModelPredict(ConfParams):
    """
    class ModelPredict：
             batch_pred
    Attributes:
        pred_all
            batch_pred
                pred_fun
            eval_mape
    """

    def __init__(self, run_mode, mdl_name, step_x, step_y):
        """
        feature columns: x_columns>raw_features>select_feature
        self.pred_slice = [[pred_df.iloc[i:i + self.MAX_PRED_SLICE, :]]
                                for i in range(0, pred_df.shape[0], self.MAX_PRED_SLICE)]
        :param stepx_dischg_edv:
        """

        ConfParams.__init__(self)
        self.step_x, self.step_y = step_x, step_y
        with open(os.path.join(self.pkl_path_mdl, mdl_name + '_raw_features.pkl'), 'rb') as rf:
            self.raw_features = pickle.load(rf)
        with open(os.path.join(self.pkl_path_mdl, mdl_name + '_std.pkl'), 'rb') as std:
            self.std = pickle.load(std)
        with open(os.path.join(self.pkl_path_mdl, mdl_name + '_mas.pkl'), 'rb') as mas:
            self.mas = pickle.load(mas)
        with open(os.path.join(self.pkl_path_mdl, mdl_name + '_selected_features.pkl'), 'rb') as sf:
            self.select_feature = pickle.load(sf)
        with open(os.path.join(self.pkl_path_mdl, mdl_name + '_fit_mdl.pkl'), 'rb') as mdl:
            self.mdl = pickle.load(mdl)
        with open(os.path.join(self.ts_avg_fatures_path, run_mode + '_cap_join_mdldata.pkl'), 'rb') as cj:
            self.pred_df = pickle.load(cj)
        print()

    def pred_fun(self, pred_path, slice_df, k):
        # logger.info('fold :{} preds ...'.format(k))
        pred_t = []
        for i, df in enumerate(slice_df):
            all_columns = self.raw_features
            df_ary = pd.DataFrame(self.mas.transform(self.std.transform(df.loc[:, all_columns])),
                                  columns=all_columns)
            y_pred = self.mdl.predict(df_ary[self.select_feature])
            df_valid = pd.DataFrame(columns=['TRUE', 'PRED', 'MAPE'], index=df.index)
            df_valid['TRUE'], df_valid['PRED'] = df[self.y_label].abs(), np.abs(y_pred)
            df_valid['MAPE'] = 100 * (1 - abs(df_valid['TRUE'] - df_valid['PRED']) / df_valid['TRUE'].abs())
            sfc_num = df.index[-1]
            try:
                # df_valid.to_csv(pred_path + sfc_num + '_pred.csv', sep=',', index=True, encoding="utf_8_sig")
                pass
            except Exception as err:
                logger.info('read errors: {},file:{}'.format(err, pred_path + sfc_num))
            pred_t.append(df_valid)
        # logger.info('fold :{} preds ends'.format(k))
        return pred_t

    def pred_fun_only(self, pred_path, slice_df, k, ):
        # logger.info('fold :{} preds ...'.format(k))
        pred_t = []
        for i, df in enumerate(slice_df):
            all_columns = self.raw_features
            df_ary = pd.DataFrame(self.mas.transform(self.std.transform(df.loc[:, all_columns])),
                                  columns=all_columns)
            y_pred = self.mdl.predict(df_ary[self.select_feature])
            df_valid = pd.DataFrame(columns=['PRED'], data=y_pred)
            sfc_num = df.index[-1]
            try:
                pass
            except Exception as err:
                logger.info('read errors: {},file:{}'.format(err, pred_path + sfc_num))
            pred_t.append(df_valid)
        # logger.info('fold :{} preds ends'.format(k))
        return pred_t

    def batch_pred(self, mdl_name):
        """
        :return:
        """
        res_asyc = []
        pred_path_t = os.path.join(self.pred_path, mdl_name)

        if not os.path.exists(pred_path_t):
            os.makedirs(pred_path_t)

        pool_num = 10
        pool = Pool(pool_num)
        for k, slice_df in enumerate(self.pred_slice):
            # self.pred_fun(pred_path_t, slice_df, k,)
            res = pool.apply_async(self.pred_fun, args=(pred_path_t, slice_df, k,))
            res_asyc.append(res)
        pool.close()
        pool.join()
        res_df = []
        for res in res_asyc:
            res_df.append(res.get())
        predfullcap_df = pd.concat(list(chain(*res_df)), axis=0)
        # predfullcap_df.to_csv(os.path.join(pred_path_t, 'fullcap_preds.csv'), index=True, encoding="utf_8_sig")
        predfullcap_df.to_excel(os.path.join(pred_path_t, 'fullcap_preds.xlsx'),
                                sheet_name='cyl_data',
                                index=True,
                                encoding="utf_8_sig")
        logger.info('pred done')
        pool.close()
        pool.join()

    def batch_pred_cyl(self, mdl_name, PRED_MODE):
        """
        self.pred_slice = [self.pred_df.iloc[i:i + 80, :] for i in range(0, self.pred_df.shape[0], 80)]
        :return:

        """
        pred_path_t = os.path.join(self.pred_path, mdl_name + '_' + PRED_MODE)
        if not os.path.exists(pred_path_t):
            os.makedirs(pred_path_t)

        prde_fun_dict = {'PRED_VALID': self.pred_fun,
                         'PRED_ONLY': self.pred_fun_only}
        prde_fun_select = prde_fun_dict[PRED_MODE]
        logger.info('preds ing')
        for k, slice_df in enumerate(self.pred_df):
            fold = slice_df.reset_index()['SN'].unique()[0]
            res = prde_fun_select(pred_path_t, [slice_df], k)
            predfullcap_df = res[0]
            fold_file = os.path.join(pred_path_t, fold)
            if not os.path.exists(fold_file):
                os.makedirs(fold_file)
            pred_sv_name = os.path.join(fold_file, '{}-{}.xlsx'.format(self.step_x, self.step_y))
            predfullcap_df.to_excel(pred_sv_name, index=True, encoding="utf_8_sig", sheet_name=self.sheet_name)
        logger.info('pred done')

    @staticmethod
    def _mape(y1, y2):
        mape_series = np.abs(y1 - y2) / np.maximum(y1, y2)
        avg_mape = 1 - np.average(mape_series)
        min_mape, max_mape = np.min(mape_series), np.max(mape_series)
        return avg_mape, min_mape, max_mape

    def eval_mape(self, mdl_type):
        for mdl_name in os.listdir(self.pred_path):
            if not mdl_name.startswith(mdl_type):
                continue
            df_tail = []
            pred_path_t = os.path.join(self.pred_path, mdl_name)
            for file_name in os.listdir(pred_path_t):
                if not file_name.endswith('.xlsx'):
                    continue
                path = os.path.join(pred_path_t, file_name)
                df_tail.append(pd.read_excel(path, sheet_name=self.sheet_name))

            df_tail = pd.concat(df_tail, axis=0)
            avg_mape, min_mape, max_mape = self._mape(df_tail['TRUE'].values[-1], df_tail['PRED'].values[-1])
            logger.info(
                'mdl_name:{},avg_mape:{},min_mape:{},max_mape:{}'.format(mdl_name, avg_mape, min_mape, max_mape))
            logger.info('*********************  {0} mape:{1} done *********************\n\n'
                        .format(mdl_name, round(avg_mape, 4)))

    def pred_all(self, mdl_name):
        logger.info('preds_df len:{} \n raw_columns:{}'.format(len(self.pred_df), self.pred_df[0].columns.tolist()))
        pred_df = pd.concat(self.pred_df, axis=0)
        self.pred_slice = [[pred_df[i:i + self.MAX_PRED_SLICE]]
                           for i in range(0, pred_df.shape[0], self.MAX_PRED_SLICE)]
        logger.info('preds_df len:{},selected_df slice:{}'.format(pred_df.shape[0], len(self.pred_slice)))
        self.batch_pred(mdl_name)
        # self.eval_mape(mdl_name)
        pass

    def pred_slice_df(self, mdl_name):
        logger.info('preds_df len:{} \n raw_columns:{}'.format(self.pred_df.shape[0], self.pred_df.columns.tolist()))

        self.pred_slice = [[self.pred_df.iloc[i:i + self.MAX_PRED_SLICE, :]]
                           for i in range(0, self.pred_df.shape[0], self.MAX_PRED_SLICE)]
        logger.info('preds_df len:{},selected_df slice:{}'.format(self.pred_df.shape[0], len(self.pred_slice)))
        self.batch_pred(mdl_name)
        self.eval_mape(mdl_name)
        pass


if __name__ == "__main__":
    # for mdl_name in ['lgbm']:
    #     mt = ModelPredict('PRED', mdl_name,step_x, step_y)
    #     mt.pred_tmp(mdl_name)
    pass
