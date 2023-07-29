# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_sdk
   Description :
   Author :       ASUS
   date：          2023-07-28
-------------------------------------------------
   Change Activity:
                   2023-07-28:
-------------------------------------------------
"""
__author__ = 'ASUS'

import numpy as np
from xfun.parse_data.parse_func_asm import LoadSmall
from xfun.preclean.merge_features import PreClean
from xfun.preclean.gen_features_dqdv import ExtractFeaturesEC
from xfun.train_model.corr_features_calcu import FeatureAnsys
from xfun.train_model.models_calcu import ModelTrain
from xfun.pred_model.model_preds import ModelPredict
from xfun.outlier_kneepoint_detect.calcu_knee_point import BaseData
from xfun.outlier_kneepoint_detect.calcu_knee_point import FitKPD
from xfun.outlier_kneepoint_detect.calcu_knee_point import PredKPD


def gen_data():
    lfp = LoadSmall()
    # ********* run split data *********
    lfp.split_cycle('TRAIN')
    lfp.split_cycle('PRED')
    lfp.split_cycle('PRED_OUTLIER')
    # ********* run split data *********

    # ********* gen count_endvalues data *********
    lfp.gen_count_features('TRAIN')
    lfp.gen_count_features('PRED')
    lfp.gen_count_features('PRED_OUTLIER')
    # ********* gen count_endvalues data *********

    etf = ExtractFeaturesEC()
    # ********* gen count_peakvalues data *********
    etf.get_peak_valley('TRAIN')
    etf.get_peak_valley('PRED')
    etf.get_peak_valley('PRED_OUTLIER')
    # ********* gen count_peakvalues data *********

    # ********* join peak-endv data *********
    etf.lfp_pv_join('TRAIN')
    etf.lfp_pv_join('PRED')
    etf.lfp_pv_join('PRED_OUTLIER')
    # ********* join peak-endv data *********

def train_mdl():
    x_range = np.arange(100, 501, 100)
    y_range = np.arange(1800, 2001, 50)
    for step_x in x_range:
        for step_y in y_range:
            pcl = PreClean()
            pcl.gen_mdldata('TRAIN', step_x, step_y)
            # 3.model train
            for mdl_name in ['lgbm']:
                fas = FeatureAnsys()
                fas.features_calcu('TRAIN', mdl_name)
                mt = ModelTrain(mdl_name)
                mt.fit_mdl(mdl_name)

def predict_mdl():
    x_range = np.arange(100, 501, 100)
    y_range = np.arange(1800, 2001, 50)
    for step_x in x_range:
        for step_y in y_range:
            pcl = PreClean()
            pcl.gen_mdldata('PRED', step_x, step_y)
            # 3.model train
            for mdl_name in ['lgbm']:

                # 4.model predict
                mt = ModelPredict('PRED', mdl_name, step_x, step_y)
                mt.batch_pred_cyl(mdl_name, 'PRED_ONLY')

def run_all():
    gen_data()
    train_mdl()
    predict_mdl()


