# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     api_cyl_capacity.py
   Description :
   Author :       ASUS
   date：          2023/4/12
-------------------------------------------------
   Change Activity:
                   2023/4/12:
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
from xfun.z_post_plot.output_api import OutPutParms
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)


@app.route('/kpd_train', methods=["POST", "GET"])
def kpd_train():
    kpc = FitKPD()
    kpc.knee_point_fit('END_CAPACITY')
    return 'preclean done'


@app.route("/kpd_pred", methods=["POST", "GET"])
def kpd_pred():
    pkp = PredKPD()
    kpd_json = pkp.knee_crosspoint('END_CAPACITY')
    return kpd_json


@app.route('/capacity_preclean_train', methods=["POST", "GET"])
def preclean_train():
    lfp = LoadSmall()
    lfp.split_cycle('TRAIN')
    lfp.gen_count_features('TRAIN')
    return 'preclean done'


@app.route('/capacity_preclean_pred', methods=["POST", "GET"])
def preclean_pred():
    lfp = LoadSmall()
    lfp.split_cycle('PRED')
    lfp.gen_count_features('PRED')
    return 'preclean done'

@app.route('/capacity_preclean_all', methods=["POST", "GET"])
def preclean_all():
    pass

@app.route('/dqdv_train', methods=["POST", "GET"])
def dqdv_calcu_train():
    etf = ExtractFeaturesEC()
    etf.get_peak_valley('TRAIN')
    etf.lfp_pv_join('TRAIN')
    return 'train dqdv done'


@app.route('/dqdv_pred', methods=["POST", "GET"])
def dqdv_calcu_pred():
    etf = ExtractFeaturesEC()
    # etf.get_peak_valley('F19_22A01CA-add')
    # etf.lfp_pv_join('F19_22A01CA-add')
    pv_cols = ['fst_peak', 'sec_peak', 'fst_peak_x', 'sec_peak_x', 'area_q1', 'area1_q23']
    opm = OutPutParms()
    dqdv_json, json_dqdv_delta = opm.output_dqdv(pv_cols)

    return dqdv_json, json_dqdv_delta


@app.route('/capacity_fit', methods=["POST", "GET"])
def capacity_train():
    x_range = np.arange(20, 101, 25)
    y_range = np.arange(300, 451, 50)
    for step_x in x_range:
        for step_y in y_range:
            pcl = PreClean()
            pcl.gen_mdldata('TRAIN', step_x, step_y)
            pcl.gen_mdldata('PRED', step_x, step_y)
            # 3.model train
            for mdl_name in ['lgbm']:
                fas = FeatureAnsys()
                fas.features_calcu('TRAIN', mdl_name)
                mt = ModelTrain(mdl_name)
                mt.fit_mdl(mdl_name)
    return 'train done'


@app.route('/capacity_pred', methods=["POST", "GET"])
def capacity_pred():
    x_range = np.arange(50, 201, 25)
    y_range = np.arange(1000, 1451, 50)
    # for step_x in x_range:
    #     for step_y in y_range:
    #         for mdl_name in ['lgbm']:
    #             # 4.model predict
    #             mt = ModelPredict('PRED', mdl_name, step_x, step_y)
    #             mt.pred_tmp(mdl_name)
    opm = OutPutParms()
    js_true, js_pred = opm.output_capacity(x_range, y_range)
    return js_true, js_pred


if __name__ == '__main__':
    # 解决jsonify中文乱码问题
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=7777, debug=False)
