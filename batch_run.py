# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     batch_run.py
   Description :
   Author :       ASUS
   date：          2023-05-20
-------------------------------------------------
   Change Activity:
                   2023-05-20:
-------------------------------------------------
"""
__author__ = 'ASUS'

import os
import numpy as np
from xfun.parse_data.parse_func_asm import LoadSmall
from xfun.preclean.merge_features import PreClean
from xfun.preclean.gen_features_dqdv import ExtractFeaturesEC
from xfun.train_model.corr_features_calcu import FeatureAnsys
from xfun.train_model.models_calcu import ModelTrain
from xfun.pred_model.model_preds import ModelPredict
from xfun.z_post_plot.plt_mdl_data import PostVisual
from log_conf.logger import logger


def run_cycle():
    # 1.parse data
    """
        parse_func.gen_plt_fun define features list:
           drop int   cycle  startswith： 0, 1, 2, 3
            index:unnamed
        ['END_CAPACITY', 'DISCHG_ENDENERGY', 'DISCHG_INITVOL', 'DISCHG_AVGVOL',
           'CHG_ENDCAPACITY', 'CHG_ENDENERGY', 'CHG_INITVOL', 'CHG_AVGVOL',
           'STAT_ENDVOL', 'DELTA_STATVOL', 'CYCLE_NUM']
            if (fold.startswith('SC9') or fold.startswith('SC19') or fold.startswith('G6')
                or fold.startswith('SC14') or fold.startswith('SC17') or fold.startswith('SC18')
                or fold.startswith('I21')):
            continue

    """
    lfp = LoadSmall()
    lfp.split_cycle('TRAIN')
    lfp.split_cycle('PRED')
    lfp.split_cycle('PRED_OUTLIER')

    lfp.gen_count_features('TRAIN')
    lfp.gen_count_features('PRED')
    lfp.gen_count_features('PRED_OUTLIER')

    etf = ExtractFeaturesEC()
    # ********* review dqdv pv profile *********
    # etf.get_peak_dqdv_review('TRAIN')
    # etf.get_peak_dqdv_review('PRED')
    # etf.get_peak_dqdv_review('PRED_OUTLIER')
    # etf.get_peak_valley_dbg()
    # ********* review dqdv pv profile *********

    # ********* review dvdq pv profile *********
    # etf.get_peak_dvdq_review('TRAIN')
    # etf.get_peak_dvdq_review('PRED')
    # ********* review dvdq pv profile *********

    etf.get_peak_valley('TRAIN')
    etf.get_peak_valley('PRED')
    etf.get_peak_valley('PRED_OUTLIER')

    etf = ExtractFeaturesEC()
    etf.lfp_pv_join('TRAIN')
    etf.lfp_pv_join('PRED')
    etf.lfp_pv_join('PRED_OUTLIER')

    x_range = np.arange(100, 501, 100)
    y_range = np.arange(1800, 2001, 50)
    # x_range = np.arange(100, 201, 50)
    # y_range = np.arange(900, 1201, 50)
    # x_range = np.arange(301, 351, 50)
    # y_range = np.arange(1200, 1251, 100)
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
                # 4.model predict
                mt = ModelPredict('PRED', mdl_name, step_x, step_y)
                mt.batch_pred_cyl(mdl_name, 'PRED_ONLY')
                # mt.batch_pred_cyl(mdl_name, 'PRED_VALID')

    # 5.plot postvisual
    pv = PostVisual('lgbm', 'PRED_VALID',  x_range, y_range)
    # pv.outlier_ts()
    # pv.plt_dqdv_diff()

    # pv.plt_cmp_allbysysid()
    pv.cut_select()
    # pv.plt_cmp_QVbysysid()
    # pv.plt_cmp_all()
    # pv.plt_dist()
    # pv.plt_heatmap_corr()
    # pv.plt_cap_corr()
    # pv.plt_feature_imp()
    # pv.plt_cyl_pred()
    # pv.plt_cyl_pred_asm()
    # pv.plt_trend1_u1()
    # pv.plt_trend1_u2()
    # pv.plt_trend1_u3()
    # pv.plt_trend2_dqdv_ux()
    # pv.plt_trend2_dqdv_yheight()
    # pv.plt_trend2_dqdv_areas()


if __name__ == "__main__":
    run_cycle()
