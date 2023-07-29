# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     clp_main_run_ncm.py
   Description :
   Author :       ASCEND
   date：          2023/1/11
-------------------------------------------------
   Change Activity:
                   2023/1/11:
-------------------------------------------------
"""
__author__ = 'ASCEND'

import numpy as np
from xfun.parse_data.parse_func_asm import LoadSmall
from xfun.preclean.merge_features import PreClean
from xfun.preclean.gen_features_dqdv import ExtractFeaturesEC
from xfun.train_model.corr_features_calcu import FeatureAnsys
from xfun.train_model.models_calcu import ModelTrain
from xfun.pred_model.model_preds import ModelPredict
from xfun.z_post_plot.plt_mdl_data import PostVisual


def run_cycle():
    # 1.parse data
    """
        parse_func.gen_plt_fun define features list:
           drop int   cycle  startswith： 0, 1, 2, 3
            index:unnamed
        ['END_CAPACITY', 'DISCHG_ENDENERGY', 'DISCHG_INITVOL', 'DISCHG_AVGVOL',
       'CHG_ENDCAPACITY', 'CHG_ENDENERGY', 'CHG_INITVOL', 'CHG_AVGVOL',
       'STAT_ENDVOL', 'DELTA_STATVOL',
       'MEAN_CHGVOL', 'MEAN_DISCHGVOL', 'RV_VOLTAGE', 'SV_VOLTAGE',
       'Q_EPSILON', 'CC_EPSILON', 'CV_Q',
       'soc1000_du', 'soc2000_du', 'soc3000_du',
       'soc1000_avgu', 'soc2000_avgu', 'soc3000_avgu',
       'CYCLE_INT', 'CYCLE_NUM']

    """
    lfp = LoadSmall()
    # lfp.split_cycle('TRAIN')
    # lfp.split_cycle('PRED')
    lfp.gen_count_features('TRAIN')
    lfp.gen_count_features('PRED')

    # 2.preclean
    """
        get_peak_valley
        gen_mdldata: gen new features CYCLE_INT
        pred_range = [
                    (20, 200), (20, 250), (20, 300), (20, 320),(50, 200), (50, 250), (50, 300), (50, 320),
                    (80, 200), (80, 250), (80, 300), (80, 320),(100, 200), (100, 250), (100, 300), (100, 320)
                   ]
    """

    etf = ExtractFeaturesEC()
    etf.get_peak_valley('TRAIN')
    etf.get_peak_valley('PRED')
    # etf.lfp_pv_join('TRAIN')
    # etf.lfp_pv_join('PRED')
    # x_range = np.arange(50, 201, 25)
    # y_range = np.arange(1000, 1451, 50)
    # for step_x in x_range:
    #     for step_y in y_range:
    #         pcl = PreClean()
    #         pcl.gen_mdldata('TRAIN', step_x, step_y)
    #         pcl.gen_mdldata('PRED', step_x, step_y)
    #         # 3.model train
    #         for mdl_name in ['lgbm']:
    #             fas = FeatureAnsys()
    #             fas.features_calcu('TRAIN', mdl_name)
    #             mt = ModelTrain(mdl_name)
    #             mt.fit_mdl(mdl_name)
    #             # 4.model predict
    #             mt = ModelPredict('PRED', mdl_name, step_x, step_y)
    #             mt.batch_pred_cyl(mdl_name, 'PRED_ONLY')
    # 5.plot postvisual
    # pv = PostVisual('TRAIN', 'lgbm', x_range, y_range)
    # pv.plt_dist()
    # pv.plt_heatmap_corr()
    # pv.plt_cap_corr()
    # pv.plt_feature_imp()
    # pv.plt_cyl_pred()
    # pv.plt_trend1_u1()
    # pv.plt_trend1_u2()
    # pv.plt_trend1_u3()
    # pv.plt_trend2_dqdv_ux()
    # pv.plt_trend2_dqdv_yheight()
    # pv.plt_trend2_dqdv_areas()

if __name__ == "__main__":
    run_cycle()
