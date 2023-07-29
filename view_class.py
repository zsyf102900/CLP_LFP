# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     view_class
   Description :
   Author :       ASUS
   date：          2023/3/28
-------------------------------------------------
   Change Activity:
                   2023/3/28:
-------------------------------------------------
"""
__author__ = 'ASUS'

# from clp_main_run_lfp import run_cycle
from batch_run import run_cycle
from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from pycallgraph2 import Config
from pycallgraph2 import GlobbingFilter

if __name__ == "__main__":
    config = Config()
    config.trace_filter = GlobbingFilter(
        include=['split_cycle', 'gen_feature_data', 'get_peak_valley', 'lfp_pv_join',

                 'gen_mdldata', 'features_calcu', 'fit_mdl', 'pred_tmp',
                 'plt_cmp_all', 'plt_dist', 'plt_heatmap_corr', 'plt_cap_corr', 'plt_feature_imp',
                 'plt_cyl_pred', 'plt_cyl_pred_asm',
                 'plt_trend1_u1', 'plt_trend1_u2', 'plt_trend1_u3',
                 'plt_trend2_dqdv_ux', 'plt_trend2_dqdv_yheight', 'plt_trend2_dqdv_areas',

                 ]
    )
    config.trace_filter = GlobbingFilter(exclude=['pycallgraph.*', ])
    graphviz = GraphvizOutput()
    graphviz.output_file = './graph.png'
    with PyCallGraph(output=graphviz, config=config):
        run_cycle()
