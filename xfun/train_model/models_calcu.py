"""
-------------------------------------------------
   File Name：    models_calcu.py
   Description :
   Author :       
   date：         2023/3/2/17:00
-------------------------------------------------
   Change Activity:
                     2023/3/2/17:00
-------------------------------------------------
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xfun.train_model.models_set import ModelsAsm
from utils.params_config import ConfPath, ConfVars, ConstVars
from log_conf.logger import logger


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)
        ModelsAsm.__init__(self)


# ************************@@@@@***************************#
#   fit models
#   outlier detection if necessary
# ************************@@@@@***************************#

class ModelTrain(ConfParams, ModelsAsm):
    """
    class ModelTrain:
               outlier_detect: if necessary
               fit_mdl
    Attributes:
               outlier_detect: if necessary
               fit_mdl
    """

    def __init__(self, mdl_name):
        ConfParams.__init__(self)
        ModelsAsm.__init__(self)

        with open(os.path.join(self.pkl_path_mdl, mdl_name + '_selected_features.pkl'), 'rb') as sf:
            self.select_feature = pickle.load(sf)
        with open(os.path.join(self.pkl_path_mdl, mdl_name + '_df_normalized.pkl'), 'rb') as dfn:
            self.df_normalized = pickle.load(dfn)

    @staticmethod
    def mape(y1, y2):
        return 1 - np.average(np.abs(y1 - y2) / np.maximum(y1, y2))

    def outlier_detect(self, df_tr):
        logger.info('isolation detecting...')

        iso_forest = IsolationForest(n_estimators=500,
                                     max_samples=0.6,
                                     contamination=0.005,
                                     n_jobs=-1)
        iso_forest.fit(df_tr)
        ary_label = iso_forest.predict(df_tr)
        df_tr['iso_label'] = ary_label
        inner_index = df_tr['iso_label'] == 1
        logger.info('isolation detect outlier_samples_xy:{}'.format(df_tr.loc[~inner_index, :].shape))
        with open(os.path.join(self.pkl_path_mdl, '_iforest.pkl'), 'wb') as ift:
            pickle.dump(iso_forest, ift)
        df_tr.to_csv(self.pkl_path_mdl + 'df_tr_norm_selected.csv', sep=',', index=True)
        return df_tr

    def fit_mdl(self, mdl_name):
        from sklearn2pmml.pipeline import PMMLPipeline
        from sklearn2pmml import sklearn2pmml
        # self.outlier_detect(df_src[self.select_feature + [self.y_label]])
        mdl = self.models_asm(mdl_name)
        mdl.fit(self.df_normalized[self.select_feature], self.df_normalized[self.y_label])
        mdl_pkl_path = os.path.join(self.pkl_path_mdl, mdl_name + '_fit_mdl.pkl')
        pickle.dump(mdl, open(mdl_pkl_path, 'wb'))

        mdl_pmml_path = os.path.join(self.pkl_path_mdl, mdl_name + '_fit_mdl.pmml')

        pipeline = PMMLPipeline([("classifier", mdl)])
        pipeline.fit(self.df_normalized[self.select_feature], self.df_normalized[self.y_label])
        sklearn2pmml(pipeline, mdl_pmml_path, with_repr=True)
        logger.info('mdl save done'.format(mdl_name))


if __name__ == "__main__":

    for mdl_name in ['lgbm']:
        mt = ModelTrain(mdl_name)
        mt.fit_mdl(mdl_name)
    pass
