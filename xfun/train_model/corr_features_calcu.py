"""
-------------------------------------------------
   File Name：    $ {NAME}.py 
   Description :
   Author :       
   date：         2023/3/2 16:56
-------------------------------------------------
   Change Activity:
                     2023/3/2 16:56
-------------------------------------------------
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy.stats import entropy, norm, spearmanr
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from xfun.train_model.models_set import ModelsAsm
from utils.params_config import ConfPath, ConfVars, ConstVars
from log_conf.logger import logger


# ************************@@@@@***************************#
#   init parms
# ************************@@@@@***************************#

class ConfParams(ConfPath, ConfVars, ConstVars, ModelsAsm):

    def __init__(self):
        ConfPath.__init__(self)
        ConfVars.__init__(self)
        ConstVars.__init__(self)
        ModelsAsm.__init__(self)


# ************************@@@@@***************************#
# FEATURE ANALYS
# FEATURES corr_SPEARMAN
# FEATURES scaled
# FEATURES importances (linear-based tree_based)
# ************************@@@@@***************************#

class FeatureAnsys(ConfParams):
    """
    class FeatureAnsys:
                  corr_features
                  scaled_x_features
                  feature_importance

    Attributes:
         features_calcu
              corr_features
              scaled_x_features
              feature_importance
                    _linear_feature
                    _tree_feature_cusum
    """

    def __init__(self):
        ConfParams.__init__(self)

        self.df_normalized = None
        self.raw_features = None
        self.select_feature = None

    def corr_features(self, df_src, mdl_name):
        """
        1.df saved in list train :concat
                [
                [df1,df2,df3],
                [df1,df2,df3],
                [df1,df2,df3]
                ]
        2.chain flatten list,then concat
        3.test: preds one by one

        *******************
        spearman not required normalized
        1.calcu features x corr to y_cap
        2.drop features: p_value>0.5
        3.corr input: dataframe be scaled
        4.spearman matrix corr should be all columns(x+y columns)
        5.df_train(x_scaled ,y_src)
        corr: y_label  normalized
        preds: y_label src
        :param  :
        :return:
        """

        logger.info('spearman相关系数检验......')
        logger.info('all_columns:{}'.format(df_src.columns.tolist()))
        # df_src.drop(['delta_pv0'], axis=1, inplace=True)
        y_cap = df_src[self.y_label]
        df_x = df_src.drop([self.y_label], axis=1)
        x_columns = df_x.columns.tolist()
        del_col = []
        for cols in x_columns:
            spearman_, pvalue_ = spearmanr(df_x[cols], y_cap)
            if pvalue_ > 0.05:
                del_col.append(cols)
        logger.info('del_columns:{}'.format(del_col))
        raw_features = list(set(x_columns) - set(del_col))
        logger.info('kept raw_features:{}'.format(raw_features))
        # raw_features = x_columns
        df_corr_spman = df_src.loc[:, raw_features + [self.y_label]].corr(method='spearman')
        df_corr_spman.to_csv(os.path.join(self.pkl_path_mdl, 'df_corr_spman.csv'), sep=',', index=True)
        pickle.dump(raw_features, open(os.path.join(self.pkl_path_mdl, mdl_name + '_raw_features.pkl'), 'wb'))

    def scaled_x_features(self, df_src):

        y_cap = df_src[self.y_label]
        df_src = df_src.drop([self.y_label], axis=1)
        std = StandardScaler()
        mas = MaxAbsScaler()
        array_std = std.fit_transform(df_src)
        array_mas = mas.fit_transform(array_std)
        df_norm = pd.DataFrame(index=df_src.index,
                               columns=df_src.columns,
                               data=array_mas)

        # saved df_train(x_scaled ,y_src),calcu x-y corr
        df_norm.loc[:, self.y_label] = y_cap

        return std, mas, df_norm

    def _tree_feature_cusum(self, feature_names, f_importance):
        df_imp = pd.DataFrame()
        df_imp['feature_names'] = feature_names
        df_imp['f_importance'] = f_importance
        df_imp['impt_percent'] = f_importance / np.sum(f_importance)
        df_imp.sort_values(by=['impt_percent'], ascending=False, inplace=True)
        df_imp['cusum'] = np.cumsum(df_imp['f_importance']).abs()
        # df_imp = df_imp[df_imp['cusum'] <= 1.0]
        df_imp = df_imp[df_imp['f_importance'] > 0.001]
        select_feature = df_imp['feature_names'].tolist()
        select_importance = df_imp['f_importance'].tolist()
        dict_select_f = dict(sorted(zip(select_feature, np.round(select_importance, 3)),
                                    key=lambda x: x[1], reverse=True))
        tree_imp_path = os.path.join(self.pkl_path_mdl, 'df_importance.csv')
        df_imp.to_csv(tree_imp_path, sep=',')
        return select_feature, dict_select_f

    def _linear_feature(self, feature_names, f_importance):
        dict_select_f = dict(filter(lambda x: abs(x[1]) > 0, zip(feature_names, f_importance)))
        dict_select_f = dict(sorted(dict_select_f.items(), key=lambda x: abs(x[1]), reverse=True))
        select_feature = list(dict_select_f.keys())

        df_imp = pd.DataFrame.from_dict(dict_select_f, orient='index')
        df_imp = df_imp.reset_index()
        df_imp.columns = ['feature_names', 'f_importance']
        lr_imp_path = os.path.join(self.pkl_path_mdl, 'df_importance.csv')
        df_imp.to_csv(lr_imp_path, sep=',')
        return select_feature, dict_select_f

    def feature_importance(self, mdl_name):
        self.df_normalized = pickle.load(open(os.path.join(self.pkl_path_mdl, mdl_name + '_df_normalized.pkl'), 'rb'))
        logger.info('train data row*clolumns:{}'.format(self.df_normalized[self.raw_features].shape))
        mdl = self.models_asm(mdl_name)
        mdl.fit(self.df_normalized[self.raw_features], self.df_normalized[self.y_label])
        logger.info('feature importance mdl fit done')
        f_importance = mdl.feature_importances_.tolist() if mdl_name in ('xgb', 'gbdt', 'lgbm') \
            else mdl.coef_.tolist()

        dict_f = dict(sorted(zip(self.raw_features, f_importance),
                             key=lambda x: abs(x[1]), reverse=True))
        logger.info('{} importance raw features:{}\n'.format(mdl_name, dict_f))
        if mdl_name in ('xgb', 'gbdt', 'lgbm'):
            select_feature, dict_select_f = self._tree_feature_cusum(self.raw_features, f_importance)
            logger.info('{} _importance>0.001   raw features:{}\n'.format(mdl_name, dict_select_f))
        else:
            select_feature, dict_select_f = self._linear_feature(self.raw_features, f_importance)
            logger.info('lasso features:{}'.format(dict_select_f))

        self.select_feature = select_feature

    def features_calcu(self, RUN_MODE, mdl_name):
        """
        function excetute order:
                                  input                  output
        1. corr_features           _cap_join              _raw_features.pkl
        2.scaled_x_features          raw_features         mas、std、df_normalized.pkl
        3.feature_importance        df_normalized.pkl    _selected_features.pkl

        """
        with open(os.path.join(self.ts_avg_fatures_path, RUN_MODE + '_cap_join_mdldata.pkl'), 'rb') as cj:
            df_src = pickle.load(cj)
        self.corr_features(df_src, mdl_name)
        with open(os.path.join(self.pkl_path_mdl, mdl_name + '_raw_features.pkl'), 'rb') as rf:
            self.raw_features = pickle.load(rf)
        logger.info('train data shape:{}'.format(df_src.shape))
        std, mas, df_norm = self.scaled_x_features(df_src[self.raw_features + [self.y_label]])
        pickle.dump(std, open(os.path.join(self.pkl_path_mdl, mdl_name + '_std.pkl'), 'wb'))
        pickle.dump(mas, open(os.path.join(self.pkl_path_mdl, mdl_name + '_mas.pkl'), 'wb'))
        pickle.dump(df_norm, open(os.path.join(self.pkl_path_mdl, mdl_name + '_df_normalized.pkl'), 'wb'))
        self.feature_importance(mdl_name)
        dump_feature_path = os.path.join(self.pkl_path_mdl, mdl_name + '_selected_features.pkl')
        with open(dump_feature_path, 'wb') as sf:
            pickle.dump(self.select_feature, sf)


if __name__ == "__main__":
    fas = FeatureAnsys()
    for mdl_name in ['lgbm']:
        fas.features_calcu('TRAIN', mdl_name)
    pass
