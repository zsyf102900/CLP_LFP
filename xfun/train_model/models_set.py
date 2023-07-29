"""
-------------------------------------------------
   File Name：    $ {NAME}.py 
   Description :
   Author :       
   date：         2023/3/3/8:44
-------------------------------------------------
   Change Activity:
                     2023/3/3/8:44
-------------------------------------------------
"""

from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import lightgbm as lgb

class ModelsAsm:

    def _xgb_mdl(self):
        # xgb_parms = self.fit_params['xgb']
        # xgb_mdl = XGBRegressor(**xgb_parms)
        # return xgb_mdl
        pass

    def _svr_mdl(self):
        mdl = SVR(kernel='rbf')
        return mdl

    def _lasso_mdl(self):
        lasso_params = {'alpha': 0.05, 'normalize': True, 'max_iter': 3000}
        lasso_mdl = Lasso(**lasso_params)
        return lasso_mdl

    def _lgbm_mdl(self):
        params = {
            'learning_rate': 0.1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'max_depth': 6,
            'objective': 'regression'
        }
        mdl = lgb.LGBMRegressor(**params)
        return mdl

    def models_asm(self, mdl_name):
        mdl = {'xgb': self._xgb_mdl(),
               'lgbm': self._lgbm_mdl(),
               'lasso': self._lasso_mdl(),
               'svr': self._svr_mdl()}
        return mdl[mdl_name]
