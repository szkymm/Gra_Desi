from catboost import CatBoostRegressor

from mode_BASE_Mod import BaseModel


class CatBoostReg(BaseModel):
    def __init__(self, params=None):
        super().__init__("CatBoost")
        self.params = params or {
            'iterations': 100,
            'depth': 3,
            'verbose': 0
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return CatBoostRegressor(**self.params)