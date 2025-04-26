from catboost import CatBoostRegressor

from mode_BASE_Mod import BaseModel


class CatBoostReg(BaseModel):
    def __init__(self):
        super().__init__("CatBoost")
        self.params = {
            'iterations': 100,
            'depth': 3,
            'verbose': 0
            }

    def build_model(self):
        return CatBoostRegressor(**self.params)
