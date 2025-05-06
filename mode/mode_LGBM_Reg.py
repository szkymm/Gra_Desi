from lightgbm import LGBMRegressor

from mode_BASE_Mod import BaseModel


class LightGBMReg(BaseModel):
    def __init__(self):
        super().__init__("LightGBM")
        self.params = {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'random_state': 42
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return LGBMRegressor(**self.params)
