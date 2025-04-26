from xgboost import XGBRegressor

from mode_BASE_Mod import BaseModel


class XGBoostReg(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")
        self.params = {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
            }

    def build_model(self):
        return XGBRegressor(**self.params)
