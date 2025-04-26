from sklearn.ensemble import RandomForestRegressor

from mode_BASE_Mod import BaseModel


class RandomForestReg(BaseModel):
    def __init__(self):
        super().__init__("RandomForest")
        self.params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42
            }

    def build_model(self):
        return RandomForestRegressor(**self.params)
