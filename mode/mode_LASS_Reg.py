from sklearn.linear_model import Lasso

from mode_BASE_Mod import BaseModel


class LassoRegression(BaseModel):
    def __init__(self):
        super().__init__("Lasso")
        self.params = {
            'alpha': 0.01,
            'random_state': 42
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return Lasso(**self.params)
