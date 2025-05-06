from sklearn.linear_model import Ridge

from mode_BASE_Mod import BaseModel


class RidgeRegression(BaseModel):
    def __init__(self):
        super().__init__("Ridge")
        self.params = {
            'alpha': 0.5,
            'random_state': 42
        }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return Ridge(**self.params)