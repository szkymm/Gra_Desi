from sklearn.linear_model import Ridge
from .base_model import BaseModel

class RidgeRegression(BaseModel):
    def __init__(self):
        super().__init__("Ridge")
        self.params = {
            'alpha': 0.5,
            'random_state': 42
        }

    def build_model(self):
        return Ridge(**self.params)