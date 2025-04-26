from sklearn.linear_model import HuberRegressor

from mode_BASE_Mod import BaseModel


class HuberReg(BaseModel):
    def __init__(self):
        super().__init__("Huber")
        self.params = {
            'epsilon': 1.5
            }

    def build_model(self):
        return HuberRegressor(**self.params)
