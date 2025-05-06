from sklearn.linear_model import ElasticNet

from mode_BASE_Mod import BaseModel


class ElasticNetReg(BaseModel):
    def __init__(self):
        super().__init__("ElasticNet")
        self.params = {
            'alpha': 0.5,
            'l1_ratio': 0.5,
            'random_state': 42
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return ElasticNet(**self.params)
