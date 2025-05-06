from sklearn.linear_model import BayesianRidge

from mode_BASE_Mod import BaseModel


class BayesianRidgeReg(BaseModel):
    def __init__(self):
        super().__init__("BayesianRidge")
        self.params = {
            'alpha_1': 1e-6,
            'alpha_2': 1e-6,
            'lambda_1': 1e-6,
            'lambda_2': 1e-6
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return BayesianRidge(**self.params)
