from sklearn.linear_model import BayesianRidge

from mode_BASE_Mod import BaseModel


class BayesianRidgeReg(BaseModel):
    def __init__(self):
        super().__init__("BayesianRidge")

    def build_model(self):
        return BayesianRidge()
