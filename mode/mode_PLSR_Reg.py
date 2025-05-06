from sklearn.cross_decomposition import PLSRegression

from mode_BASE_Mod import BaseModel


class PLSReg(BaseModel):
    def __init__(self):
        super().__init__("PLS")
        self.params = {
            'n_components': 5
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return PLSRegression(**self.params)
