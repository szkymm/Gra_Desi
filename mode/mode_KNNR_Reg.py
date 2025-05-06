from sklearn.neighbors import KNeighborsRegressor

from mode_BASE_Mod import BaseModel


class KNNReg(BaseModel):
    def __init__(self):
        super().__init__("KNN")
        self.params = {
            'n_neighbors': 7
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return KNeighborsRegressor(**self.params)
