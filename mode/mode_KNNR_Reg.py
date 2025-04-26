from sklearn.neighbors import KNeighborsRegressor

from mode_BASE_Mod import BaseModel


class KNNReg(BaseModel):
    def __init__(self):
        super().__init__("KNN")
        self.params = {
            'n_neighbors': 7
            }

    def build_model(self):
        return KNeighborsRegressor(**self.params)
