from sklearn.svm import SVR

from mode_BASE_Mod import BaseModel


class SVRRBF(BaseModel):
    def __init__(self):
        super().__init__("SVR-RBF")
        self.params = {
            'kernel': 'rbf',
            'C': 5,
            'gamma': 'scale'
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return SVR(**self.params)
