from sklearn.svm import SVR

from mode_BASE_Mod import BaseModel


class SVRPoly(BaseModel):
    def __init__(self):
        super().__init__("SVR-Poly")
        self.params = {
            'kernel': 'poly',
            'degree': 3,
            'C': 2
            }

    def build_model(self):
        return SVR(**self.params)
