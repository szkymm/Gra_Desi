from sklearn.neural_network import MLPRegressor

from mode_BASE_Mod import BaseModel


class MLPReg(BaseModel):
    def __init__(self):
        super().__init__("MLP")
        self.params = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'early_stopping': True,
            'random_state': 42
            }

    def build_model(self):
        if not self.params:
            raise ValueError("Parameters for ElasticNet model are not defined.")
        return MLPRegressor(**self.params)
