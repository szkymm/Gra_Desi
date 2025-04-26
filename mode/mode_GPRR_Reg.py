from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from mode_BASE_Mod import BaseModel


class GaussianProcessReg(BaseModel):
    def __init__(self):
        super().__init__("GPR")
        self.kernel = ConstantKernel() * RBF()

    def build_model(self):
        return GaussianProcessRegressor(
                kernel=self.kernel,
                random_state=42
                )
