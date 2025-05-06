from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, RBF

from mode_BASE_Mod import BaseModel


class GaussianProcessReg(BaseModel):
    def __init__(self):
        super().__init__("GPR")
        self.kernel: Kernel = ConstantKernel() * RBF()

    def build_model(self):
        if self.kernel is None:
            raise ValueError("Kernel for GaussianProcessRegressor is not defined.")

        # 记录核函数配置信息
        self.logger.info(f"Using kernel configuration: {self.kernel}")
        
        return GaussianProcessRegressor(
                kernel=self.kernel,
                random_state=42
                )
