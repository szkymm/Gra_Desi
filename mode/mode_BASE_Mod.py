import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

import joblib


class BaseModel(ABC):
    """模型抽象基类"""

    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self.scaler = None  # 初始化为 None
        self.logger = self._setup_logger()  # 初始化日志记录器

    def _setup_logger(self):
        """
        设置日志记录器，用于记录模型训练过程中的信息。
        
        Returns:
            logger: 配置好的日志记录器。
        """
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"logs/Training_Log_{timestamp}.log"

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)

        # 配置日志记录器
        logger = logging.getLogger(self.model_name)
        logger.setLevel(logging.INFO)

        # 创建文件处理器
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将处理器添加到记录器
        logger.addHandler(file_handler)

        return logger

    @abstractmethod
    def build_model(self):
        """
        子类必须实现此方法以构建模型。
        
        Returns:
            model: 构建的模型实例。
        """
        raise NotImplementedError("Subclasses must implement the build_model method.")

    def fit(self, x_train, y_train):
        self.model = self.build_model()
        if self.model is None:
            raise ValueError("build_model must return a valid model instance.")

        # 记录训练开始信息
        self.logger.info(f"开始训练模型：{self.model_name}")
        self.model.fit(x_train, y_train)

        # 记录训练完成信息
        self.logger.info(f"模型训练结束：{self.model_name}")
        return self

    def predict(self, vara_x):
        return self.model.predict(vara_x)

    def save(self, path):
        try:
            joblib.dump(self.model, path)
            self.logger.info(f"模型文件保存在：{path}")
        except Exception as e:
            self.logger.error(f"针对该路径：{path}的模型文件保存失败，\n原因是: {e}")
            raise RuntimeError(f"Failed to save model to {path}: {e}")

    @classmethod
    def load(cls, path):
        try:
            model = joblib.load(path)
            if not isinstance(model, cls):
                raise TypeError(f"Loaded object is not an instance of {cls.__name__}.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")
