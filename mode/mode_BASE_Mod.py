from abc import ABC, abstractmethod

import joblib


class BaseModel(ABC):
    """模型抽象基类"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.scaler = None

    @abstractmethod
    def build_model(self):
        """子类必须实现模型构建方法"""
        pass

    def fit(self, X_train, y_train):
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
