import keras

from mode_BASE_Mod import BaseModel


class CNN1DReg(BaseModel):
    def __init__(self):
        super().__init__("1D-CNN")
        self.params = {
            'input_shape': (204, 1),
            'epochs': 50,
            'batch_size': 16
            }

    def build_model(self):
        """
        构建一个一维卷积神经网络 (1D-CNN) 模型。
        
        Returns:
            model: 构建的 Keras 模型实例。
        """
        try:
            # 日志记录：开始构建模型
            self.logger.info(f"开始构建模型：{self.model_name}")

            # 定义模型架构
            model = keras.Sequential(
                    [
                        keras.layers.Conv1D(
                                filters=self.params.get('filters', 32),
                                kernel_size=self.params.get('kernel_size', 3),
                                activation=self.params.get('activation', 'relu'),
                                input_shape=self.params['input_shape']
                                ),
                        keras.layers.MaxPooling1D(pool_size=self.params.get('pool_size', 2)),
                        keras.layers.Flatten(),
                        keras.layers.Dense(units=self.params.get('dense_units', 64), activation='relu'),
                        keras.layers.Dense(units=1)
                        ]
                    )

            # 编译模型
            model.compile(
                    optimizer=self.params.get('optimizer', 'adam'),
                    loss=self.params.get('loss', 'mse')
                    )

            # 日志记录：模型构建完成
            self.logger.info(f"模型构建完成：{self.model_name}")
            return model

        except Exception as e:
            self.logger.error(f"模型构建失败：{e}")
            raise RuntimeError(f"Failed to build model: {e}")

    def fit(self, X, y):
        self.model = self.build_model()
        self.model.fit(
                X.reshape(X.shape[0], -1, 1), y,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                verbose=0
                )
        return self
