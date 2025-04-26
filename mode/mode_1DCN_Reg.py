import tensorflow as tf

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
        model = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        32, 3, activation='relu',
                        input_shape=self.params['input_shape']
                        ),
                    tf.keras.layers.MaxPooling1D(2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1)
                    ]
                )
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X, y):
        self.model = self.build_model()
        self.model.fit(
                X.reshape(X.shape[0], -1, 1), y,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                verbose=0
                )
        return self
