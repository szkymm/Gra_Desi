import pandas as pd
from sklearn.preprocessing import RobustScaler


class DataLoader:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.scaler = RobustScaler()

    def load_datasets(self):
        """加载并标准化数据集"""
        train = pd.read_csv(f"{self.data_dir}train.csv")
        val = pd.read_csv(f"{self.data_dir}val.csv")
        test = pd.read_csv(f"{self.data_dir}test.csv")

        # 标准化
        X_train = self.scaler.fit_transform(train.drop(['SPAD', 'N'], axis=1))
        X_val = self.scaler.transform(val.drop(['SPAD', 'N'], axis=1))
        X_test = self.scaler.transform(test.drop(['SPAD', 'N'], axis=1))

        return (X_train, train['SPAD'],
                X_val, val['SPAD'],
                X_test, test['SPAD'])
