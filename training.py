"""
模型训练模块
功能：动态加载模型配置、执行批量训练、验证筛选、结果保存
版本：2.0
"""

import importlib
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from tqdm.auto import tqdm

from data_loader import HyperspectralDataLoader
from evaluator import HyperspectralEvaluator

# 模型配置清单
MODEL_CONFIGS = [
    # (变量名, 内部类名, 是否深度学习模型)
    ('mode_RIDG_Reg', 'RidgeRegression', False),
    ('mode_LASS_Reg', 'LassoRegression', False),
    ('mode_ENET_Reg', 'ElasticNetReg', False),
    ('mode_PLSR_Reg', 'PLSReg', False),
    ('mode_SVRR_RBF', 'SVRRBF', False),
    ('mode_SVRR_POL', 'SVRPoly', False),
    ('mode_BRID_Reg', 'BayesianRidgeReg', False),
    ('mode_HUBR_Reg', 'HuberReg', False),
    ('mode_RAFR_Reg', 'RandomForestReg', False),
    ('mode_XGBT_Reg', 'XGBoostReg', False),
    ('mode_LGBM_Reg', 'LightGBMReg', False),
    ('mode_CATB_Reg', 'CatBoostReg', False),
    ('mode_1DCN_Reg', 'CNN1DReg', True),
    ('mode_MLPR_Reg', 'MLPReg', False),
    ('mode_KNNR_Reg', 'KNNReg', False),
    ('mode_GPRR_Reg', 'GaussianProcessReg', False)
    ]


class ModelTrainer:
    def __init__(self, model_configs: List[Tuple] = MODEL_CONFIGS):
        """
        初始化训练器
        :param model_configs: 模型配置三元组列表 (模块名, 类名, 是否深度学习)
        """
        self.model_configs = model_configs
        self.data_loader = HyperspectralDataLoader()
        self.evaluator = HyperspectralEvaluator()
        self.output_dir = Path("trained_models")
        self.output_dir.mkdir(exist_ok=True)

    def dynamic_load_model(self, module_path: str, class_name: str):
        """动态加载模型类"""
        try:
            module = importlib.import_module(f"models.{module_path}")
            return getattr(module, class_name)()
        except Exception as e:
            print(f"加载模型 {module_path}.{class_name} 失败: {str(e)}")
            return None

    def _train_single_model(
            self,
            model,
            model_name: str,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            is_dl: bool
            ) -> Dict:
        """单个模型训练流程"""
        try:
            # 训练进度管理
            with tqdm(total=100, desc=f"训练 {model_name}", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
                if is_dl:
                    # 深度学习模型特殊处理
                    history = model.fit(
                            X_train.reshape(X_train.shape[0], -1, 1), y_train,
                            validation_data=(X_val.reshape(X_val.shape[0], -1, 1), y_val),
                            epochs=model.params['epochs'],
                            batch_size=model.params['batch_size'],
                            verbose=0,
                            callbacks=[self._get_dl_callback(pbar)]
                            )
                else:
                    # 传统机器学习模型
                    model.fit(X_train, y_train)
                    pbar.update(100)

                # 验证筛选
                val_pred = model.predict(X_val.reshape(X_val.shape[0], -1, 1) if is_dl else X_val)
                val_r2 = r2_score(y_val, val_pred)

                # 保存达标模型
                if val_r2 >= 0.75:
                    ext = "h5" if is_dl else "joblib"
                    save_path = self.output_dir / f"{model_name}.{ext}"
                    model.save(save_path) if is_dl else joblib.dump(model, save_path)
                    return {"status": "qualified", "path": save_path, "val_r2": val_r2}
                return {"status": "unqualified", "val_r2": val_r2}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_dl_callback(self, pbar: tqdm):
        """生成深度学习训练回调"""

        class DLCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                pbar.update(100 // self.params['epochs'])
                pbar.set_postfix(
                        {
                            'loss': f"{logs['loss']:.3f}",
                            'val_loss': f"{logs['val_loss']:.3f}"
                            }, refresh=False
                        )

        return DLCallback()

    def execute_pipeline(self):
        """执行完整训练流水线"""
        # 数据加载
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_datasets()

        # 批量训练
        results = []
        for module_path, class_name, is_dl in tqdm(self.model_configs, desc="全局进度"):
            model = self.dynamic_load_model(module_path, class_name)
            if not model:
                continue

            result = self._train_single_model(
                    model, class_name, X_train, y_train, X_val, y_val, is_dl
                    )
            results.append(
                    {
                        "model": class_name,
                        "status": result["status"],
                        "val_r2": result.get("val_r2", 0)
                        }
                    )

            # 达标模型自动评估
            if result["status"] == "qualified":
                test_pred = model.predict(X_test.reshape(X_test.shape[0], -1, 1) if is_dl else model.predict(X_test)
                report = self.evaluator.evaluate_single(test_pred, y_test)
                print(f"\n{'-' * 40}\n{class_name} 测试报告:\n{report}\n{'-' * 40}")

                # 生成总结报告
                summary_df = pd.DataFrame(results)
                summary_path = Path("reports/training_summary.csv")
                summary_df.to_csv(summary_path, index=False)
                print(f"\n✅ 训练总结报告已保存至 {summary_path}")

                if __name__ == "__main__":
                    trainer = ModelTrainer()
                trainer.execute_pipeline()
