"""
增强型模型评估模块
功能：多目标评估、动态指标管理、交互式可视化报告生成
版本：2.1
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm


class EnhancedEvaluator:
    def __init__(
            self,
            target_names: List[str] = ["SPAD", "N"],
            metrics_config: Optional[Dict[str, Dict]] = None,
            output_dir: str = "reports",
            progress_theme: str = "classic"
            ):
        """
        初始化增强型评估器

        :param target_names: 目标变量名称列表 (支持多目标)
        :param metrics_config: 指标配置字典 {指标名: {func: 计算函数, kwargs: 参数字典}}
        :param output_dir: 报告输出目录
        :param progress_theme: 进度条主题 ('classic'/'modern')
        """
        self.target_names = target_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_theme = progress_theme

        # 初始化默认指标配置
        self.metrics = self._init_metrics(metrics_config)

        # 配置可视化风格
        sns.set(style="whitegrid", palette="muted")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
        plt.rcParams['axes.unicode_minus'] = False

    def _init_metrics(self, config: Optional[Dict]) -> Dict:
        """初始化评估指标"""
        default_metrics = {
            'R2': {
                'func': r2_score,
                'kwargs': {},
                'format': '.3f'
                },
            'RMSE': {
                'func': lambda y, p: np.sqrt(mean_squared_error(y, p)),
                'kwargs': {},
                'format': '.2f'
                },
            'MAE': {
                'func': mean_absolute_error,
                'kwargs': {},
                'format': '.2f'
                },
            'Spearman': {
                'func': lambda y, p: spearmanr(y, p).correlation,
                'kwargs': {},
                'format': '.3f'
                },
            'Tolerance10%': {
                'func': lambda y, p: np.mean(np.abs((y - p) / y) <= 0.1) * 100,
                'kwargs': {},
                'format': '.1f%'
                }
            }
        return config or default_metrics

    def evaluate(
            self,
            model_paths: List[Union[str, Path]],
            X_test: np.ndarray,
            y_test: np.ndarray,
            model_types: List[str],
            save_details: bool = True
            ) -> pd.DataFrame:
        """
        执行多模型多目标评估

        :param model_paths: 模型文件路径列表
        :param X_test: 测试集特征矩阵
        :param y_test: 测试集目标值 (n_samples, n_targets)
        :param model_types: 模型类型列表 ('sklearn'/'keras')
        :param save_details: 是否保存详细预测结果
        :return: 评估结果DataFrame
        """
        results = []
        pbar_config = self._get_progress_config()

        with tqdm(total=len(model_paths), **pbar_config) as main_pbar:
            for path, mtype in zip(model_paths, model_types):
                main_pbar.set_description(f"评估 {Path(path).stem}")

                # 加载模型
                model = self._load_model(path, mtype)
                if model is None:
                    continue

                # 执行预测
                y_pred = self._predict_with_progress(model, X_test, mtype)

                # 计算多目标指标
                model_metrics = {}
                for target_idx, target in enumerate(self.target_names):
                    y_true_single = y_test[:, target_idx]
                    y_pred_single = y_pred[:, target_idx]

                    for metric, config in self.metrics.items():
                        key = f"{target}_{metric}"
                        try:
                            value = config['func'](y_true_single, y_pred_single, **config['kwargs'])
                            model_metrics[key] = float(value)
                        except Exception as e:
                            print(f"计算 {metric} 失败: {str(e)}")
                            model_metrics[key] = np.nan

                # 记录结果
                model_metrics["Model"] = Path(path).stem
                results.append(model_metrics)

                # 保存详细信息
                if save_details:
                    self._save_prediction_details(
                            y_test, y_pred, path, target_names=self.target_names
                            )

                main_pbar.update(1)

        # 生成报告
        report_df = pd.DataFrame(results)
        self._save_reports(report_df, y_test)
        return report_df

    def _load_model(self, path: Union[str, Path], model_type: str):
        """安全加载模型"""
        try:
            if model_type == "keras":
                from keras.models import load_model
                return load_model(path)
            return load(path)
        except Exception as e:
            print(f"加载模型失败 {Path(path).name}: {str(e)}")
            return None

    def _predict_with_progress(self, model, X: np.ndarray, model_type: str) -> np.ndarray:
        """带内存管理的预测流程"""
        # 调整输入维度
        if model_type == "keras":
            X = X.reshape(X.shape[0], -1, 1)

        # 分块预测配置
        chunk_size = min(512, len(X))
        preds = []
        pbar_config = self._get_progress_config(desc="预测进度")

        with tqdm(total=len(X), **pbar_config) as pred_pbar:
            for i in range(0, len(X), chunk_size):
                chunk = X[i:i + chunk_size]

                try:
                    if model_type == "keras":
                        pred = model.predict(chunk, verbose=0)
                    else:
                        pred = model.predict(chunk)
                    preds.append(pred)
                except Exception as e:
                    print(f"预测失败: {str(e)}")
                    preds.append(np.full((len(chunk), len(self.target_names)), np.nan))

                pred_pbar.update(len(chunk))

        return np.vstack(preds)

    def _save_prediction_details(
            self, y_true: np.ndarray, y_pred: np.ndarray,
            model_path: Path, target_names: List[str]
            ):
        """保存详细预测结果"""
        details = {}
        for idx, name in enumerate(target_names):
            details.update(
                    {
                        f"True_{name}": y_true[:, idx],
                        f"Pred_{name}": y_pred[:, idx],
                        f"AE_{name}": np.abs(y_true[:, idx] - y_pred[:, idx]),
                        f"RE_{name}": np.abs((y_true[:, idx] - y_pred[:, idx]) / (y_true[:, idx] + 1e-6))
                        }
                    )

        df = pd.DataFrame(details)
        save_path = self.output_dir / f"details_{model_path.stem}.parquet"
        df.to_parquet(save_path)  # 使用Parquet格式节省空间

    def _save_reports(self, df: pd.DataFrame, y_test: np.ndarray):
        """生成并保存所有报告"""
        # CSV报告
        csv_path = self.output_dir / "performance_report.csv"
        df.to_csv(csv_path, index=False)

        # JSON摘要
        summary = {
            "best_models": {},
            "avg_metrics": {}
            }
        for target in self.target_names:
            target_metrics = [m for m in df.columns if m.startswith(f"{target}_")]
            best_model = df.loc[df[f"{target}_R2"].idxmax()]
            summary["best_models"][target] = {
                "name": best_model["Model"],
                "R2": best_model[f"{target}_R2"],
                "RMSE": best_model[f"{target}_RMSE"]
                }

        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # 可视化报告
        self._generate_visualizations(df, y_test)

    def _generate_visualizations(self, df: pd.DataFrame, y_test: np.ndarray):
        """生成交互式可视化报告"""
        plt.figure(figsize=(18, 10))

        # 模型性能热力图
        plt.subplot(2, 2, 1)
        metrics_to_plot = [m for m in self.metrics if m != 'Tolerance10%']
        plot_data = df[[f"{t}_{m}" for t in self.target_names for m in metrics_to_plot]]
        sns.heatmap(plot_data.corr(), annot=True, cmap="coolwarm")
        plt.title("指标相关性热力图")

        # 多目标R2对比
        plt.subplot(2, 2, 2)
        for target in self.target_names:
            sns.kdeplot(df[f"{target}_R2"], label=target, fill=True)
        plt.xlabel("R² Score")
        plt.legend()
        plt.title("各目标R²分布")

        # 误差分布雷达图
        plt.subplot(2, 2, 3, polar=True)
        metrics = ['RMSE', 'MAE', 'Spearman', 'Tolerance10%']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        for idx, model in enumerate(df["Model"]):
            values = []
            for metric in metrics:
                values.append(np.mean([df.loc[idx, f"{t}_{metric}"] for t in self.target_names]))
            values += values[:1]
            plt.plot(angles + angles[:1], values, label=model, alpha=0.5)
        plt.xticks(angles, metrics)
        plt.title("多目标误差雷达图")

        # 真实值-预测值分布
        plt.subplot(2, 2, 4)
        for target in self.target_names:
            true_values = y_test[:, self.target_names.index(target)]
            plt.scatter(true_values, true_values, alpha=0.3, label=f"{target} 基准线")
            for model in df["Model"]:
                preds = pd.read_parquet(self.output_dir / f"details_{model}.parquet")[f"Pred_{target}"]
                plt.scatter(true_values, preds, alpha=0.2, label=model)
        plt.xlabel("真实值")
        plt.ylabel("预测值")
        plt.title("预测值分布对比")

        plt.tight_layout()
        plt.savefig(self.output_dir / "visual_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _get_progress_config(self, desc: str = "评估进度") -> Dict:
        """获取进度条样式配置"""
        themes = {
            'classic': {
                'desc': desc,
                'bar_format': "{l_bar}{bar}| {n_fmt}/{total_fmt}",
                'colour': 'green'
                },
            'modern': {
                'desc': f"📊 {desc}",
                'bar_format': "{desc}: {percentage:.0f}%|{bar:20}| {n_fmt}/{total_fmt}",
                'colour': '#FF6F00'
                }
            }
        return themes.get(self.progress_theme, themes['classic'])

    @staticmethod
    def create_custom_metric(
            name: str,
            calculation_func: callable,
            format_str: str = ".2f",
            **kwargs
            ) -> Dict:
        """
        创建自定义评估指标

        :param name: 指标名称
        :param calculation_func: 计算函数 (y_true, y_pred) -> float
        :param format_str: 结果格式化字符串
        :param kwargs: 传递给计算函数的固定参数
        :return: 指标配置字典
        """
        return {
            name: {
                'func': calculation_func,
                'kwargs': kwargs,
                'format': format_str
                }
            }
