#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime as dt
import logging as log
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


class PearsonCorrelationAnalysis:
    def __init__(self):
        # 初始化基础路径（获取项目根目录的父级目录）
        self.base_path = Path(sys.argv[0]).resolve().parent.parent
        # 初始化日志管理系统
        self.root_logg = BandCorrelationAnalysis().root_logg
        self.band_data = BandCorrelationAnalysis().band_data

    @staticmethod
    def analysis_calculation(data_comx, data_comy):
        pearson_corr, pearson_p = pearsonr(data_comy, data_comx)
        return pearson_corr, pearson_p

    def run(self):
        data_comy = self.band_data["SPAD"]
        rows_name = self.band_data.keys()
        psca_rezu = {}
        for name_rows in rows_name:
            data_comx = self.band_data[name_rows]
            psca_corr, psca_varp = self.analysis_calculation(data_comx, data_comy)
            psca_rezu[name_rows] = {
                "psca_corr": psca_corr,
                "psca_varp": psca_varp
                }
        return psca_rezu


class SpearmanRankCorrelationAnalysis:
    def __init__(self):
        self.base_path = Path(sys.argv[0]).resolve().parent.parent
        self.root_logg = BandCorrelationAnalysis().root_logg
        self.band_data = BandCorrelationAnalysis().band_data

    @staticmethod
    def analysis_calculation(data_comx, data_comy):
        pearson_corr, pearson_p = spearmanr(data_comx, data_comy)
        return pearson_corr, pearson_p

    def run(self):
        data_comy = self.band_data["SPAD"]
        srca_rezu = {}
        rows_name = self.band_data.keys()
        for name_rows in rows_name:
            data_comx = self.band_data[name_rows]
            srca_corr, srca_varp = self.analysis_calculation(data_comx, data_comy)
            srca_rezu[name_rows] = {
                "srca_corr": srca_corr,
                "srca_varp": srca_varp
                }
        return srca_rezu


class BandCorrelationAnalysis:
    def __init__(self):
        self.base_path = Path(sys.argv[0]).resolve().parent.parent
        # 日志文件存储路径（存放系统运行日志）
        self.logs_path = Path(self.base_path, "logs")
        self.root_logg = self._init_logger_manager()
        self.file_name = "rezu_vege_indi.csv"
        self.data_path = Path(self.base_path, "results")
        self.data_file = Path(self.data_path, self.file_name)
        self.band_data = self._init_reflectance_csv()

    def _init_logger_manager(self):
        """
        为类实例初始化日志管理器。
        该方法执行以下操作：配置名为 "ADModelTrainerCore" 的日志记录器；设置日志级别为INFO；确保日志目录存在；
        创建带时间戳的日志文件；定义日志消息格式；并在文件处理器不存在时向日志记录器添加FileHandler。
        Initializes the logger manager for the class instance.
        This method configures a logger with the name "ADModelTrainerCore", sets its logging level to INFO,
        ensures the existence of the log directory, and creates a timestamped log file. It also defines
        the log message format and adds a FileHandler to the logger if one does not already exist.
        :param: None
        :return: None
        :raises: None
        """
        # 创建日志记录器实例
        self.root_logg = log.getLogger("ADModelTrainerCore")
        # 设置日志级别为INFO
        self.root_logg.setLevel(log.INFO)
        # 确保日志目录存在
        self.logs_path.mkdir(exist_ok=True)
        # 格式化当前时间为指定格式字符串
        stri_time = dt.datetime.now().strftime("%Y_%m%d_%H%M_00%S")
        # 定义日志文件名
        logs_name = f"Tran_Logs_{stri_time}.log"
        # 定义日志消息的格式
        logs_fmts = log.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")
        # 构造日志文件的完整路径
        path_logs = Path(self.logs_path, logs_name)
        # 创建一个对象用于写入日志文件
        objt_logs_hand = log.FileHandler(path_logs, encoding="utf-8")
        # 为日志对象设置日志记录格式
        objt_logs_hand.setFormatter(logs_fmts)
        # 避免重复添加处理器（检查现有处理器类型）
        if not any(isinstance(h, log.FileHandler) for h in self.root_logg.handlers):
            # 如果不存在，则添加文件处理器
            self.root_logg.addHandler(objt_logs_hand)
        # 记录初始化完成日志
        self.root_logg.info("✅ 日志系统初始化完成。")
        return self.root_logg

    def _init_reflectance_csv(self):
        band_data = pd.read_csv(self.data_file, encoding="utf-8")
        return band_data

    @staticmethod
    def sort_by_significance(data_dict, data_keys):
        # 将字典项转换为列表，并按p值升序、相关系数绝对值降序排序
        sorted_bands = sorted(
                data_dict.items(),
                key=lambda item: (item[1][f"{data_keys}_varp"], -abs(item[1][f"{data_keys}_corr"]))
                )

        # 生成带排名的结果列表
        rank_rezu = []
        for rank_numb, (band_name, band_andt) in enumerate(sorted_bands, start=1):
            rank_rezu.append(
                    {
                        "rank_numb": rank_numb,
                        "band_name": band_name,
                        f"{data_keys}_corr": float(1-((1- band_andt[f"{data_keys}_corr"])/2)),
                        f"{data_keys}_varp": float(band_andt[f"{data_keys}_varp"])
                        }
                    )
        return rank_rezu

    def run(self):
        clas_srca = SpearmanRankCorrelationAnalysis()
        clas_psca = PearsonCorrelationAnalysis()
        rezu_srca = clas_srca.run()
        rezu_psca = clas_psca.run()
        srca_rank = self.sort_by_significance(rezu_srca, "srca")
        print("◆"*10 + "斯皮尔曼秩相关系数"+"◆"*10+"\n")
        for rank_numb in srca_rank[:12]:

            print(
                f"Rank {rank_numb["rank_numb"]}| {rank_numb["band_name"]} | "
                f"r = {rank_numb["srca_corr"]:.4f}| p = {rank_numb["srca_varp"]:.4f}"
                )
        print("◆"*10 + "皮尔森相关系数"+"◆"*10+"\n")
        psca_rank = self.sort_by_significance(rezu_psca, "psca")
        for numb_rank in psca_rank[:12]:
            print(
                f"Rank {numb_rank["rank_numb"]}| {numb_rank["band_name"]} | "
                f"r = {numb_rank["psca_corr"]:.4f}| p = {numb_rank["psca_varp"]:.4f}"
                )


if __name__ == "__main__":
    BandCorrelationAnalysis().run()
