#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""
import datetime as dt
import json as js
import logging as log
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.svm import SVR


class AutoDataModelTrainerCore:
    """

    """

    def __init__(self):
        """

        """
        self.base_path = Path(sys.argv[0]).resolve().parent.parent  # 获取项目基础路径
        self.ckpt_path = Path(self.base_path, "ckpt")  # 定义模型检查点存储路径
        self.data_path = Path(self.base_path, "data")  # 定义数据文件存储路径
        self.logs_path = Path(self.base_path, "logs")  # 定义日志文件存储路径
        self.mode_path = Path(self.base_path, "mode")  # 定义模块文件存储路径
        self.rezu_path = Path(self.base_path, "results")  # 定义结果文件存储路径
        self.sets_path = Path(self.base_path, "sets")  # 定义设定文件存储路径
        self._init_directories()  # 初始化所需的目录结构
        self._init_logger_manager()  # 初始化日志管理器并配置日志记录器
        self.regi_mode = self._init_model_registry()  # 初始化模型注册表并载入模型配置
        self.retr_maxm = 3

    def _init_directories(self):
        """
        初始化所需的目录结构。
        该方法定义了一组必要的文件路径，并检查这些路径是否存在。如果路径不存在，
        则创建对应的目录并记录相关信息到日志和控制台输出中。
        Initialize the required directory structure. This method defines a set of necessary file paths,
        checks whether these paths exist, creates corresponding directories if they are missing, and logs
        relevant information to both the console output and log files.
        :param: None
        :return: None
        :raises: None
        """
        # 定义需求的文件路径，使用字典存储路径和描述信息
        dict_reqs_dirs = {
            self.ckpt_path: "模型检查点存储路径",
            self.data_path: "数据文件存储路径",
            self.logs_path: "日志文件存储路径",
            self.mode_path: "模块文件存储路径",
            self.rezu_path: "结果文件存储路径",
            self.sets_path: "设定文件存储路径"
            }
        # 遍历字典中的每个路径及其描述信息
        for full_path, desc_info in dict_reqs_dirs.items():
            if not full_path.exists():  # 检查路径是否存在
                full_path.mkdir()  # 如果路径不存在，则创建该路径
                # 使用日志记录器记录路径创建信息
                self.root_logg.info(f"{desc_info}：{full_path}，文件路径已创建。")
                # 打印路径创建信息到控制台
                print(f"{desc_info}：{full_path}，文件路径已创建。")

    def _init_logger_manager(self):
        """
        为类实例初始化日志管理器。
        该方法执行以下操作：配置名为'ADModelTrainerCore'的日志记录器；设置日志级别为INFO；确保日志目录存在；
        创建带时间戳的日志文件；定义日志消息格式；并在文件处理器不存在时向日志记录器添加FileHandler。
        Initializes the logger manager for the class instance.
        This method configures a logger with the name "ADModelTrainerCore", sets its logging level to INFO,
        ensures the existence of the log directory, and creates a timestamped log file. It also defines
        the log message format and adds a FileHandler to the logger if one does not already exist.
        :param: None
        :return: None
        :raises: None
        """
        self.root_logg = log.getLogger("ADModelTrainerCore")  # 设置名为"ADModelTrainerCore"的logger实例
        self.root_logg.setLevel(log.INFO)  # 设置日志记录级别为INFO
        self.logs_path.mkdir(exist_ok=True)  # 确保日志目录存在，如果不存在则创建
        stri_time = dt.datetime.now().strftime("%Y_%m%d_%H%M_00%S")  # 格式化当前时间为指定格式字符串
        logs_name = f"Tran_Logs_{stri_time}.log"  # 定义日志文件名，包含时间戳
        logs_fmts = log.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")  # 定义日志消息的格式
        path_logs = Path(self.logs_path, logs_name)  # 构造日志文件的完整路径
        objt_logs_hand = log.FileHandler(path_logs, encoding="utf-8")  # 创建一个对象用于写入日志文件
        objt_logs_hand.setFormatter(logs_fmts)  # 为日志对象设置日志记录格式
        # 检查当前logger实例中是否已存在FileHandler类型的处理器，避免重复添加
        if not any(isinstance(h, log.FileHandler) for h in self.root_logg.handlers):
            self.root_logg.addHandler(objt_logs_hand)  # 如果不存在，则将定义好的FileHandler添加到logger实例中
        self.root_logg.info("✅ 日志管理器初始化成功完成，已进入就绪状态。")  # 记录一条INFO级别的日志

    def _init_model_registry(self) -> Dict[str, dict]:
        """
        初始化模型注册表，加载注册表文件并构建模型的初始化函数与参数配置的映射关系。
        此方法首先检查注册表文件是否存在，如果不存在则记录警告日志并抛出异常。
        如果文件存在，则解析其内容并遍历每个模型配置，验证初始化函数是否存在，
        并将模型名称与其对应的初始化函数和参数配置存入字典中。最后返回该字典。
        抛出错误：FileNotFoundError: 如果注册表文件未找到，则抛出此异常
        抛出错误：Exception: 如果某个模型的初始化函数未找到，则抛出此异常
        返回：包含模型名称与对应初始化函数及参数配置的字典，
                 键为模型名称，值为元组 (初始化函数, 参数配置)
        Initialize the model registry, load the registry file, and build a mapping between the model's initialization 
        function and parameter configuration.
        This method first checks if the registry file exists. If it does not exist, it logs a warning and throws an exception.
        If the file exists, it parses its content and iterates through each model configuration to verify whether the 
        initialization function exists.
        It then stores the model name along with its corresponding initialization function and parameter configuration into a 
        dictionary. Finally, it returns this dictionary.
                :param: None
        :raises FileNotFoundError: Thrown if the registry file is not found
        :raises Exception: Thrown if an initialization function for a certain model is not found
        :return: dict[str, tuple[callable, dict]]
                 A dictionary containing the model name as the key and a tuple of (initialization function, 
                 parameter configuration) as the value.
        :rtype: dict[str, tuple[callable, dict]]
        """
        name_regi = "sets_mode_regi.json"  # 定义注册表文件的名称
        path_full = Path(self.sets_path, name_regi)  # 构造注册表文件的完整路径
        # 检查注册表文件是否存在
        if not path_full.exists():
            self.root_logg.error(f"❗ 注册表文件：{name_regi}未找到。")  # 如果文件不存在，记录警告日志
            print(f"❗ 注册表文件：{name_regi}未找到。")  # 在控制台打印错误信息
            raise FileNotFoundError(f"❗ 模型注册表：{name_regi}文件未找到。")  # 抛出错误代码
        # 打开注册表文件并加载其内容
        with open(path_full, "r", encoding="utf-8") as regi_file:
            regi_json = js.load(regi_file)  # 使用json模块加载文件内容
        dict_mode_regi = {}  # 初始化字典用于存储模型注册信息
        # 遍历注册表中的每个模型配置
        for mode_name, conf_mode in regi_json.items():
            func_name = conf_mode.get("init_func")  # 获取模型初始化函数的名称
            conf_para = conf_mode.get("para_conf", {})  # 获取模型参数配置（如果存在）
            # 检查当前对象是否具有该初始化函数
            if hasattr(self, func_name):
                init_func = getattr(self, func_name)  # 如果存在，获取该函数对象
                dict_mode_regi[mode_name] = (init_func, conf_para)  # 将模型名称与对应的初始化函数和参数配置存入字典
                self.root_logg.info("✅ 模型注册表初始化完成。")  # 记录模型注册成功的日志
            else:
                # 如果初始化函数不存在，记录错误日志
                self.root_logg.error(f"❌ 未找到初始化函数：{func_name},模型名称：{mode_name}。")
                print(f"❌ 未找到初始化函数：{func_name},模型名称：{mode_name}。")  # 在控制台打印错误信息
                raise Exception(f"❌ 未找到初始化函数：{func_name},模型名称：{mode_name}。")  # 抛出错误代码
        return dict_mode_regi

    def _init_linear_model(self, **para_mode) -> BaseEstimator:
        """"""
        mode_name = para_mode.pop("mode_name")  # 从参数字典中移除并获取键为"mode_name"的值，该值代表模型名称。
        dict_mode = {  # 定义一个字典
            "Ridge": Ridge,  # 将字符串"Ridge"映射到Ridge类。
            "Lasso": Lasso,  # 将字符串"Lasso"映射到Lasso类。
            "ElasticNet": ElasticNet,  # 将字符串"ElasticNet"映射到ElasticNet类。
            "PLSReg": PLSRegression  # 将字符串"PLSReg"映射到PLSRegression类。
            }
        if mode_name not in dict_mode:  # 检查提供的模型名称是否不在支持的模型字典中。
            self.root_logg.error(f"❌ 不支持的模型：{mode_name}。")  # 如果模型不被支持，记录错误日志信息。
            raise ValueError(f"❌ 不支持的模型：{mode_name}。")  # 抛出一个ValueError异常，提示模型不被支持。
        mode_objt = dict_mode[mode_name](**para_mode)  # 使用提供的参数实例化对应的模型对象。
        self.root_logg.info(f"✅ 模型：{mode_name}，已初始化完成。")
        print(f"✅ 模型：{mode_name}成功载入，准备开始训练。")
        return mode_objt

    def _init_svm_model(self, **para_mode) -> BaseEstimator:
        """"""
        mode_name = para_mode.pop("mode_name")
        vali_kern = ["rbf", "poly"]
        kern_type = para_mode.get("kernel", "")
        if kern_type not in vali_kern:
            self.root_logg.error(f"❌ 不支持的SVR核类型：{kern_type}。")
            raise ValueError(f"❌ SVR模型 {mode_name} 配置错误：无效核类型")
        else:
            self.root_logg.info(f"✅ 模型：{mode_name}，已初始化完成。")
            print(f"✅ 模型：{mode_name}成功载入，准备开始训练。")
        return SVR(**para_mode)

    def _train_single_model(self, mode_name: str) -> Dict[str, Any]:
        """仍需打磨，很多内容还不达标，并且还有深度学习模型的问题，但是先打磨单一模型吧，这个最后来完成。"""
        self.root_logg.info(f"▶ 开始训练模型：{mode_name}")
        if mode_name not in self.regi_mode:
            self.root_logg.error(f"❌ 未注册的模型：{mode_name}")
            raise KeyError(f"❌ 未注册的模型：{mode_name}")
        func_init, para_conf = self.regi_mode[mode_name]
        dict_resu = {
            "stri_stat": "未达标",
            "dete_coef": -np.inf,
            "vari_atte": 0,
            "full_path": None
            }
        para_tran = {'mode_name': mode_name, **para_conf}
        objt_mode = func_init(**para_tran)
        for vari_atte in range(1, self.retr_maxm + 1):
            try:
                self.root_logg.info(f"🔄 尝试第 {vari_atte}/{self.retr_maxm} 次训练")
                print(f"正在尝试 {mode_name} 第 {vari_atte} 次训练...")
                objt_mode.fit("self.x_train", "self.y_train")
                y_preds = objt_mode.predict("self.y_test")
                current_r2 = r2_score("self.y_train", y_preds)
                if current_r2 > dict_resu['dete_coef']:
                    dict_resu.update(
                            {
                                'dete_coef': current_r2,
                                'vari_atte': vari_atte
                                }
                            )
                    self.root_logg.info(f"📈 更新最佳R²值：{current_r2:.6f}")
                else:
                    objt_mode = func_init(**para_tran)
                    continue
                if current_r2 >= 0.75:  # 你的R²阈值
                    dict_resu['stts_train'] = '达标'
                    self.root_logg.info(f"✅ 第 {vari_atte} 次尝试达标")
                    break
            except Exception as e:
                error_msg = f"❌ 第 {vari_atte} 次训练失败：{str(e)}"
                self.root_logg.error(error_msg)
                print(error_msg)
                continue
        if dict_resu['best_r2'] > -1:
            stri_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{mode_name}_{stri_time}.pkl"
            path_ckpt = Path(self.ckpt_path, file_name)

            try:
                dump(objt_mode, self.ckpt_path / f"{mode_name}.joblib")
                dict_resu['path_ckpt'] = str(path_ckpt)
                self.root_logg.info(f"💾 检查点已保存至：{path_ckpt}")
            except Exception as e:
                error_msg = f"❌ 模型保存失败：{str(e)}"
                self.root_logg.error(error_msg)
                print(error_msg)

        # 记录最终结果
        self.root_logg.info(
                f"▷ 训练完成：{mode_name} | 状态：{dict_resu['stts_train']} | "
                f"最佳R²：{dict_resu['best_r2']:.4f} | 尝试次数：{dict_resu['n_retry']}"
                )
        return dict_resu
