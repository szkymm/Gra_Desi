#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import csv
import datetime as dt
import json as js
import logging as log
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from catboost import CatBoostRegressor
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


class AutoDataModelTrainerCore:
    """

    """

    def __init__(self):
        """

        """
        # 初始化基础路径（获取项目根目录的父级目录）
        self.base_path = Path(sys.argv[0]).resolve().parent.parent
        # 检查点存储路径（保存训练好的模型）
        self.ckpt_path = Path(self.base_path, "ckpt")
        # 原始数据存储路径（存放输入数据文件）
        self.data_path = Path(self.base_path, "data")
        # 日志文件存储路径（存放系统运行日志）
        self.logs_path = Path(self.base_path, "logs")
        # 模块代码存储路径（存放功能模块文件）
        self.mode_path = Path(self.base_path, "mode")
        # 结果文件存储路径（存放计算结果数据）
        self.rezu_path = Path(self.base_path, "results")
        # 配置文件存储路径（存放参数配置文件）
        self.sets_path = Path(self.base_path, "sets")
        # 初始化日志管理系统
        self.root_logg = self._init_logger_manager()
        # 创建必要目录结构
        self._init_directories()
        # 模型注册表配置文件名
        self.name_regi = "sets_mode_regi.json"
        # 加载模型注册表配置
        self.regi_mode = self._init_model_registry()
        # 单模型最大训练尝试次数
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
        # 目录配置字典（路径对象: 目录描述）
        dict_reqs_dirs = {
            self.ckpt_path: "模型检查点目录",
            self.data_path: "原始数据目录",
            self.logs_path: "系统日志目录",
            self.mode_path: "功能模块目录",
            self.rezu_path: "计算结果目录",
            self.sets_path: "配置设置目录"
            }
        # 遍历字典中的每个路径及其描述信息
        for full_path, desc_info in dict_reqs_dirs.items():
            # 检查路径是否存在
            if not full_path.exists():
                # 如果路径不存在，则创建该路径
                full_path.mkdir()
                # 记录日志
                self.root_logg.info(f"✅ {desc_info}：{full_path}，文件路径已创建。")
                # 控制台输出
                print(f"✅ {desc_info}：{full_path}，文件路径已创建。")

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

    def _init_model_registry(self) -> Dict[str, dict]:
        """
        初始化模型注册表，加载注册表文件并构建模型的初始化函数与参数配置的映射关系。
        此方法首先检查注册表文件是否存在，如果不存在则记录警告日志并抛出异常。如果文件存在，则解析其内容并遍历每个模型配置，验证初始化函数是否存在，
        并将模型名称与其对应的初始化函数和参数配置存入字典中。最后返回该字典。
        抛出错误：FileNotFoundError: 如果注册表文件未找到，则抛出此异常
        抛出错误：Exception: 如果某个模型的初始化函数未找到，则抛出此异常
        返回：包含模型名称与对应初始化函数及参数配置的字典，
             键为模型名称，值为元组 (初始化函数, 参数配置)
        Initialize the model registry, load the registry file, and build a mapping between the model's initialization 
        function and parameter configuration.
        this method first checks if the registry file exists. if it does not exist, it logs a warning and throws an
        exception. if the file exists, it parses its content and iterates through each model configuration to verify whether
        the initialization function exists. it then stores the model name along with its corresponding initialization
        function and parameter configuration into a dictionary. finally, it returns this dictionary.
        :param: none.
        :raises FileNotFoundError: thrown if the registry file is not found.
        :raises Exception: thrown if an initialization function for a certain model is not found.
        :return: dict[str, tuple[callable, dict]]
                 A dictionary containing the model name as the key and a tuple of (initialization function,
                 parameter configuration) as the value.
        :rtype: dict[str, tuple[callable, dict]]
        """
        # 构造注册表文件的完整路径
        regi_path = Path(self.sets_path, self.name_regi)
        # 检查注册表文件是否存在
        if not regi_path.exists():
            # 如果文件不存在，记录警告日志
            self.root_logg.error(f"❗ 注册表文件缺失：{self.name_regi}")
            # 抛出错误代码
            raise FileNotFoundError(f"❗ 模型注册表文件 {self.name_regi} 未找到")
        # 打开注册表文件并加载其内容
        with open(regi_path, "r", encoding="utf-8") as regi_file:
            # 使用json模块加载文件内容
            regi_data = js.load(regi_file)
        # 初始化注册字典
        dict_mode_regi = {}
        # 加载模型名称与配置
        for mode_name, conf_mode in regi_data.items():
            # 获取模型函数名称
            init_func_name = conf_mode.get("init_func")
            # 获取模型参数配置（如果存在）
            init_func_conf = conf_mode.get("para_conf", {})
            # 判断函数是否存在
            if not hasattr(self, init_func_name):
                # 记录错误信息
                self.root_logg.error(f"❌ 初始化函数缺失：{init_func_name}")
                # 抛出错误代码
                raise AttributeError(f"❌ 模型 {init_func_name} 初始化函数未找到")
            # 配置模型函数
            init_func = getattr(self, init_func_name)
            # 将结果存入注册字典中
            dict_mode_regi[mode_name] = (init_func, init_func_conf)
        return dict_mode_regi

    def _init_simple_fit_model(self, **para_mode) -> BaseEstimator:
        """
        Initialize and return the specified machine learning model object.
        This method selects and instantiates the corresponding model object from a predefined model dictionary based on the
        provided model name and parameters. If the provided model name is not in the list of supported models, an error log
        is recorded and an exception is raised. After successful initialization, a log message is recorded, and a message
        indicating that the model has been loaded is printed.
        :param para_mode: A dictionary containing the model name and initialization parameters.
                          Must contain the key "mode_name", whose value is a string representing the model name.
                          The remaining key-value pairs will be passed as parameters for model initialization.
        :return: An instantiated machine learning model object, belonging to the BaseEstimator class or its subclass.
        :raises ValueError: If the provided model name is not in the list of supported models, this exception is raised.

        初始化并返回指定的机器学习模型对象。
        此方法根据提供的模型名称和参数，从预定义的模型字典中选择并实例化对应的模型对象。如果提供的模型名称不在支持的模型列表中，则记录错误日志并抛出异常。
        成功初始化后，会记录日志信息并打印模型加载完成的消息。
        :param para_mode: 包含模型名称和初始化参数的字典。
                          必须包含键 "mode_name"，其值为字符串类型，表示模型名称。
                          其余键值对将作为模型初始化的参数传递。
        :return: 实例化的机器学习模型对象，属于 BaseEstimator 类或其子类。
        :raises ValueError: 如果提供的模型名称不在支持的模型列表中，则抛出此异常。
        """
        mode_name = para_mode.pop("mode_name")  # 从参数字典中移除并获取键为"mode_name"的值，该值代表模型名称
        # 定义模型字典
        dict_mode = {
            "Ridge": Ridge,  # 将字符串"Ridge"映射到Ridge类
            "Lasso": Lasso,  # 将字符串"Lasso"映射到Lasso类
            "ElasticNet": ElasticNet,  # 将字符串"ElasticNet"映射到ElasticNet类
            "PLSReg": PLSRegression,  # 将字符串"PLSReg"映射到PLSRegression类
            "XGBoostReg": XGBRegressor,  # 将字符串"XGBoostReg"映射到XGBRegressor类
            "LightGBMReg": LGBMRegressor,  # 将字符串"LightGBMReg"映射到LGBMRegressor类
            "CatBoostReg": CatBoostRegressor,  # 将字符串"CatBoostReg"映射到CatBoostRegressor类
            "MLPRegressor": MLPRegressor,  # 将字符串"MLPRegressor"映射到MLPRegressor类
            "KNNReg": KNeighborsRegressor  # 将字符串"KNNReg"映射到KNeighborsRegressor类
            }
        if mode_name not in dict_mode:  # 检查提供的模型名称是否不在支持的模型字典中
            self.root_logg.error(f"❌ 不支持的模型：{mode_name}。")  # 如果模型不被支持，记录错误日志信息
            raise ValueError(f"❌ 不支持的模型：{mode_name}。")  # 抛出一个ValueError异常，提示模型不被支持
        mode_objt = dict_mode[mode_name](**para_mode)  # 使用提供的参数实例化对应的模型对象
        self.root_logg.info(f"✅ 模型：{mode_name}，已初始化完成。")  # 记录模型初始化信息
        print(f"✅ 模型：{mode_name}成功载入，准备开始训练。")  # 在控制台打印开始训练信息
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
        mode_objt = SVR(**para_mode)
        return mode_objt

    def _init_gp_model(self, **para_mode) -> BaseEstimator:
        """"""
        mode_name = para_mode.pop("mode_name")
        kern_sets = para_mode.pop("kernel", {})
        base_kern = para_mode.pop("base_kernel", "RBF")
        kern_type = {
            "RBF": RBF(length_scale=kern_sets.get("length_scale", 1.0)),
            "WhiteKernel": WhiteKernel(noise_level=kern_sets.get("noise_level", 1.0)),
            }.get(base_kern, RBF())
        mode_objt = GaussianProcessRegressor(kern_type, **para_mode)
        self.root_logg.info(f"✅ 模型：{mode_name}，已初始化完成。")
        print(f"✅ 模型：{mode_name}成功载入，准备开始训练。")
        return mode_objt

    def _train_single_model(self, mode_name: str, dict_spli: Dict[str, Any]) -> Dict[str, Any]:
        self.root_logg.info(f"▶ 开始训练模型：{mode_name}")
        if mode_name not in self.regi_mode:
            self.root_logg.error(f"❌ 未注册的模型：{mode_name}")
            raise KeyError(f"❌ 未注册的模型：{mode_name}")
            # 数据标准化提取
        x_train, y_train = dict_spli["tran_sets"]
        x_test, y_test = dict_spli["test_sets"]
        func_init, para_conf = self.regi_mode[mode_name]
        dict_resu = {
            "stts_train": "未达标",
            "best_r2": -np.inf,
            "n_retry": 0,
            "path_ckpt": None
            }
        para_tran = {'mode_name': mode_name, **para_conf}
        best_model = None
        for vari_atte in range(1, self.retr_maxm + 1):
            try:
                self.root_logg.info(f"🔄 尝试第 {vari_atte}/{self.retr_maxm} 次训练")
                print(f"正在尝试 {mode_name} 第 {vari_atte} 次训练...")
                # 模型初始化
                objt_mode = func_init(**para_tran)
                objt_mode.fit(x_train, y_train)
                # 性能评估
                y_pred = objt_mode.predict(x_test)
                current_r2 = r2_score(y_test, y_pred)
                # 更新最佳结果
                if current_r2 > dict_resu['best_r2']:
                    best_model = objt_mode
                    dict_resu.update(
                            {
                                'best_r2': current_r2,
                                'n_retry': vari_atte
                                }
                            )
                    self.root_logg.info(f"📈 更新最佳R²值：{current_r2:.6f}")
                    # 达标检查
                if current_r2 >= 0.75:
                    dict_resu['stts_train'] = '达标'
                    self.root_logg.info(f"✅ 第 {vari_atte} 次尝试达标")
                    break
            except Exception as e:
                error_msg = f"❌ 第 {vari_atte} 次训练失败：{str(e)}"
                self.root_logg.error(error_msg)
                print(error_msg)
                # 模型持久化
        if best_model and dict_resu['best_r2'] > -np.inf:
            stri_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{mode_name}_{stri_time}.joblib"
            path_ckpt = Path(self.ckpt_path, file_name)
            try:
                dump(best_model, path_ckpt)
                dict_resu['path_ckpt'] = str(path_ckpt)
                self.root_logg.info(f"💾 检查点已保存至：{path_ckpt}")
            except Exception as e:
                error_msg = f"❌ 模型保存失败：{str(e)}"
                self.root_logg.error(error_msg)
                print(error_msg)
                # 训练结果记录
        self.root_logg.info(
                f"▷ 训练完成：{mode_name} | 状态：{dict_resu['stts_train']} | "
                f"最佳R²：{dict_resu['best_r2']:.4f} | 尝试次数：{dict_resu['n_retry']}"
                )

        return dict_resu

    def run(self):
        spli_data = DataPreprocessing().run()
        for func_name in spli_data.keys():
            func_data = spli_data[func_name]
            for mode_name in self.regi_mode.keys():
                self._train_single_model(mode_name, func_data)


class DataPreprocessing:
    def __init__(self):
        """

        """

        self.base_path = AutoDataModelTrainerCore().base_path
        self.data_path = AutoDataModelTrainerCore().rezu_path
        self.sets_path = AutoDataModelTrainerCore().sets_path
        self.band_name = "sets_band_wave.json"
        self.refl_name = "rezu_spad_refl.csv"
        self.func_name = "sets_data_func.json"
        self.root_logg = AutoDataModelTrainerCore().root_logg
        self.dict_refl = self._init_reflectance_csv()
        self.band_wave = self._init_gain_wave_band()
        self.tran_rati = 0.8
        self.vali_rati = 0.1
        self.test_rati = 0.1
        self.spli_dids = self._init_random_dataset_selector()
        self.func_data = self._init_fetch_formula_config()

    def _init_random_dataset_selector(self):
        """

        """
        data_sids = list(range(1, 241))
        random.shuffle(data_sids)
        tran_size = int(self.tran_rati * len(data_sids))
        vali_size = int(self.vali_rati * len(data_sids))
        test_size = int(self.test_rati * len(data_sids))
        list_tran = data_sids[:tran_size]
        list_vali = data_sids[tran_size:(tran_size + vali_size)]
        list_test = data_sids[test_size:]
        dict_spli = {
            "tran_sets": list_tran,
            "vali_sets": list_vali,
            "test_sets": list_test
            }
        return dict_spli

    def _init_gain_wave_band(self):
        """

        """
        band_path = Path(self.sets_path, self.band_name)
        if not band_path.exists():
            self.root_logg.error(f"波段配置文件不存在：{band_path.name}。")
            raise FileNotFoundError(f"波段配置文件缺失：{band_path.name}。")
        with open(band_path, "r") as band_file:
            band_data = js.load(band_file)
        self.root_logg.info("波段配置文件正常读取")
        dict_band_wave = band_data.get("band_wave", {})
        return dict_band_wave

    def find_closest_band(self, targ_wave: float, thre_shol: float = 5.0) -> int:
        """

        """
        if targ_wave <= 0:
            self.root_logg.error(f"目标波长{targ_wave}必须为正数。")
            raise ValueError("目标波长必须为正数")
        if thre_shol <= 0:
            self.root_logg.error(f"目标波长{thre_shol}必须为正数")
            raise ValueError("阈值必须为正数")
        clos_band = None
        mini_diff = float('inf')
        # 遍历波段字典，寻找最接近的波段
        for band_numb, wave_lent in self.band_wave.items():
            diff_lent = abs(wave_lent - targ_wave)
            if diff_lent < mini_diff:
                mini_diff = diff_lent
                clos_band = band_numb
        # 检查是否找到有效波段
        if clos_band is None:
            raise ValueError("未找到符合条件的波段")
        # 如果最小差值超过阈值，记录日志
        if mini_diff > thre_shol:
            self.root_logg.warning(
                    f"目标波长：{targ_wave:.3f}nm 的最接近波段 {clos_band} "
                    f"({self.band_wave[clos_band]:.3f}nm) 的波长差为 {mini_diff:.3f}nm，超过限度。"
                    )
        return clos_band

    def create_index_function(self, func_stri):
        """

        """
        list_wave = {float(wave_leng) for wave_leng in re.findall(r'@(\d+\.?\d*)', func_stri)}
        replacements = {f'@{int(wave_leng)}': f"reflectance['{self.find_closest_band(wave_leng)}']"
                        for wave_leng in list_wave}
        stri_expr = func_stri
        for stri_part, stri_repl in replacements.items():
            stri_expr = stri_expr.replace(stri_part, stri_repl)
        return lambda func_refl: eval(
                stri_expr,
                {'__builtins__': None},
                {'reflectance': func_refl}
                )

    def _init_reflectance_csv(self):
        """

        """
        need_floa = {"SPAD"} | {f"Band_{numb_rows}" for numb_rows in range(1, 205)}
        refl_path = Path(self.data_path, self.refl_name)
        if not refl_path.exists():
            raise FileNotFoundError(f"反射率文件缺失：{refl_path}。")
        with open(refl_path, "r") as refl_file:
            data_read = csv.DictReader(refl_file)
            refl_data = {}
            for row in data_read:
                conv_rows = {}
                for band_keys, band_valu in row.items():
                    if band_keys in need_floa:
                        try:
                            # 尝试将数值字段转换为浮点数
                            conv_rows[band_keys] = float(band_valu)
                        except ValueError:
                            # 如果转换失败，记录警告日志并设置为 None
                            self.root_logg.warning(
                                    f"字段 {band_keys} 的值 {band_valu} 无法转换为浮点数，设置为 None"
                                    )
                            conv_rows[band_keys] = None
                    else:
                        # 非数值字段保持原始值
                        conv_rows[band_keys] = band_valu
                # 使用 ID 作为键存储转换后的行
                refl_data[conv_rows["ID"]] = conv_rows
        return refl_data

    def _init_fetch_formula_config(self):
        """

        """
        func_path = Path(self.sets_path, self.func_name)
        with open(func_path, "r", encoding="utf-8") as func_file:
            func_json = js.load(func_file)
        dict_func_sets = func_json["func_list"]
        return dict_func_sets

    def create_data_splits(self, func_data):
        dict_spli_data = {}
        tran_lisx = []
        tran_lisy = []
        vali_lisx = []
        vali_lisy = []
        test_lisx = []
        test_lisy = []
        for tran_cunt in self.spli_dids["tran_sets"]:
            tran_lisx.append([func_data[str(tran_cunt)]["comp_datx"], 1])
            tran_lisy.append(func_data[str(tran_cunt)]["comp_daty"])
        dict_spli_data["tran_sets"] = tran_lisx, tran_lisy
        for vali_cunt in self.spli_dids["vali_sets"]:
            vali_lisx.append([func_data[str(vali_cunt)]["comp_datx"], 1])
            vali_lisy.append(func_data[str(vali_cunt)]["comp_daty"])
        dict_spli_data["vali_sets"] = vali_lisx, vali_lisy
        for test_cunt in self.spli_dids["test_sets"]:
            test_lisx.append([func_data[str(test_cunt)]["comp_datx"], 1])
            test_lisy.append(func_data[str(test_cunt)]["comp_daty"])
        dict_spli_data["test_sets"] = test_lisx, test_lisy
        return dict_spli_data

    def compute_singel_vegetation_indices(self, func_name):
        stri_func = self.func_data[func_name]
        func_objt = self.create_index_function(stri_func)
        keyw_data = {}
        for coun_grop in range(1, 241):
            sign_refl = self.dict_refl[str(coun_grop)]
            func_rezu = func_objt(sign_refl)
            comp_daty = sign_refl["SPAD"]
            keyw_data[str(coun_grop)] = {
                "comp_datx": func_rezu,
                "comp_daty": comp_daty
                }
        spli_data = self.create_data_splits(keyw_data)
        return spli_data

    def run(self):
        dict_spli_data = {}
        for func_name in self.func_data.keys():
            func_data = self.compute_singel_vegetation_indices(func_name)
            dict_spli_data[func_name] = func_data
        return dict_spli_data


if __name__ == "__main__":
    AutoDataModelTrainerCore().run()
