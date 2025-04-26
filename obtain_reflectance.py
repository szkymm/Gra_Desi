import glob
import logging
import os
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning

# 获取全局logger
logger = logging.getLogger("app_logger")


def validate_files(dat_path, coord_csv, image_id):
    """验证文件是否存在"""
    if not os.path.exists(dat_path):
        logger.warning(f"跳过 {image_id}: 未找到.dat文件 {os.path.basename(dat_path)}")
        return False
    if not os.path.exists(coord_csv):
        logger.warning(f"跳过 {image_id}: 未找到坐标文件 {os.path.basename(coord_csv)}")
        return False
    return True


def read_coordinates(coord_csv, min_points, image_id):
    """读取并验证坐标数据"""
    try:
        coordinates_df = pd.read_csv(coord_csv)
        if len(coordinates_df) < min_points:
            logger.warning(f"{image_id} 坐标点不足{min_points}个（当前{len(coordinates_df)}个）")
            return None
        return coordinates_df
    except Exception as e:
        logger.error(f"读取坐标文件失败：{coord_csv}\n错误详情：{str(e)}")
        return None


def process_reflectance(dat_path, coordinates_df, output_csv, image_id):
    """处理单个图像ID的反射率数据"""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(dat_path) as src:
                logger.info(f"成功打开文件 {dat_path}")
                if not src.transform or src.transform == rasterio.Affine.identity():
                    logger.warning(f"警告❕ 文件缺少地理参考信息")

                # 转换坐标索引
                coordinate_indices = np.array(
                        [(y - 1, x - 1) for x, y in coordinates_df[["X", "Y"]].values], dtype=int
                        )

                # 边界检查
                if (coordinate_indices >= [src.height, src.width]).any():
                    logger.error(f"错误：{image_id} 存在越界坐标")
                    return False

                # 批量读取反射率
                reflectance_data = src.read()[:, coordinate_indices[:, 0], coordinate_indices[:, 1]].T

                # 构建结果DataFrame
                columns = ["ID", "X", "Y"] + [f"Band_{i + 1}" for i in range(reflectance_data.shape[1])]
                reflectance_results = pd.DataFrame(
                        np.column_stack([coordinates_df[["ID", "X", "Y"]].values, reflectance_data]),
                        columns=columns
                        )

                # 保存结果
                reflectance_results.to_csv(output_csv, index=False)
                logger.info(
                    "\n" + "=" * 20 + f"\n成功处理：{image_id}\n输出文件：{os.path.relpath(output_csv)}\n包含数据："
                                      f"{len(reflectance_results)}条记录" + "\n" + "=" * 20
                    )
                return True
    except Exception as e:
        logger.error(f"处理{image_id}失败：{str(e)}")
        return False


def process_data(image_id, output_base=".\\results", min_points=59):
    """处理单个图像ID对应的dat文件和坐标数据"""
    # 路径配置
    dat_path = os.path.join(".\\meta_data", image_id, f"results\\REFLECTANCE_{image_id}.dat")
    coord_csv = os.path.join(output_base, image_id, f"{image_id}_points.csv")
    output_csv = os.path.join(output_base, image_id, f"reflectance_{image_id}.csv")

    # 验证文件存在性
    if not validate_files(dat_path, coord_csv, image_id):
        return

    # 读取坐标数据
    coordinates_df = read_coordinates(coord_csv, min_points, image_id)
    if coordinates_df is None:
        return

    # 处理反射率数据
    process_reflectance(dat_path, coordinates_df, output_csv, image_id)


def batch_process():
    """批量处理所有有效图像ID"""
    # 自动发现所有可能存在的image_id
    dat_files = glob.glob(".\\meta_data\\**\\REFLECTANCE_*.dat", recursive=True)
    image_ids = list(set(re.findall(r"REFLECTANCE_(\d+)\.dat", f)[0] for f in dat_files))

    logger.info(f"找到 {len(image_ids)} 个待处理图像ID")

    for image_id in image_ids:
        process_data(image_id)


if __name__ == "__main__":
    batch_process()
