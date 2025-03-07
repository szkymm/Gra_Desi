import os
import glob
import re
import pandas as pd
import numpy as np
import rasterio

def process_data(image_id, output_base="./results", min_points=60):
    """处理单个图像ID对应的dat文件和坐标数据"""
    # 路径配置
    dat_path = os.path.join("./meta_data", image_id, f"results/REFLECTANCE_{image_id}.dat")
    coord_csv = os.path.join(output_base, image_id, f"{image_id}_points.csv")
    output_csv = os.path.join(output_base, image_id, f"reflectance_{image_id}.csv")

    # 验证文件存在性
    if not os.path.exists(dat_path):
        print(f"跳过 {image_id}: 未找到.dat文件 {os.path.basename(dat_path)}")
        return
    if not os.path.exists(coord_csv):
        print(f"跳过 {image_id}: 未找到坐标文件 {os.path.basename(coord_csv)}")
        return

    # 读取坐标数据
    try:
        xy_df = pd.read_csv(coord_csv)
        if len(xy_df) < min_points:
            print(f"警告：{image_id} 坐标点不足{min_points}个（当前{len(xy_df)}个）")
            return
    except Exception as e:
        print(f"读取坐标文件失败：{coord_csv}\n错误详情：{str(e)}")
        return

    # 确保xy_df存在后继续执行
    try:
        with rasterio.open(dat_path) as src:
            # 转换坐标索引（修复xy_df引用）
            indices = np.array([(y-1, x-1) for x, y in xy_df[["X", "Y"]].values], dtype=int)

            # 边界检查
            if (indices >= [src.height, src.width]).any():
                print(f"错误：{image_id} 存在越界坐标")
                return

            # 批量读取反射率
            reflectance = src.read()[:, indices[:, 0], indices[:, 1]].T

            # 构建结果DataFrame
            results = pd.DataFrame(
                np.column_stack([xy_df[["ID", "X", "Y"]].values, reflectance]),
                columns=["ID", "X", "Y"] + [f"Band_{i+1}" for i in range(reflectance.shape[1])]
            )

            # 保存结果
            results.to_csv(output_csv, index=False)
            print(f"成功处理：{image_id}")
            print(f"  输出文件：{os.path.relpath(output_csv)}")
            print(f"  包含数据：{len(results)}条记录\n")

    except Exception as e:
        print(f"处理{image_id}失败：{str(e)}")

def batch_process():
    """批量处理所有有效图像ID"""
    # 自动发现所有可能存在的image_id
    dat_files = glob.glob("./meta_data/**/REFLECTANCE_*.dat", recursive=True)
    image_ids = list(set(re.findall(r"REFLECTANCE_(\d+)\.dat", f)[0] for f in dat_files))

    print(f"找到 {len(image_ids)} 个待处理图像ID")

    for image_id in image_ids:
        process_data(image_id)

if __name__ == "__main__":
    batch_process()