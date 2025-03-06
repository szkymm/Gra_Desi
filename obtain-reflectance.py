import os
import glob
import re
import pandas as pd
import numpy as np
import rasterio

def process_data(folder_path='./points', min_points=58, output_csv='reflectance_data_20250306_184437.csv'):
    """主处理函数：从dat文件中提取坐标点反射率，确保每个文件至少提取min_points个数据"""
    # 读取坐标文件
    xy_path = os.path.join(folder_path, 'xy2.xlsx')
    if not os.path.exists(xy_path):
        raise FileNotFoundError(f"坐标文件 {xy_path} 不存在")
    xy_df = pd.read_excel(xy_path, names=['id', 'x', 'y'])  # 强制列名

    # 获取所有dat文件
    dat_files = glob.glob(os.path.join(folder_path, '**', '*.dat'), recursive=True)
    print(f"找到 {len(dat_files)} 个dat文件")

    all_results = []
    for file_path in dat_files:
        # 从文件名提取数字ID
        file_id = int(re.search(r'(\d+)', os.path.basename(file_path)).group())

        # 匹配当前文件的所有坐标点
        points = xy_df[xy_df['id'] == file_id][['x', 'y']].values
        if len(points) < min_points:
            raise ValueError(f"文件 {file_id}.dat 的坐标点不足{min_points}个（当前{len(points)}个）")

        # 批量读取坐标点反射率
        try:
            with rasterio.open(file_path) as src:
                # 转换为0-based索引并转置为(y, x)格式
                indices = np.array([(y-1, x-1) for x, y in points], dtype=int)

                # 检查坐标越界
                if (indices >= [src.height, src.width]).any():
                    raise IndexError(f"文件 {file_id}.dat 中存在越界坐标")

                # 使用NumPy一次性读取所有波段和坐标点 (bands, points)
                reflectance = src.read()[:, indices[:, 0], indices[:, 1]].T  # 转置为(points, bands)

                # 构建结果数组
                file_results = np.column_stack([
                    np.repeat(os.path.basename(file_path), len(points)),  # 文件名
                    points[:, 0],  # x列
                    points[:, 1],  # y列
                    reflectance    # 波段数据
                ])
                all_results.append(file_results)
                print(f"文件 {file_id}.dat 处理完成，提取{len(points)}个点")

        except Exception as e:
            print(f"处理文件 {file_id}.dat 失败: {str(e)}")
            continue

    # 合并结果并保存
    if all_results:
        final_array = np.vstack(all_results)
        bands = [f'Band_{i+1}' for i in range(reflectance.shape[1])]
        final_df = pd.DataFrame(final_array, columns=['File Name', 'x', 'y'] + bands)
        final_df.to_csv(output_csv, index=False)
        print(f"数据已保存至 {output_csv}，共{len(final_df)}个数据点")
    else:
        print("无有效数据可保存")

if __name__ == "__main__":
    process_data(min_points=58)  # 强制每个文件至少提取60个点