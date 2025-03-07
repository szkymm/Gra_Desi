import os
import time

from image_tag import batch_process_images  # 确保文件名为image_tag.py
from obtain_reflectance import batch_process  # 确保文件名为obtain_reflectance.py


def main():
    # 创建必要目录
    os.makedirs("./images", exist_ok=True)
    os.makedirs("./meta_data", exist_ok=True)

    # 阶段1: 图像标注处理
    print("\n" + "=" * 40)
    print("阶段1：图像特征点提取".center(36))
    print("=" * 40)
    start_time = time.time()

    batch_process_images()  # 调用image-tag的批量处理

    phase1_time = time.time() - start_time
    print(f"\n✅ 图像处理完成 耗时: {phase1_time:.1f}秒")

    # 阶段2: 反射率数据提取
    print("\n" + "=" * 40)
    print("阶段2：反射率数据提取".center(36))
    print("=" * 40)
    start_time = time.time()

    batch_process()  # 调用obtain-reflectance的批量处理

    phase2_time = time.time() - start_time
    print(f"\n✅ 反射率提取完成 耗时: {phase2_time:.1f}秒")

    # 最终统计
    total_time = phase1_time + phase2_time
    print("\n" + "=" * 40)
    print(f"🏁 全部处理完成 | 总耗时: {total_time:.1f}秒")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
