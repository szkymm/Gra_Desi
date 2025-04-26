import logging
import os
import time

from image_tag import batch_process_images  # 确保文件名为image_tag.py
from obtain_reflectance import batch_process as batch_process_reflectance  # 确保文件名为obtain_reflectance.py


# 配置日志系统
def setup_logger():
    logger_object = logging.getLogger("app_logger")
    logger_object.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # 创建文件处理器
    file_handler = logging.FileHandler("./logs/" + time.strftime("Obtain_Log_%Y-%m-%d_%H-%M-%S.log"), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 添加处理器
    logger_object.addHandler(console_handler)
    logger_object.addHandler(file_handler)

    return logger_object


logger = setup_logger()


def main():
    # 创建必要目录
    os.makedirs("./images", exist_ok=True)
    os.makedirs("./meta_data", exist_ok=True)

    # 阶段1: 图像标注处理
    logger.info("\n" + "=" * 40 + "\n阶段1：图像特征点提取" + "\n" + "=" * 40)
    start_time = time.time()

    try:
        batch_process_images()  # 调用image-tag的批量处理
        phase1_time = time.time() - start_time
        logger.info(f"✅ 图像处理完成 耗时: {phase1_time:.1f}秒")
    except Exception as e:
        logger.error(f"阶段1处理失败: {str(e)}")

    # 阶段2: 反射率数据提取
    logger.info("\n" + "=" * 40 + "\n阶段2：反射率数据提取" + "\n" + "=" * 40)
    start_time = time.time()

    try:
        batch_process_reflectance()  # 调用obtain-reflectance的批量处理
        phase2_time = time.time() - start_time
        logger.info(f"\n✅ 反射率提取完成 耗时: {phase2_time:.1f}秒")
    except Exception as e:
        logger.error(f"阶段2处理失败: {str(e)}")

    # 最终统计
    total_time = phase1_time + phase2_time
    logger.info("\n" + "=" * 40 + f"\n🏁 全部处理完成 | 总耗时: {total_time:.1f}秒" + "\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()
