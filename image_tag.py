import csv
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger("app_logger")


def process_image(image_path, output_dir):
    """处理单个图像并保存结果（已修改排序逻辑）"""
    # 读取图像并转换颜色空间
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"警告：无法读取图像 {image_path}")
        return [], "", ""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 1: 消除白色干扰
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    image[white_mask == 255] = [0, 0, 0]

    # Step 2: 检测黄绿色区域
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Step 3: 形态学优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: 轮廓检测
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

    # Step 5: 计算中心点
    points = []
    for cnt in valid_contours:
        list_moment = cv2.moments(cnt)
        if list_moment["m00"] != 0:
            cx = int(list_moment["m10"] / list_moment["m00"])
            cy = int(list_moment["m01"] / list_moment["m00"])
        else:
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx), int(cy)
        points.append((cx, cy))

    # 新增排序逻辑：按列分组，每列从下到上排序
    if points:
        # 按X坐标排序初步分组
        points_sorted = sorted(points, key=lambda p: p[0])

        # 动态列分组（X坐标差超过20视为新列）
        columns = []
        current_col = [points_sorted[0]]
        for p in points_sorted[1:]:
            if abs(p[0] - current_col[-1][0]) > 20:
                columns.append(current_col)
                current_col = [p]
            else:
                current_col.append(p)
        columns.append(current_col)

        # 每列内部按Y从大到小排序（图像坐标系Y向下增大）
        for col in columns:
            col.sort(key=lambda p: -p[1])

        # 合并所有点并重新编号
        sorted_points = []
        for col in columns:
            sorted_points.extend(col)

        # 重新生成带序号的点集
        points = [(i + 1, x, y) for i, (x, y) in enumerate(sorted_points)]
    else:
        points = []

    # 生成调试图像
    debug_image = image.copy()
    overlay = image.copy()
    cv2.drawContours(overlay, valid_contours, -1, (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.3, debug_image, 0.7, 0, debug_image)
    """
    # 绘制序号和点（使用排序后的顺序）
    for idx, (x, y) in enumerate(sorted_points, 1):  # start=1
        cv2.circle(debug_image, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
                debug_image, f"{idx}", (x + 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
    """
    # 生成输出路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_img = os.path.join(output_dir, f"{base_name}_check1.png")
    output_csv = os.path.join(output_dir, f"{base_name}_points.csv")

    # 保存结果
    cv2.imwrite(output_img, debug_image)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "X", "Y"])
        for item in points:
            writer.writerow(item)

    return points, output_img, output_csv


def batch_process_images():
    input_dir = "./images"
    output_base = "./results"

    if not os.path.exists(input_dir):
        logger.error(f"错误：输入目录 {input_dir} 不存在")
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(input_dir, filename)

            # 创建输出目录
            output_dir = os.path.join(output_base, image_id)
            os.makedirs(output_dir, exist_ok=True)

            # 处理图像
            points, img_path, csv_path = process_image(image_path, output_dir)
            logger.info(
                    "\n" + "=" * 20 + f"\n文件：{filename}处理完成。\n本文件共检测到： {len(points)} 个点。\n生成校验图路径："
                                      f"{os.path.relpath(img_path)}\n提取坐标文件路径：{os.path.relpath(csv_path)}" + "\n" + "=" * 20
                    )


if __name__ == "__main__":
    batch_process_images()
