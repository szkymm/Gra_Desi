import csv
import os

import cv2
import numpy as np


def process_image(image_path, output_dir):
    """处理单个图像并保存结果"""
    # 读取图像并转换颜色空间
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取图像 {image_path}")
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
    debug_image = image.copy()
    overlay = image.copy()
    cv2.drawContours(overlay, valid_contours, -1, (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.3, debug_image, 0.7, 0, debug_image)

    for idx, cnt in enumerate(valid_contours):
        list_moment = cv2.moments(cnt)
        if list_moment["m00"] != 0:
            cx = int(list_moment["m10"] / list_moment["m00"])
            cy = int(list_moment["m01"] / list_moment["m00"])
        else:
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)

        points.append((cx, cy))
        cv2.circle(debug_image, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(
            debug_image, f"{idx + 1}", (cx + 10, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

    # 生成输出路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_img = os.path.join(output_dir, f"{base_name}_check.png")
    output_csv = os.path.join(output_dir, f"{base_name}_points.csv")

    # 保存结果
    cv2.imwrite(output_img, debug_image)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "X", "Y"])
        for i, (x, y) in enumerate(points):
            writer.writerow([i + 1, x, y])

    return points, output_img, output_csv


def batch_process_images():
    """批量处理images目录下的所有PNG文件"""
    input_dir = "./images"
    output_base = "./results"

    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在")
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            image_id = os.path.splitext(filename)[0]
            image_path = os.path.join(input_dir, filename)

            # 创建输出目录：results/<image_id>/
            output_dir = os.path.join(output_base, image_id)
            os.makedirs(output_dir, exist_ok=True)

            # 处理图像
            points, img_path, csv_path = process_image(image_path, output_dir)
            print("==========")
            print(f"处理完成：{filename}")
            print(f"检测到 {len(points)} 个点")
            print(f"校验图路径：{os.path.relpath(img_path)}")
            print(f"坐标文件路径：{os.path.relpath(csv_path)}")


if __name__ == "__main__":
    batch_process_images()
