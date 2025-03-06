import cv2
import numpy as np
import os

def process_image_with_markers(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 1: 颜色区域分割
    lower_green = (25, 40, 40)
    upper_green = (90, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_white = (0, 0, 200)
    upper_white = (180, 30, 255)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    combined_mask = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_white))

    # Step 2: 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    refined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Step 3: 轮廓检测与过滤
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = image.shape[0] * image.shape[1]
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < img_area * 0.05]

    # Step 4: 计算特征点
    points = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for contour in filtered_contours:
        contour_mask = np.zeros_like(refined_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        ys, xs = np.where(contour_mask == 255)

        colors = image_rgb[ys, xs]
        avg_color = np.mean(colors, axis=0)
        distances = np.linalg.norm(colors - avg_color, axis=1)
        min_idx = np.argmin(distances)
        points.append((xs[min_idx], ys[min_idx]))

    # Step 5: 在图像上绘制标记点
    marked_image = image.copy()
    for (x, y) in points:
        # 绘制红色圆形标记（BGR颜色空间）
        cv2.circle(marked_image, (x, y), 5, (0, 0, 255), -1)    # 实心圆
        cv2.circle(marked_image, (x, y), 7, (255, 255, 255), 2)  # 白色边框

    # Step 6: 保存校验图像
    base_name, ext = os.path.splitext(image_path)
    output_path = f"{base_name}_check{ext}"
    cv2.imwrite(output_path, marked_image)
    print(f"校验图像已保存至：{output_path}")

    return points

# 使用示例
file_path_name = "./png/1723.png"
result_points = process_image_with_markers(file_path_name)
print("检测到的最近点坐标：")
for idx, (x, y) in enumerate(result_points):
    print(f"{file_path_name.replace(r'./png/','').replace('.png','')}|{x}|{y}")