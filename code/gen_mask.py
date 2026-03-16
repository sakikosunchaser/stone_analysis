import os
import csv
import numpy as np
import cv2
from PIL import Image
import json

# 定义保存掩膜的文件夹路径
output_folder = r"C:\Users\Lenovo\Desktop\coins\coins_mask"
os.makedirs(output_folder, exist_ok=True)

# 定义输入 CSV 文件夹路径
annotations_folder = r"C:\Users\Lenovo\Desktop\coins\coins_csv"

# 定义图像文件夹路径
image_folder = r"C:\Users\Lenovo\Desktop\coins\coins_image"

# 获取 annotations 文件夹中的所有 CSV 文件
csv_files = [f for f in os.listdir(annotations_folder) if f.endswith('.csv')]

# 遍历每个 CSV 文件
for csv_file in csv_files:
    csv_path = os.path.join(annotations_folder, csv_file)
    print(f"正在处理文件: {csv_path}")

    # 读取 CSV 文件
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)  # 将所有行读取到列表中

    # 如果 CSV 文件为空，跳过
    if not rows:
        print(f"文件 {csv_file} 为空，跳过。")
        continue

    # 遍历每个实例，生成掩膜并保存
    for row in rows:
        filename = row['filename']
        image_id = os.path.splitext(filename)[0]  # 获取图像的 ID（不带扩展名）
        region_id = row['region_id']
        mask_filename = f"{image_id}_{region_id}.png"  # 按照命名规则生成掩膜文件名
        mask_path = os.path.join(output_folder, mask_filename)

        # 获取图像路径并读取图像以获取宽度和高度
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        width, height = image.size

        # 创建一个全零掩膜，与原始图像大小一致
        mask = np.zeros((height, width), dtype=np.uint8)

        # 解析形状属性
        region_shape_attributes = json.loads(row['region_shape_attributes'])

        if region_shape_attributes['name'] == 'circle':
            # 圆形
            center = (int(region_shape_attributes['cx']), int(region_shape_attributes['cy']))
            radius = int(region_shape_attributes['r'])
            cv2.circle(mask, center, radius, 255, -1)  # 填充圆形
        elif region_shape_attributes['name'] == 'ellipse':
            # 椭圆
            center = (int(region_shape_attributes['cx']), int(region_shape_attributes['cy']))
            axes = (int(region_shape_attributes['rx']), int(region_shape_attributes['ry']))
            angle = int(region_shape_attributes['theta'])
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)  # 填充椭圆
        elif region_shape_attributes['name'] == 'polygon':
            # 多边形
            all_points_x = region_shape_attributes['all_points_x']
            all_points_y = region_shape_attributes['all_points_y']
            polygon = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)  # 填充多边形
        else:
            print(f"未知形状类型: {region_shape_attributes['name']}，跳过该实例。")
            continue

        # 保存掩膜图像
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_path)

        # 打印每个实例掩膜生成后的信息
        print(f"掩膜文件已生成: {mask_path}")

print(f"所有掩膜已成功生成并保存到 {output_folder}")