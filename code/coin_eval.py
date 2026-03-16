import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义加载模型的函数
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights=None)  # 使用 weights 参数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# 加载模型
def load_model(model_path, num_classes):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

# 可视化函数
def visualize_predictions(image, predictions, threshold=0.5):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    masks = predictions['masks'].cpu().numpy()
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # 定义颜色
    boundary_color = (255, 0, 0)  # 红色，用于边界框和掩膜轮廓
    mask_color = (255, 255, 255)  # 白色，用于掩膜填充
    bound_box_color = (0, 0, 255)

    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i]
            mask = masks[i, 0]
            label = labels[i]
            score = scores[i]

            # 绘制边界框（红色）
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), bound_box_color, 2)

            # 处理掩膜
            mask = (mask > threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 填充掩膜区域（白色）
            cv2.drawContours(image, contours, -1, mask_color, -1)  # 填充白色

            # 绘制掩膜轮廓（红色）
            cv2.drawContours(image, contours, -1, boundary_color, 2)  # 轮廓红色

            # 添加标签和置信度（红色）
            label_text = f"Class {label} ({score:.2f})"
            cv2.putText(image, label_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bound_box_color, 2)

    return image

# 测试函数
def test_model(model_path, test_image_path, num_classes=2, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, num_classes)
    model.to(device)

    # 读取测试图片
    image = Image.open(test_image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # 运行模型
    with torch.no_grad():
        predictions = model(image_tensor)

    # 可视化结果
    result_image = visualize_predictions(image, predictions[0], threshold)

    # 将原图和处理过的图像横向拼接
    original_image = np.array(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    combined_image = np.hstack((original_image, result_image))

    # 显示结果
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()



# 测试
if __name__ == "__main__":
    model_path = r"E:\coins\out\try2\plane_instance_segmentation_final.pth"  # 替换为你的模型路径
    test_image_path = r"E:\coins\test_image\3.jpg"  # 替换为你的测试图片路径
    test_model(model_path, test_image_path)