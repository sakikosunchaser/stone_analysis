import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# 定义数据预处理函数
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    img = transform(img)
    return img

# 加载模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(num_classes=2)  # 假设背景和岩石两类
model.load_state_dict(torch.load(r"C:\Users\Lenovo\Desktop\rock_segmentation\out\rock_instance_segmentation_final.pth", map_location=device))
model.eval()
model.to(device)

# 加载测试图片
image_path = r"C:\Users\Lenovo\Desktop\rock_segmentation\second_stone_images\stone_images\stone1 (45).jpg"
img = preprocess_image(image_path)

# 将图片移动到设备
img = img.unsqueeze(0).to(device)  # 添加批次维度

# 进行预测
with torch.no_grad():
    predictions = model(img)

# 解析预测结果
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
masks = predictions[0]['masks'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

# 加载原始图像
original_img = Image.open(image_path).convert("RGB")
original_width, original_height = original_img.size

# 创建一个列表来存储所有单独的预测结果图像
all_instance_imgs = []

# 为每个检测框生成单独的图像
for i in range(len(boxes)):
    box = boxes[i]
    label = labels[i]
    score = scores[i]
    mask = masks[i, 0]  # 取第一个通道

    if score > 0.5:  # 只处理置信度大于 0.5 的实例
        # 创建一张与原始图像大小相同的图像
        single_instance_img = original_img.copy()
        draw = ImageDraw.Draw(single_instance_img)

        # 绘制边界框
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)
        draw.text((box[0], box[1]), f"Rock: {score:.2f}", fill="red")

        # 可视化掩膜
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((original_width, original_height), Image.NEAREST)
        single_instance_img.paste(mask_img, mask=mask_img)

        # 将当前实例图像添加到列表中
        all_instance_imgs.append(single_instance_img)

# 如果没有检测到任何有效实例，直接显示原始图像
if not all_instance_imgs:
    original_img.show(title="No valid detections")
else:
    # 计算堆叠图像的尺寸
    num_instances = len(all_instance_imgs)
    cols = int(np.ceil(np.sqrt(num_instances)))  # 每行的图像数量
    rows = int(np.ceil(num_instances / cols))  # 行数
    single_img_width, single_img_height = original_img.size
    stacked_img_width = cols * single_img_width
    stacked_img_height = rows * single_img_height

    # 创建堆叠图像
    stacked_img = Image.new('RGB', (stacked_img_width, stacked_img_height), (255, 255, 255))

    # 将所有实例图像粘贴到堆叠图像中
    for idx, instance_img in enumerate(all_instance_imgs):
        col = idx % cols
        row = idx // cols
        stacked_img.paste(instance_img, (col * single_img_width, row * single_img_height))

    # 显示堆叠后的图像
    stacked_img.show(title="All Instances")