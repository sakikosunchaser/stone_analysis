import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2
from torchvision.transforms import ToTensor
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

class TransformWrapper:
    def __init__(self, image_size=(600, 600)):
        self.image_size = image_size
        self.to_tensor = ToTensor()

    def __call__(self, img, target):
        # 调整图像大小
        img = img.resize(self.image_size)
        img = self.to_tensor(img)

        # 调整掩膜大小
        masks = target["masks"].numpy()
        masks = np.array([cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST) for mask in masks])
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)

        # 调整边界框
        boxes = target["boxes"]
        width, height = self.image_size
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        target["boxes"] = boxes

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

class RockDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, image_size=(600, 600)):
        self.root = root
        self.transforms = transforms
        self.image_size = image_size
        self.imgs = list(sorted(os.listdir(os.path.join(root, "coins_image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "coins_mask"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "coins_image", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        mask_path = os.path.join(self.root, "coins_mask")
        mask_files = [f for f in self.masks if f.startswith(os.path.splitext(self.imgs[idx])[0])]
        masks = []
        for mask_file in mask_files:
            mask = Image.open(os.path.join(mask_path, mask_file)).convert("L")
            mask = np.array(mask)
            mask = (mask > 0).astype(np.uint8)  # 将掩膜转换为二值化
            masks.append(mask)
        
        if not masks:
            raise ValueError(f"No masks found for image {self.imgs[idx]}")

        # 调整图像和掩膜的大小到统一尺寸
        img = img.resize(self.image_size)
        masks = np.array([cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST) for mask in masks])

        num_objs = len(masks)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # 确保边界框的宽度和高度大于零
            if xmax <= xmin or ymax <= ymin:
                continue  # 跳过无效的边界框

            boxes.append([xmin, ymin, xmax, ymax])

        if not boxes:
            raise ValueError(f"No valid boxes found for image {self.imgs[idx]}")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model



# 定义数据集路径
root = r"E:\coins"

# 定义输出路径
output_dir = os.path.join(root, 'out\\try2')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义日志文件路径
log_file = os.path.join(output_dir, 'training_log.txt')

# 定义数据预处理
transform = TransformWrapper(image_size=(600, 600))

# 数据加载器
dataset = RockDataset(root, transforms=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(num_classes=2)  # 假设背景和岩石两类
model.to(device)

# 优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 训练
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 更新进度条
        progress_bar.set_postfix({"loss": f"{losses.item():.4f}"})

    # 打印每个 epoch 的平均损失
    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # 保存每个 epoch 的模型
    epoch_model_path = os.path.join(output_dir, f'plane_instance_segmentation_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), epoch_model_path)
    print(f"Epoch {epoch+1} 模型已保存到 {epoch_model_path}")

    # 记录每个 epoch 的损失到日志文件
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}\n")

# 保存最终模型
final_model_path = os.path.join(output_dir, 'plane_instance_segmentation_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"最终模型已保存到 {final_model_path}")