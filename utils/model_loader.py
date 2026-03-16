import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os

class ModelLoader:
    """模型加载器类，用于加载硬币和岩石识别模型"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def get_model_instance_segmentation(self, num_classes):
        """创建Mask R-CNN模型实例"""
        # 使用新的API
        model = maskrcnn_resnet50_fpn(weights=None)
        
        # 替换分类器
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 替换掩膜预测器
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        return model
    
    def load_coin_model(self, model_name="coin_instance_segmentation_final.pth"):
        """加载硬币识别模型"""
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"硬币模型文件未找到: {model_path}")
        
        model = self.get_model_instance_segmentation(num_classes=2)  # 背景 + 硬币
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        print(f"硬币模型已加载: {model_path}")
        return model
    
    def load_rock_model(self, model_name="rock_instance_segmentation_final.pth"):
        """加载岩石识别模型"""
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"岩石模型文件未找到: {model_path}")
        
        model = self.get_model_instance_segmentation(num_classes=2)  # 背景 + 岩石
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        print(f"岩石模型已加载: {model_path}")
        return model 