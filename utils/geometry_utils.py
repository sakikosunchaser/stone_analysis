import numpy as np
import cv2
from scipy import ndimage
import math

class GeometryCalculator:
    """几何计算工具类"""
    
    def __init__(self, coin_diameter_mm=25.0):
        """
        初始化几何计算器
        Args:
            coin_diameter_mm: 硬币的实际直径（毫米），默认为25mm（1元硬币）
        """
        self.coin_diameter_mm = coin_diameter_mm
        self.pixels_per_mm = None
    
    def calculate_coin_diameter_pixels(self, mask):
        """
        计算硬币在图像中的像素直径
        Args:
            mask: 硬币的二值掩膜
        Returns:
            diameter_pixels: 硬币的像素直径
        """
        # 确保掩膜是二值的
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 如果掩膜太小，可能需要调整尺寸
        if binary_mask.shape[0] < 10 or binary_mask.shape[1] < 10:
            print(f"警告: 掩膜尺寸太小 {binary_mask.shape}")
            return None
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter_pixels = 2 * radius
        
        return diameter_pixels
    
    def set_scale_from_coin(self, coin_mask):
        """
        根据硬币设置比例尺
        Args:
            coin_mask: 硬币的掩膜
        """
        diameter_pixels = self.calculate_coin_diameter_pixels(coin_mask)
        if diameter_pixels is not None and diameter_pixels > 0:
            self.pixels_per_mm = diameter_pixels / self.coin_diameter_mm
            print(f"比例尺设置完成: {self.pixels_per_mm:.2f} 像素/毫米")
            return True
        return False
    
    def calculate_equivalent_diameter(self, mask):
        """
        计算等效直径（地质学中常用的粒径定义）
        等效直径 = 2 * sqrt(面积 / π)
        Args:
            mask: 岩石的二值掩膜
        Returns:
            equivalent_diameter_mm: 等效直径（毫米）
        """
        if self.pixels_per_mm is None:
            raise ValueError("请先设置比例尺（通过硬币）")
        
        # 确保掩膜是二值的
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 计算面积（像素数）
        area_pixels = np.sum(binary_mask)
        
        if area_pixels == 0:
            return 0
        
        # 计算等效直径（像素）
        equivalent_diameter_pixels = 2 * math.sqrt(area_pixels / math.pi)
        
        # 转换为毫米
        equivalent_diameter_mm = equivalent_diameter_pixels / self.pixels_per_mm
        
        return equivalent_diameter_mm
    
    def calculate_feret_diameter(self, mask):
        """
        计算Feret直径（最大投影长度）
        Args:
            mask: 岩石的二值掩膜
        Returns:
            max_feret_mm: 最大Feret直径（毫米）
            min_feret_mm: 最小Feret直径（毫米）
        """
        if self.pixels_per_mm is None:
            raise ValueError("请先设置比例尺（通过硬币）")
        
        # 确保掩膜是二值的
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        width, height = rect[1]
        
        # Feret直径
        max_feret_pixels = max(width, height)
        min_feret_pixels = min(width, height)
        
        # 转换为毫米
        max_feret_mm = max_feret_pixels / self.pixels_per_mm
        min_feret_mm = min_feret_pixels / self.pixels_per_mm
        
        return max_feret_mm, min_feret_mm
    
    def calculate_roundness(self, mask):
        """
        计算圆度（4π×面积/周长²）
        Args:
            mask: 岩石的二值掩膜
        Returns:
            roundness: 圆度值（0-1，1为完美圆形）
        """
        # 确保掩膜是二值的
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算面积和周长
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return None
        
        # 计算圆度
        roundness = 4 * math.pi * area / (perimeter ** 2)
        
        return roundness
    
    def get_scale_info(self):
        """获取当前比例尺信息"""
        if self.pixels_per_mm is not None:
            return {
                'pixels_per_mm': self.pixels_per_mm,
                'mm_per_pixel': 1.0 / self.pixels_per_mm,
                'coin_diameter_mm': self.coin_diameter_mm
            }
        return None 