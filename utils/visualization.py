import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os
import colorsys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Visualizer:
    """可视化工具类"""
    
    def __init__(self):
        self.colors = {
            'coin': (255, 215, 0),      # 金色
            'rock': (139, 69, 19),      # 棕色
            'bbox': (255, 0, 0),        # 红色
            'text': (255, 255, 255),    # 白色
            'text_bg': (0, 0, 0),       # 黑色背景
            'mask_overlay': (0, 255, 0) # 绿色
        }
        
        # 尝试加载中文字体
        self.font_path = self._find_chinese_font()
    
    def _generate_colors(self, n):
        """生成n种不同的颜色"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + (i % 3) * 0.1  # 0.7-0.9
            value = 0.8 + (i % 2) * 0.2       # 0.8-1.0
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    def _find_chinese_font(self):
        """查找系统中的中文字体"""
        possible_fonts = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
        ]
        
        for font_path in possible_fonts:
            if os.path.exists(font_path):
                return font_path
        return None
    
    def _put_text_with_background(self, img, text, position, font_scale=0.6, 
                                 text_color=(255, 255, 255), bg_color=(0, 0, 0), 
                                 thickness=2, padding=5):
        """在图像上绘制带背景的文本"""
        x, y = position
        
        if self.font_path:
            try:
                # 使用PIL绘制中文
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                # 计算字体大小
                font_size = int(20 * font_scale)
                font = ImageFont.truetype(self.font_path, font_size)
                
                # 获取文本尺寸
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 绘制背景矩形
                bg_rect = [x - padding, y - text_height - padding, 
                          x + text_width + padding, y + padding]
                draw.rectangle(bg_rect, fill=bg_color[::-1], outline=None)
                
                # 绘制文本
                draw.text((x, y - text_height), text, font=font, fill=text_color[::-1])
                
                # 转换回OpenCV格式
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                img[:] = img_cv[:]
                return True
            except Exception as e:
                print(f"中文字体绘制失败: {e}")
        
        # 回退到OpenCV默认字体
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            text.encode('ascii', 'ignore').decode('ascii'), 
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # 绘制背景矩形
        cv2.rectangle(img, 
                     (x - padding, y - text_height - padding - baseline),
                     (x + text_width + padding, y + padding),
                     bg_color, -1)
        
        # 绘制文本
        cv2.putText(img, text.encode('ascii', 'ignore').decode('ascii'), 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        return False
    
    def draw_detection_results(self, image, coin_results, rock_results, geometry_calc):
        """
        绘制检测结果
        Args:
            image: 原始图像（PIL Image或numpy array）
            coin_results: 硬币检测结果
            rock_results: 岩石检测结果
            geometry_calc: 几何计算器实例
        Returns:
            result_image: 绘制了结果的图像
        """
        # 确保图像是numpy array格式
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # 转换为BGR格式（OpenCV格式）
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array.copy()
        
        # 绘制硬币检测结果
        if coin_results:
            self._draw_coins(img_bgr, coin_results, geometry_calc)
        
        # 绘制岩石检测结果
        if rock_results:
            self._draw_rocks(img_bgr, rock_results, geometry_calc)
        
        # 转换回RGB格式
        result_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return result_image
    
    def _draw_coins(self, img_bgr, coin_results, geometry_calc):
        """绘制硬币检测结果"""
        boxes = coin_results['boxes']
        masks = coin_results['masks']
        scores = coin_results['scores']
        
        # 只显示最佳硬币
        best_coin_idx = None
        best_score = 0
        
        for i, score in enumerate(scores):
            if score > 0.5 and score > best_score:
                best_score = score
                best_coin_idx = i
        
        if best_coin_idx is not None:
            i = best_coin_idx
            box = boxes[i]
            mask = masks[i]
            score = scores[i]
            
            x1, y1, x2, y2 = box.astype(int)
            
            # 处理掩膜 - 确保正确的维度和数据类型
            if len(mask.shape) == 3:
                mask_2d = mask[0]  # 取第一个通道
            else:
                mask_2d = mask
            
            # 调整掩膜尺寸到图像尺寸
            h, w = img_bgr.shape[:2]
            mask_resized = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            
            # 创建彩色半透明覆盖层
            overlay = img_bgr.copy()
            overlay[binary_mask == 1] = self.colors['coin']
            cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0, img_bgr)
            
            # 查找轮廓并绘制边缘
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, contours, -1, self.colors['coin'], 3)
            
            # 计算硬币直径
            diameter_pixels = geometry_calc.calculate_coin_diameter_pixels(mask_2d)
            if diameter_pixels:
                # 绘制标签
                label = f"硬币: {geometry_calc.coin_diameter_mm}mm ({score:.2f})"
                self._put_text_with_background(img_bgr, label, (x1, y1-10), 
                                             text_color=(255, 255, 255), 
                                             bg_color=(0, 0, 0))
    
    def _draw_rocks(self, img_bgr, rock_results, geometry_calc):
        """绘制岩石检测结果"""
        boxes = rock_results['boxes']
        masks = rock_results['masks']
        scores = rock_results['scores']
        
        # 筛选有效的岩石
        valid_rocks = [(i, score) for i, score in enumerate(scores) if score > 0.5]
        
        if not valid_rocks:
            return
        
        # 为每个岩石生成不同颜色
        rock_colors = self._generate_colors(len(valid_rocks))
        
        for idx, (i, score) in enumerate(valid_rocks):
            box = boxes[i]
            mask = masks[i]
            rock_color = rock_colors[idx]
            
            x1, y1, x2, y2 = box.astype(int)
            
            # 处理掩膜 - 确保正确的维度和数据类型
            if len(mask.shape) == 3:
                mask_2d = mask[0]  # 取第一个通道
            else:
                mask_2d = mask
            
            # 调整掩膜尺寸到图像尺寸
            h, w = img_bgr.shape[:2]
            mask_resized = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            
            # 创建彩色半透明覆盖层
            overlay = img_bgr.copy()
            overlay[binary_mask == 1] = rock_color
            cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
            
            # 查找轮廓并绘制边缘
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, contours, -1, rock_color, 2)
            
            # 计算岩石粒径
            try:
                equiv_diameter = geometry_calc.calculate_equivalent_diameter(mask_2d)
                max_feret, min_feret = geometry_calc.calculate_feret_diameter(mask_2d)
                roundness = geometry_calc.calculate_roundness(mask_2d)
                
                # 绘制标签 - 使用更紧凑的格式
                label1 = f"#{idx+1}: {equiv_diameter:.1f}mm"
                label2 = f"F:{max_feret:.1f}×{min_feret:.1f} R:{roundness:.2f}"
                
                # 统一使用白色文字和黑色背景，确保可读性
                self._put_text_with_background(img_bgr, label1, (x1, y1-30), 
                                             text_color=(255, 255, 255), bg_color=(0, 0, 0))
                self._put_text_with_background(img_bgr, label2, (x1, y1-10), 
                                             text_color=(255, 255, 255), bg_color=(0, 0, 0))
                
            except ValueError as e:
                # 如果没有设置比例尺，只显示基本信息
                label = f"岩石 #{idx+1} ({score:.2f})"
                self._put_text_with_background(img_bgr, label, (x1, y1-10), 
                                             text_color=(255, 255, 255), bg_color=(0, 0, 0))
    
    def create_analysis_report(self, coin_results, rock_results, geometry_calc, save_path=None):
        """
        创建分析报告图表
        Args:
            coin_results: 硬币检测结果
            rock_results: 岩石检测结果
            geometry_calc: 几何计算器实例
            save_path: 保存路径
        """
        if not rock_results or geometry_calc.pixels_per_mm is None:
            print("无法生成分析报告：缺少岩石检测结果或比例尺信息")
            return
        
        # 计算所有岩石的粒径数据
        rock_data = []
        for i, (mask, score) in enumerate(zip(rock_results['masks'], rock_results['scores'])):
            if score > 0.5:
                try:
                    # 处理掩膜维度
                    if len(mask.shape) == 3:
                        mask_2d = mask[0]  # 取第一个通道
                    else:
                        mask_2d = mask
                    
                    equiv_diameter = geometry_calc.calculate_equivalent_diameter(mask_2d)
                    max_feret, min_feret = geometry_calc.calculate_feret_diameter(mask_2d)
                    roundness = geometry_calc.calculate_roundness(mask_2d)
                    
                    rock_data.append({
                        'id': i+1,
                        'equivalent_diameter': equiv_diameter,
                        'max_feret': max_feret,
                        'min_feret': min_feret,
                        'roundness': roundness,
                        'score': score
                    })
                except Exception as e:
                    print(f"计算岩石 {i+1} 的参数时出错: {e}")
        
        if not rock_data:
            print("没有有效的岩石数据用于分析")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('岩石粒径分析报告', fontsize=16, fontweight='bold')
        
        # 提取数据
        equiv_diameters = [r['equivalent_diameter'] for r in rock_data]
        max_ferets = [r['max_feret'] for r in rock_data]
        min_ferets = [r['min_feret'] for r in rock_data]
        roundness_values = [r['roundness'] for r in rock_data]
        
        # 1. 等效直径分布直方图
        axes[0, 0].hist(equiv_diameters, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('等效直径分布')
        axes[0, 0].set_xlabel('等效直径 (mm)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feret直径散点图
        axes[0, 1].scatter(max_ferets, min_ferets, alpha=0.7, color='orange')
        axes[0, 1].set_title('Feret直径关系')
        axes[0, 1].set_xlabel('最大Feret直径 (mm)')
        axes[0, 1].set_ylabel('最小Feret直径 (mm)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加对角线
        max_val = max(max(max_ferets), max(min_ferets))
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1线')
        axes[0, 1].legend()
        
        # 3. 圆度分布
        axes[1, 0].hist(roundness_values, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('圆度分布')
        axes[1, 0].set_xlabel('圆度')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 粒径统计表
        axes[1, 1].axis('off')
        
        # 计算统计数据
        stats_data = [
            ['参数', '最小值', '最大值', '平均值', '标准差'],
            ['等效直径 (mm)', f'{min(equiv_diameters):.1f}', f'{max(equiv_diameters):.1f}', 
             f'{np.mean(equiv_diameters):.1f}', f'{np.std(equiv_diameters):.1f}'],
            ['最大Feret (mm)', f'{min(max_ferets):.1f}', f'{max(max_ferets):.1f}', 
             f'{np.mean(max_ferets):.1f}', f'{np.std(max_ferets):.1f}'],
            ['最小Feret (mm)', f'{min(min_ferets):.1f}', f'{max(min_ferets):.1f}', 
             f'{np.mean(min_ferets):.1f}', f'{np.std(min_ferets):.1f}'],
            ['圆度', f'{min(roundness_values):.2f}', f'{max(roundness_values):.2f}', 
             f'{np.mean(roundness_values):.2f}', f'{np.std(roundness_values):.2f}']
        ]
        
        table = axes[1, 1].table(cellText=stats_data[1:], colLabels=stats_data[0], 
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('统计数据')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分析报告已保存到: {save_path}")
        
        plt.show()
        
        return rock_data
    
    def save_results(self, result_image, save_path):
        """保存结果图像"""
        if isinstance(result_image, np.ndarray):
            result_pil = Image.fromarray(result_image)
        else:
            result_pil = result_image
        
        result_pil.save(save_path)
        print(f"结果图像已保存到: {save_path}") 