import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import os
from datetime import datetime
import cv2

base_dir = r"C:\Users\22849\Desktop\文档\stone(1)\stone"

default_image_path = os.path.join(base_dir, "image1.png")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")

print("项目目录:", base_dir)
print("默认图像路径:", default_image_path)
print("默认图像是否存在:", os.path.exists(default_image_path))
print("models路径:", models_dir)
print("models是否存在:", os.path.exists(models_dir))
print("硬币模型是否存在:", os.path.exists(os.path.join(models_dir, "coin_instance_segmentation_final.pth")))
print("岩石模型是否存在:", os.path.exists(os.path.join(models_dir, "rock_instance_segmentation_final.pth")))

from utils.model_loader import ModelLoader
from utils.geometry_utils import GeometryCalculator
from utils.visualization import Visualizer
from utils.post_processing import PostProcessor


class RockGrainAnalyzer:
    """岩石粒径分析器主类"""

    def __init__(self, models_dir="models", results_dir="results", coin_diameter_mm=25.0):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.coin_diameter_mm = coin_diameter_mm

        os.makedirs(results_dir, exist_ok=True)

        self.model_loader = ModelLoader(models_dir)
        self.geometry_calc = GeometryCalculator(coin_diameter_mm)
        self.visualizer = Visualizer()
        self.post_processor = PostProcessor(iou_threshold=0.5, mask_overlap_threshold=0.3)

        self.coin_model = None
        self.rock_model = None

        print("岩石粒径分析器初始化完成")
        print(f"模型目录: {models_dir}")
        print(f"结果目录: {results_dir}")
        print(f"硬币直径: {coin_diameter_mm}mm")

    def load_models(self):
        """加载硬币和岩石识别模型"""
        try:
            print("正在加载模型...")
            self.coin_model = self.model_loader.load_coin_model()
            self.rock_model = self.model_loader.load_rock_model()
            print("所有模型加载完成！")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def enhance_image_for_segmentation(self, image_pil):
        """
        温和输入增强：
        - 双边滤波保边去噪
        - CLAHE局部对比度增强
        避免过强锐化带来的边界外扩
        """
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        filtered = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=55, sigmaSpace=55)

        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_rgb)

    def preprocess_image(self, image_path):
        """预处理图像"""
        image = Image.open(image_path).convert("RGB")
        enhanced_image = self.enhance_image_for_segmentation(image)
        image_tensor = F.to_tensor(enhanced_image).unsqueeze(0).to(self.model_loader.device)
        return enhanced_image, image_tensor

    def detect_coins(self, image_tensor):
        """检测硬币"""
        if self.coin_model is None:
            raise ValueError("硬币模型未加载")

        with torch.no_grad():
            predictions = self.coin_model(image_tensor)

        result = predictions[0]
        coin_results = {
            'boxes': result['boxes'].cpu().numpy(),
            'labels': result['labels'].cpu().numpy(),
            'masks': result['masks'].cpu().numpy(),
            'scores': result['scores'].cpu().numpy()
        }
        return coin_results

    def keep_best_coin_only(self, coin_results, score_threshold=0.5):
        """
        只保留最佳硬币
        """
        masks = coin_results['masks']
        scores = coin_results['scores']
        boxes = coin_results['boxes']
        labels = coin_results['labels']

        valid_indices = np.where(scores > score_threshold)[0]

        if len(valid_indices) == 0:
            return {
                'boxes': np.array([]),
                'labels': np.array([]),
                'masks': np.array([]),
                'scores': np.array([])
            }

        best_idx = None
        best_value = -1

        for idx in valid_indices:
            mask = masks[idx][0] if len(masks[idx].shape) == 3 else masks[idx]
            binary = (mask > 0.6).astype(np.uint8)
            area = np.sum(binary)

            if area == 0:
                continue

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            contour_area = cv2.contourArea(largest_contour)

            circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0.0
            score = scores[idx]

            final_value = score * 0.7 + circularity * 0.3

            if final_value > best_value:
                best_value = final_value
                best_idx = idx

        if best_idx is None:
            best_idx = valid_indices[np.argmax(scores[valid_indices])]

        print(f"硬币后处理：原始候选 {len(valid_indices)} 个，仅保留最佳硬币索引 {best_idx}")

        return {
            'boxes': np.array([boxes[best_idx]]),
            'labels': np.array([labels[best_idx]]),
            'masks': np.array([masks[best_idx]]),
            'scores': np.array([scores[best_idx]])
        }

    def detect_rocks_with_coin_masking(self, image_tensor, coin_mask=None):
        """
        检测岩石，如果提供了硬币mask则先遮蔽硬币区域
        第二轮改进：
        - 对硬币mask先膨胀，减少硬币边缘干扰
        """
        if self.rock_model is None:
            raise ValueError("岩石模型未加载")

        if coin_mask is not None:
            print("使用硬币mask遮蔽硬币区域进行岩石检测...")
            masked_image_tensor = image_tensor.clone()

            coin_mask_2d = coin_mask[0] if len(coin_mask.shape) == 3 else coin_mask

            _, _, h, w = masked_image_tensor.shape
            coin_mask_resized = cv2.resize(coin_mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
            coin_mask_binary = (coin_mask_resized > 0.6).astype(np.uint8)

            # 第二轮：膨胀硬币mask，扩大遮蔽范围
            kernel = np.ones((5, 5), np.uint8)
            coin_mask_binary = cv2.dilate(coin_mask_binary, kernel, iterations=1).astype(np.float32)

            for c in range(3):
                masked_image_tensor[0, c][coin_mask_binary == 1] = 1.0

            with torch.no_grad():
                predictions = self.rock_model(masked_image_tensor)
        else:
            with torch.no_grad():
                predictions = self.rock_model(image_tensor)

        result = predictions[0]
        rock_results = {
            'boxes': result['boxes'].cpu().numpy(),
            'labels': result['labels'].cpu().numpy(),
            'masks': result['masks'].cpu().numpy(),
            'scores': result['scores'].cpu().numpy()
        }
        return rock_results

    def detect_rocks(self, image_tensor):
        return self.detect_rocks_with_coin_masking(image_tensor, None)

    def set_scale_from_coins(self, coin_results, threshold=0.5):
        """从硬币检测结果设置比例尺"""
        masks = coin_results['masks']
        scores = coin_results['scores']

        best_coin_idx = None
        best_score = 0

        for i, score in enumerate(scores):
            if score > threshold and score > best_score:
                best_score = score
                best_coin_idx = i

        if best_coin_idx is not None:
            coin_mask = masks[best_coin_idx, 0]
            success = self.geometry_calc.set_scale_from_coin(coin_mask)
            if success:
                print(f"使用最佳硬币设置比例尺 (置信度: {best_score:.3f})")
                return True, {'index': best_coin_idx, 'score': best_score}

        print("未找到合适的硬币用于设置比例尺")
        return False, None

    def analyze_image(self, image_path, save_results=True, show_plots=True):
        """分析图像中的硬币和岩石"""
        print(f"\n开始分析图像: {image_path}")

        if self.coin_model is None or self.rock_model is None:
            if not self.load_models():
                return None

        image, image_tensor = self.preprocess_image(image_path)
        print(f"图像尺寸: {image.size}")

        print("正在检测硬币...")
        coin_results = self.detect_coins(image_tensor)
        coin_results = self.keep_best_coin_only(coin_results, score_threshold=0.5)
        total_coin_detections = len(coin_results['scores'])
        print(f"后处理后硬币数量: {total_coin_detections}")

        scale_set = False
        best_coin_info = None
        selected_coin_mask = None

        if total_coin_detections > 0:
            scale_set, best_coin_info = self.set_scale_from_coins(coin_results)
            if scale_set and best_coin_info is not None:
                coin_idx = best_coin_info['index']
                selected_coin_mask = coin_results['masks'][coin_idx]
                print(f"将使用硬币 {coin_idx} 进行遮蔽检测")

        print("正在检测岩石...")
        rock_results = self.detect_rocks_with_coin_masking(image_tensor, selected_coin_mask)

        # 第二轮：岩石置信度门槛提高
        raw_rock_count = np.sum(rock_results['scores'] > 0.6)
        print(f"检测到 {raw_rock_count} 个高置信度岩石候选")

        print("正在进行后处理...")
        rock_results = self.post_processor.process_results(
            rock_results,
            score_threshold=0.6,
            remove_overlaps=False,
            use_conflict_resolution=True,
            selected_coin_mask=selected_coin_mask,
            exclude_coin_region=False,
            remove_misclassified_coin=True,
            fix_fragmented_masks=True,
            resolve_boundary_conflicts=True,
            split_touching_instances=True,
            large_mask_area_threshold=1800,
            min_split_area=140,
            max_overlap_ratio=0.03
        )

        final_rock_count = np.sum(rock_results['scores'] > 0.6)
        print(f"后处理完成，最终岩石数量: {final_rock_count}")

        analysis_results = {
            'image_path': image_path,
            'image_size': image.size,
            'coin_results': coin_results,
            'rock_results': rock_results,
            'total_coin_detections': total_coin_detections,
            'best_coin_info': best_coin_info,
            'rock_count': final_rock_count,
            'original_rock_count': raw_rock_count,
            'scale_set': scale_set,
            'scale_info': self.geometry_calc.get_scale_info(),
            'timestamp': datetime.now().isoformat()
        }

        if scale_set and final_rock_count > 0:
            print("正在计算岩石粒径...")
            rock_data = self._calculate_rock_parameters(rock_results, threshold=0.6)
            analysis_results['rock_data'] = rock_data
            print(f"成功计算 {len(rock_data)} 个岩石的粒径参数")
        else:
            analysis_results['rock_data'] = []
            if not scale_set:
                print("警告: 未设置比例尺，无法计算实际粒径")

        if save_results or show_plots:
            self._save_and_visualize_results(image, analysis_results, save_results, show_plots)

        return analysis_results

    def _calculate_rock_parameters(self, rock_results, threshold=0.6):
        """计算岩石参数"""
        rock_data = []

        for i, (mask, score) in enumerate(zip(rock_results['masks'], rock_results['scores'])):
            if score > threshold:
                try:
                    current_mask = mask[0] if len(mask.shape) == 3 else mask
                    equiv_diameter = self.geometry_calc.calculate_equivalent_diameter(current_mask)
                    max_feret, min_feret = self.geometry_calc.calculate_feret_diameter(current_mask)
                    roundness = self.geometry_calc.calculate_roundness(current_mask)

                    rock_data.append({
                        'id': i + 1,
                        'equivalent_diameter_mm': equiv_diameter,
                        'max_feret_mm': max_feret,
                        'min_feret_mm': min_feret,
                        'roundness': roundness,
                        'confidence': score
                    })

                except Exception as e:
                    print(f"计算岩石 {i+1} 参数时出错: {e}")

        return rock_data

    def _save_and_visualize_results(self, image, analysis_results, save_results, show_plots):
        """保存和可视化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(analysis_results['image_path']))[0]

        result_image = self.visualizer.draw_detection_results(
            image,
            analysis_results['coin_results'],
            analysis_results['rock_results'],
            self.geometry_calc
        )

        if save_results:
            result_path = os.path.join(self.results_dir, f"{base_name}_result_{timestamp}.jpg")
            self.visualizer.save_results(result_image, result_path)
            print(f"结果图已保存到: {result_path}")

            if analysis_results['scale_set'] and analysis_results['rock_data']:
                report_path = os.path.join(self.results_dir, f"{base_name}_analysis_{timestamp}.png")
                self.visualizer.create_analysis_report(
                    analysis_results['coin_results'],
                    analysis_results['rock_results'],
                    self.geometry_calc,
                    report_path
                )
                print(f"分析报告已保存到: {report_path}")

        if show_plots:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 10))
            plt.imshow(result_image)
            plt.axis('off')
            plt.title(f'检测结果 - 硬币: {analysis_results["total_coin_detections"]}, 岩石: {analysis_results["rock_count"]}')
            plt.tight_layout()
            plt.show()

    def print_summary(self, analysis_results):
        """打印分析结果摘要"""
        print("\n" + "=" * 60)
        print("分析结果摘要")
        print("=" * 60)
        print(f"图像文件: {os.path.basename(analysis_results['image_path'])}")
        print(f"��像尺寸: {analysis_results['image_size']}")
        print(f"检测到硬币: {analysis_results['total_coin_detections']} 个")
        print(f"检测到岩石: {analysis_results['rock_count']} 个")
        print(f"比例尺设置: {'成功' if analysis_results['scale_set'] else '失败'}")

        if analysis_results['scale_info']:
            scale_info = analysis_results['scale_info']
            print(f"比例尺: {scale_info['pixels_per_mm']:.2f} 像素/毫米")

        if analysis_results['rock_data']:
            print("\n岩石粒径统计:")
            rock_data = analysis_results['rock_data']
            equiv_diameters = [r['equivalent_diameter_mm'] for r in rock_data]
            print(f"  等效直径范围: {min(equiv_diameters):.1f} - {max(equiv_diameters):.1f} mm")
            print(f"  平均等效直径: {np.mean(equiv_diameters):.1f} mm")
            print(f"  标准差: {np.std(equiv_diameters):.1f} mm")

        print("=" * 60)


def main():
    """主函数"""
    analyzer = RockGrainAnalyzer(
        models_dir=models_dir,
        results_dir=results_dir,
        coin_diameter_mm=25.0
    )

    current_image_path = r"C:\Users\22849\Desktop\文档\sea.png"
    print("当前将分析图片:", current_image_path)
    print("图片是否存在:", os.path.exists(current_image_path))

    if os.path.exists(current_image_path):
        results = analyzer.analyze_image(
            image_path=current_image_path,
            save_results=True,
            show_plots=True
        )

        if results:
            analyzer.print_summary(results)
        else:
            print("分析失败")
    else:
        print(f"测试图像不存在: {current_image_path}")


if __name__ == "__main__":
    main()