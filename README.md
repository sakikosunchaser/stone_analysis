# 岩石粒径分析系统

基于深度学习的岩石粒径自动测量系统，使用硬币作为比例尺进行精确的粒径计算。

## 功能特点

- **硬币识别**: 自动识别图像中的硬币并设置比例尺
- **岩石检测**: 使用Mask R-CNN进行岩石实例分割
- **粒径计算**: 计算多种地质学常用的粒径参数
  - 等效直径 (Equivalent Diameter)
  - Feret直径 (最大/最小投影长度)
  - 圆度 (Roundness)
- **可视化分析**: 生成详细的分析报告和统计图表
- **结果保存**: 自动保存检测结果和分析报告

## 项目结构

```
stone/
├── models/                          # 模型文件目录
│   ├── coin_instance_segmentation_final.pth
│   └── rock_instance_segmentation_final.pth
├── utils/                           # 工具类
│   ├── model_loader.py             # 模型加载器
│   ├── geometry_utils.py           # 几何计算工具
│   └── visualization.py            # 可视化工具
├── results/                         # 结果保存目录
├── code/                           # 原始代码（备份）
├── rock_grain_analyzer.py          # 主分析器
├── requirements.txt                # 依赖包
├── README.md                       # 说明文档
└── stone.png                       # 测试图像
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from rock_grain_analyzer import RockGrainAnalyzer

# 创建分析器实例
analyzer = RockGrainAnalyzer(
    models_dir="models",
    results_dir="results", 
    coin_diameter_mm=25.0  # 1元硬币直径
)

# 分析图像
results = analyzer.analyze_image(
    image_path="stone.png",
    save_results=True,
    show_plots=True
)

# 打印结果摘要
analyzer.print_summary(results)
```

### 命令行使用

```bash
python rock_grain_analyzer.py
```

## 粒径参数说明

### 等效直径 (Equivalent Diameter)
- 定义: 与岩石颗粒面积相等的圆的直径
- 公式: $D_{eq} = 2 × \sqrt{Area / π}$
- 用途: 地质学中最常用的粒径表示方法

### Feret直径
- **最大Feret直径**: 颗粒在所有方向上的最大投影长度
- **最小Feret直径**: 颗粒在所有方向上的最小投影长度
- 用途: 描述颗粒的形状特征

### 圆度 (Roundness)
- 定义: $4π × Area / Perimeter²$
- 范围: $[0-1]$，1表示完美圆形
- 用途: 量化颗粒的圆度程度

## 比例尺设置

系统使用硬币作为比例尺：
- 默认硬币直径: 25.0mm (中国1元硬币)
- 自动检测置信度最高的硬币
- 计算像素/毫米比例关系

## 输出结果

### 检测结果图像
- 显示所有检测到的硬币和岩石
- 标注粒径参数和置信度
- 保存为高分辨率图像

### 分析报告
- 等效直径分布直方图
- Feret直径关系散点图
- 圆度分布统计
- 详细统计数据表格

## 注意事项

1. **图像质量**: 确保图像清晰，硬币和岩石边界明显
2. **硬币类型**: 默认使用25mm直径硬币，可根据实际情况调整
3. **光照条件**: 避免强烈阴影和反光
4. **比例尺**: 必须包含至少一个清晰的硬币用作比例尺

## 技术细节

- **深度学习框架**: PyTorch
- **检测模型**: Mask R-CNN (ResNet-50 backbone)
- **图像处理**: OpenCV, PIL
- **数值计算**: NumPy, SciPy
- **可视化**: Matplotlib

## 系统要求

- Python 3.7+
- CUDA支持的GPU (推荐)
- 内存: 8GB+ (推荐)
- 存储: 2GB+ (用于模型文件) 