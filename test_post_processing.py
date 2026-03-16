import numpy as np
import cv2
from utils.post_processing import PostProcessor


def create_test_data():
    """创建测试数据：一个大mask与多个小mask重叠的情况"""
    height, width = 400, 400

    masks = []
    boxes = []
    scores = []

    large_mask = np.zeros((height, width), dtype=np.float32)
    large_mask[100:300, 100:300] = 1.0
    masks.append(np.expand_dims(large_mask, axis=0))
    boxes.append([100, 100, 300, 300])
    scores.append(0.8)

    small_mask1 = np.zeros((height, width), dtype=np.float32)
    small_mask1[120:160, 120:160] = 1.0
    masks.append(np.expand_dims(small_mask1, axis=0))
    boxes.append([120, 120, 160, 160])
    scores.append(0.9)

    small_mask2 = np.zeros((height, width), dtype=np.float32)
    small_mask2[180:220, 180:220] = 1.0
    masks.append(np.expand_dims(small_mask2, axis=0))
    boxes.append([180, 180, 220, 220])
    scores.append(0.85)

    small_mask3 = np.zeros((height, width), dtype=np.float32)
    small_mask3[240:280, 140:180] = 1.0
    masks.append(np.expand_dims(small_mask3, axis=0))
    boxes.append([140, 240, 180, 280])
    scores.append(0.75)

    independent_mask = np.zeros((height, width), dtype=np.float32)
    independent_mask[50:90, 320:360] = 1.0
    masks.append(np.expand_dims(independent_mask, axis=0))
    boxes.append([320, 50, 360, 90])
    scores.append(0.7)

    results = {
        'masks': np.array(masks, dtype=np.float32),
        'boxes': np.array(boxes, dtype=np.float32),
        'scores': np.array(scores, dtype=np.float32),
        'labels': np.ones(len(masks), dtype=np.int64)
    }

    return results


def create_touching_rocks_data():
    """创建两个粘连石块的测试mask"""
    height, width = 300, 300

    touching_mask = np.zeros((height, width), dtype=np.float32)
    cv2.circle(touching_mask, (110, 150), 45, 1.0, -1)
    cv2.circle(touching_mask, (170, 150), 45, 1.0, -1)

    results = {
        'masks': np.array([np.expand_dims(touching_mask, axis=0)], dtype=np.float32),
        'boxes': np.array([[65, 105, 215, 195]], dtype=np.float32),
        'scores': np.array([0.92], dtype=np.float32),
        'labels': np.array([1], dtype=np.int64)
    }

    return results


def create_fragmented_mask_data():
    """创建一个被分裂成多个连通区域的mask测试数据"""
    height, width = 300, 300

    fragmented_mask = np.zeros((height, width), dtype=np.float32)
    fragmented_mask[60:110, 60:110] = 1.0
    fragmented_mask[160:220, 160:220] = 1.0

    results = {
        'masks': np.array([np.expand_dims(fragmented_mask, axis=0)], dtype=np.float32),
        'boxes': np.array([[60, 60, 220, 220]], dtype=np.float32),
        'scores': np.array([0.88], dtype=np.float32),
        'labels': np.array([1], dtype=np.int64)
    }

    return results


def get_mask_area(mask):
    current_mask = mask[0] if len(mask.shape) == 3 else mask
    return np.sum(current_mask > 0.5)


def visualize_results(original_results, processed_results, title="Results"):
    print(f"\n=== {title} ===")
    print(f"原始检测数量: {len(original_results['masks'])}")
    print(f"处理后数量: {len(processed_results['masks'])}")

    print("\n原始检测:")
    for i, (box, score) in enumerate(zip(original_results['boxes'], original_results['scores'])):
        mask_area = get_mask_area(original_results['masks'][i])
        print(f"  Mask {i}: 边界框={box}, 置信度={score:.3f}, 面积={mask_area}")

    print("\n处理后检测:")
    for i, (box, score) in enumerate(zip(processed_results['boxes'], processed_results['scores'])):
        mask_area = get_mask_area(processed_results['masks'][i])
        print(f"  Mask {i}: 边界框={box}, 置信度={score:.3f}, 面积={mask_area}")


def test_conflict_resolution(processor):
    print("创建测试数据...")
    test_results = create_test_data()

    print("\n测试场景：一个大mask (200x200) 与三个小mask (40x40) 重叠")
    print("预期结果：应该移除大mask，保留三个小mask和一个独立mask")

    print("\n" + "=" * 50)
    print("使用新的冲突解决策略")
    print("=" * 50)
    processed_results = processor.resolve_mask_conflicts(test_results, score_threshold=0.5)
    visualize_results(test_results, processed_results, "新冲突解决策略")

    print("\n" + "=" * 50)
    print("使用旧的重叠检测去除方法")
    print("=" * 50)
    processed_results_old = processor.remove_overlapping_detections(test_results, score_threshold=0.5)
    visualize_results(test_results, processed_results_old, "旧重叠检测去除")

    print("\n" + "=" * 50)
    print("使用完整后处理流程（新策略）")
    print("=" * 50)
    final_results = processor.process_results(
        test_results,
        score_threshold=0.5,
        use_conflict_resolution=True,
        split_touching_instances=True,
        large_mask_area_threshold=3000,
        min_split_area=80
    )
    visualize_results(test_results, final_results, "完整后处理流程")


def test_touching_rocks_split(processor):
    print("\n" + "=" * 50)
    print("测试粘连石块拆分")
    print("=" * 50)

    touching_results = create_touching_rocks_data()
    visualize_results(touching_results, touching_results, "原始粘连石块")

    split_results = processor.split_large_masks_in_results(
        touching_results,
        score_threshold=0.5,
        area_threshold=1000,
        min_split_area=50
    )
    visualize_results(touching_results, split_results, "粘连石块拆分结果")


def test_fragmented_mask_fix(processor):
    print("\n" + "=" * 50)
    print("测试分裂mask修复")
    print("=" * 50)

    fragmented_results = create_fragmented_mask_data()
    visualize_results(fragmented_results, fragmented_results, "原始分裂mask")

    fixed_results = processor.fix_fragmented_masks(fragmented_results)
    visualize_results(fragmented_results, fixed_results, "修复后的mask")


def test_post_processing():
    processor = PostProcessor(iou_threshold=0.3, mask_overlap_threshold=0.2)

    test_conflict_resolution(processor)
    test_touching_rocks_split(processor)
    test_fragmented_mask_fix(processor)


if __name__ == "__main__":
    test_post_processing()