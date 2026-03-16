import numpy as np
import cv2
import torch
from scipy import ndimage


class PostProcessor:
    """后处理工具类，用于去除重叠检测和优化结果"""

    def __init__(self, iou_threshold=0.5, mask_overlap_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.mask_overlap_threshold = mask_overlap_threshold

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_mask_overlap(self, mask1, mask2):
        binary_mask1 = (mask1 > 0.5).astype(np.uint8)
        binary_mask2 = (mask2 > 0.5).astype(np.uint8)

        intersection = np.logical_and(binary_mask1, binary_mask2)
        intersection_area = np.sum(intersection)

        area1 = np.sum(binary_mask1)
        area2 = np.sum(binary_mask2)

        if area1 == 0 or area2 == 0:
            return 0.0

        smaller_area = min(area1, area2)
        overlap_ratio = intersection_area / smaller_area

        return overlap_ratio

    def refine_mask_edges(self, mask, threshold=0.6, erosion_iter=1, close_iter=1):
        """
        基础边界精修：
        - 提高阈值
        - 轻微闭运算让轮廓更完整
        - 轻微腐蚀减少外扩
        """
        binary = (mask > threshold).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)

        if close_iter > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        if erosion_iter > 0:
            binary = cv2.erode(binary, kernel, iterations=erosion_iter)

        return binary.astype(np.float32)

    def refine_mask_edges_adaptive(self, mask):
        """
        自适应边界精修：
        大块更严格，小块更温和
        """
        area = np.sum(mask > 0.5)

        if area > 15000:
            return self.refine_mask_edges(mask, threshold=0.65, erosion_iter=2, close_iter=1)
        elif area > 7000:
            return self.refine_mask_edges(mask, threshold=0.62, erosion_iter=2, close_iter=1)
        elif area > 2500:
            return self.refine_mask_edges(mask, threshold=0.60, erosion_iter=1, close_iter=1)
        else:
            return self.refine_mask_edges(mask, threshold=0.55, erosion_iter=1, close_iter=1)

    def refine_results_masks(self, results, min_area=140):
        """
        对整组结果统一做自适应mask边界精修
        """
        if len(results['masks']) == 0:
            return results

        refined_masks = []
        refined_boxes = []
        refined_scores = []
        refined_labels = []

        labels = results.get('labels', np.ones(len(results['scores'])))

        for box, mask, score, label in zip(results['boxes'], results['masks'], results['scores'], labels):
            current_mask = mask[0] if len(mask.shape) == 3 else mask
            refined = self.refine_mask_edges_adaptive(current_mask)

            if np.sum(refined) < min_area:
                continue

            ys, xs = np.where(refined > 0.5)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            refined_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            refined_masks.append(np.expand_dims(refined, axis=0))
            refined_scores.append(score)
            refined_labels.append(label)

        if len(refined_masks) == 0:
            return {
                'boxes': np.array([]),
                'masks': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }

        return {
            'boxes': np.array(refined_boxes, dtype=np.float32),
            'masks': np.array(refined_masks, dtype=np.float32),
            'scores': np.array(refined_scores, dtype=np.float32),
            'labels': np.array(refined_labels, dtype=np.int64)
        }

    def remove_overlapping_detections(self, results, score_threshold=0.6):
        boxes = results['boxes']
        masks = results['masks']
        scores = results['scores']
        labels = results.get('labels', np.ones(len(scores)))

        valid_indices = scores > score_threshold
        boxes = boxes[valid_indices]
        masks = masks[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]

        if len(boxes) == 0:
            return {
                'boxes': np.array([]),
                'masks': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }

        areas = []
        for mask in masks:
            mask_2d = mask[0] if len(mask.shape) == 3 else mask
            area = np.sum(mask_2d > 0.5)
            areas.append(area)
        areas = np.array(areas)

        sorted_indices = np.argsort(areas)
        keep_indices = []
        removed_indices = set()

        for idx in sorted_indices:
            if idx in removed_indices:
                continue

            should_keep = True

            for kept_idx in keep_indices:
                if kept_idx in removed_indices:
                    continue

                iou = self.calculate_iou(boxes[idx], boxes[kept_idx])

                mask_current = masks[idx][0] if len(masks[idx].shape) == 3 else masks[idx]
                mask_kept = masks[kept_idx][0] if len(masks[kept_idx].shape) == 3 else masks[kept_idx]

                if mask_current.shape != mask_kept.shape:
                    target_shape = (
                        max(mask_current.shape[0], mask_kept.shape[0]),
                        max(mask_current.shape[1], mask_kept.shape[1])
                    )
                    mask_current = cv2.resize(mask_current, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_kept = cv2.resize(mask_kept, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

                mask_overlap = self.calculate_mask_overlap(mask_current, mask_kept)

                if iou > self.iou_threshold or mask_overlap > self.mask_overlap_threshold:
                    if areas[idx] >= areas[kept_idx]:
                        should_keep = False
                        break
                    else:
                        removed_indices.add(kept_idx)

            if should_keep:
                keep_indices.append(idx)

        final_keep_indices = [idx for idx in keep_indices if idx not in removed_indices]
        final_keep_indices = sorted(final_keep_indices)

        return {
            'boxes': boxes[final_keep_indices],
            'masks': masks[final_keep_indices],
            'scores': scores[final_keep_indices],
            'labels': labels[final_keep_indices]
        }

    def find_overlapping_masks(self, masks, boxes, areas, target_idx):
        overlapping_indices = []

        for i, mask in enumerate(masks):
            if i == target_idx:
                continue

            iou = self.calculate_iou(boxes[target_idx], boxes[i])

            mask_target = masks[target_idx][0] if len(masks[target_idx].shape) == 3 else masks[target_idx]
            mask_current = mask[0] if len(mask.shape) == 3 else mask

            if mask_target.shape != mask_current.shape:
                target_shape = (
                    max(mask_target.shape[0], mask_current.shape[0]),
                    max(mask_target.shape[1], mask_current.shape[1])
                )
                mask_target = cv2.resize(mask_target, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_current = cv2.resize(mask_current, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

            mask_overlap = self.calculate_mask_overlap(mask_target, mask_current)

            if iou > self.iou_threshold or mask_overlap > self.mask_overlap_threshold:
                overlapping_indices.append(i)

        return overlapping_indices

    def resolve_mask_conflicts(self, results, score_threshold=0.6):
        boxes = results['boxes']
        masks = results['masks']
        scores = results['scores']
        labels = results.get('labels', np.ones(len(scores)))

        valid_indices = scores > score_threshold
        boxes = boxes[valid_indices]
        masks = masks[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]

        if len(boxes) == 0:
            return {
                'boxes': np.array([]),
                'masks': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }

        areas = []
        for mask in masks:
            mask_2d = mask[0] if len(mask.shape) == 3 else mask
            area = np.sum(mask_2d > 0.5)
            areas.append(area)
        areas = np.array(areas)

        conflict_groups = []
        processed = set()

        for i in range(len(masks)):
            if i in processed:
                continue

            overlapping = self.find_overlapping_masks(masks, boxes, areas, i)
            if overlapping:
                conflict_group = [i] + overlapping
                conflict_groups.append(conflict_group)
                processed.update(conflict_group)

        keep_indices = []

        for i in range(len(masks)):
            if i not in processed:
                keep_indices.append(i)

        for group in conflict_groups:
            group_areas = [areas[idx] for idx in group]
            group_scores = [scores[idx] for idx in group]

            sorted_group = sorted(zip(group, group_areas, group_scores), key=lambda x: x[1])

            largest_idx = sorted_group[-1][0]
            smaller_masks = [idx for idx, area, score in sorted_group[:-1]]

            if len(smaller_masks) >= 2:
                keep_indices.extend(smaller_masks)
            else:
                if len(sorted_group) == 2:
                    idx1, area1, score1 = sorted_group[0]
                    idx2, area2, score2 = sorted_group[1]

                    if abs(score1 - score2) > 0.2:
                        keep_indices.append(idx1 if score1 > score2 else idx2)
                    else:
                        keep_indices.append(idx1)
                else:
                    keep_indices.append(sorted_group[0][0])

        keep_indices = sorted(keep_indices)

        return {
            'boxes': boxes[keep_indices],
            'masks': masks[keep_indices],
            'scores': scores[keep_indices],
            'labels': labels[keep_indices]
        }

    def exclude_selected_coin(self, rock_results, selected_coin_mask, overlap_threshold=0.8):
        if selected_coin_mask is None:
            return rock_results

        coin_mask_2d = selected_coin_mask[0] if len(selected_coin_mask.shape) == 3 else selected_coin_mask
        keep_indices = []

        for rock_idx, (rock_mask, rock_score) in enumerate(zip(rock_results['masks'], rock_results['scores'])):
            if rock_score <= 0.6:
                continue

            rock_mask_2d = rock_mask[0] if len(rock_mask.shape) == 3 else rock_mask

            if rock_mask_2d.shape != coin_mask_2d.shape:
                target_shape = (
                    max(rock_mask_2d.shape[0], coin_mask_2d.shape[0]),
                    max(rock_mask_2d.shape[1], coin_mask_2d.shape[1])
                )
                rock_mask_resized = cv2.resize(rock_mask_2d, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                coin_mask_resized = cv2.resize(coin_mask_2d, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                rock_mask_resized = rock_mask_2d
                coin_mask_resized = coin_mask_2d

            overlap_ratio = self.calculate_mask_overlap(rock_mask_resized, coin_mask_resized)

            if overlap_ratio <= overlap_threshold:
                keep_indices.append(rock_idx)

        if keep_indices:
            return {
                'boxes': rock_results['boxes'][keep_indices],
                'masks': rock_results['masks'][keep_indices],
                'scores': rock_results['scores'][keep_indices],
                'labels': rock_results['labels'][keep_indices] if 'labels' in rock_results else np.ones(len(keep_indices))
            }

        return {
            'boxes': np.array([]),
            'masks': np.array([]),
            'scores': np.array([]),
            'labels': np.array([])
        }

    def find_and_remove_coin_misclassified_as_rock(self, rock_results, selected_coin_mask, overlap_threshold=0.5):
        if selected_coin_mask is None:
            return rock_results

        coin_mask_2d = selected_coin_mask[0] if len(selected_coin_mask.shape) == 3 else selected_coin_mask
        coin_area = np.sum(coin_mask_2d > 0.5)

        best_match_idx = None
        best_similarity = 0

        for rock_idx, (rock_mask, rock_score) in enumerate(zip(rock_results['masks'], rock_results['scores'])):
            if rock_score <= 0.6:
                continue

            rock_mask_2d = rock_mask[0] if len(rock_mask.shape) == 3 else rock_mask

            if rock_mask_2d.shape != coin_mask_2d.shape:
                target_shape = (
                    max(rock_mask_2d.shape[0], coin_mask_2d.shape[0]),
                    max(rock_mask_2d.shape[1], coin_mask_2d.shape[1])
                )
                rock_mask_resized = cv2.resize(rock_mask_2d, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                coin_mask_resized = cv2.resize(coin_mask_2d, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                rock_mask_resized = rock_mask_2d
                coin_mask_resized = coin_mask_2d

            overlap = self.calculate_mask_overlap(coin_mask_resized, rock_mask_resized)
            rock_area = np.sum(rock_mask_resized > 0.5)
            area_ratio = min(coin_area, rock_area) / max(coin_area, rock_area) if max(coin_area, rock_area) > 0 else 0
            similarity = overlap * 0.7 + area_ratio * 0.3

            if overlap > overlap_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = rock_idx

        if best_match_idx is not None:
            keep_indices = [i for i in range(len(rock_results['masks'])) if i != best_match_idx]

            if keep_indices:
                return {
                    'boxes': rock_results['boxes'][keep_indices],
                    'masks': rock_results['masks'][keep_indices],
                    'scores': rock_results['scores'][keep_indices],
                    'labels': rock_results['labels'][keep_indices] if 'labels' in rock_results else np.ones(len(keep_indices))
                }

        return rock_results

    def fix_fragmented_masks(self, results):
        masks = results['masks']
        boxes = results['boxes']
        scores = results['scores']
        labels = results.get('labels', np.ones(len(scores)))

        fixed_masks = []
        keep_indices = []

        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score <= 0.6:
                continue

            if len(mask.shape) == 3:
                mask_2d = mask[0]
                original_shape = mask.shape
            else:
                mask_2d = mask
                original_shape = mask.shape

            binary_mask = (mask_2d > 0.5).astype(np.uint8)
            num_labels, labels_map = cv2.connectedComponents(binary_mask)

            if num_labels > 2:
                max_area = 0
                max_label = 0

                for label in range(1, num_labels):
                    area = np.sum(labels_map == label)
                    if area > max_area:
                        max_area = area
                        max_label = label

                fixed_mask_2d = (labels_map == max_label).astype(np.float32)

                if len(original_shape) == 3:
                    fixed_mask = np.zeros(original_shape, dtype=np.float32)
                    fixed_mask[0] = fixed_mask_2d
                else:
                    fixed_mask = fixed_mask_2d
            else:
                fixed_mask = mask

            fixed_masks.append(fixed_mask)
            keep_indices.append(i)

        if keep_indices:
            return {
                'boxes': boxes[keep_indices],
                'masks': np.array(fixed_masks),
                'scores': scores[keep_indices],
                'labels': labels[keep_indices]
            }

        return {
            'boxes': np.array([]),
            'masks': np.array([]),
            'scores': np.array([]),
            'labels': np.array([])
        }

    def split_touching_rocks(self, mask, min_area=140):
        """
        第二轮：
        更适合大块粘连的拆分
        """
        binary = (mask > 0.6).astype(np.uint8)

        if np.sum(binary) < min_area:
            return [binary.astype(np.float32)]

        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return [binary.astype(np.float32)]

        _, sure_fg = cv2.threshold(dist, 0.32 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(sure_fg)

        if num_labels <= 2:
            return [binary.astype(np.float32)]

        sub_masks = []
        for label_idx in range(1, num_labels):
            part = (labels == label_idx).astype(np.uint8)
            part = cv2.dilate(part, kernel, iterations=3)
            part = np.logical_and(part > 0, binary > 0).astype(np.uint8)

            if np.sum(part) >= min_area:
                refined = self.refine_mask_edges_adaptive(part.astype(np.float32))
                if np.sum(refined) >= min_area:
                    sub_masks.append(refined.astype(np.float32))

        if len(sub_masks) == 0:
            return [binary.astype(np.float32)]

        return sub_masks

    def split_large_masks_in_results(self, results, score_threshold=0.6, area_threshold=1800, min_split_area=140):
        new_boxes = []
        new_labels = []
        new_masks = []
        new_scores = []

        labels = results.get('labels', np.ones(len(results['scores'])))

        for box, label, mask, score in zip(results['boxes'], labels, results['masks'], results['scores']):
            current_mask = mask[0] if len(mask.shape) == 3 else mask
            area = np.sum(current_mask > 0.5)

            refined_direct = self.refine_mask_edges_adaptive(current_mask)

            # 小块不拆，只精修
            if score <= score_threshold or area < area_threshold:
                if np.sum(refined_direct) < min_split_area:
                    continue

                ys, xs = np.where(refined_direct > 0.5)
                if len(xs) == 0 or len(ys) == 0:
                    continue

                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

                new_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                new_labels.append(label)
                new_masks.append(np.expand_dims(refined_direct.astype(np.float32), axis=0))
                new_scores.append(score)
                continue

            # 大块尝试拆分
            sub_masks = self.split_touching_rocks(current_mask, min_area=min_split_area)

            if len(sub_masks) == 1:
                refined = self.refine_mask_edges_adaptive(sub_masks[0])

                if np.sum(refined) < min_split_area:
                    continue

                ys, xs = np.where(refined > 0.5)
                if len(xs) == 0 or len(ys) == 0:
                    continue

                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

                new_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                new_labels.append(label)
                new_masks.append(np.expand_dims(refined.astype(np.float32), axis=0))
                new_scores.append(score)
            else:
                print(f"大mask拆分成功: 1 -> {len(sub_masks)}")

                for sub_mask in sub_masks:
                    ys, xs = np.where(sub_mask > 0.5)
                    if len(xs) == 0 or len(ys) == 0:
                        continue

                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()

                    new_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                    new_labels.append(label)
                    new_masks.append(np.expand_dims(sub_mask.astype(np.float32), axis=0))
                    new_scores.append(float(score) * 0.95)

        if len(new_masks) == 0:
            return {
                'boxes': np.array([]),
                'labels': np.array([]),
                'masks': np.array([]),
                'scores': np.array([])
            }

        return {
            'boxes': np.array(new_boxes, dtype=np.float32),
            'labels': np.array(new_labels, dtype=np.int64),
            'masks': np.array(new_masks, dtype=np.float32),
            'scores': np.array(new_scores, dtype=np.float32)
        }

    def resolve_mask_boundary_conflicts(self, results, overlap_threshold=0.03, inclusion_threshold=0.8):
        boxes = results['boxes']
        masks = results['masks']
        scores = results['scores']
        labels = results.get('labels', np.ones(len(scores)))

        if len(boxes) == 0:
            return results

        modified_masks = [mask.copy() for mask in masks]
        keep_indices = list(range(len(masks)))

        overlapping_pairs = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                mask_i = modified_masks[i][0] if len(modified_masks[i].shape) == 3 else modified_masks[i]
                mask_j = modified_masks[j][0] if len(modified_masks[j].shape) == 3 else modified_masks[j]

                if mask_i.shape != mask_j.shape:
                    target_shape = (
                        max(mask_i.shape[0], mask_j.shape[0]),
                        max(mask_i.shape[1], mask_j.shape[1])
                    )
                    mask_i = cv2.resize(mask_i, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                    mask_j = cv2.resize(mask_j, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

                binary_i = (mask_i > 0.5).astype(np.uint8)
                binary_j = (mask_j > 0.5).astype(np.uint8)

                intersection = np.logical_and(binary_i, binary_j)
                intersection_area = np.sum(intersection)

                area_i = np.sum(binary_i)
                area_j = np.sum(binary_j)

                if area_i == 0 or area_j == 0 or intersection_area == 0:
                    continue

                overlap_ratio_i = intersection_area / area_i
                overlap_ratio_j = intersection_area / area_j

                if max(overlap_ratio_i, overlap_ratio_j) > overlap_threshold:
                    overlapping_pairs.append((i, j, binary_i, binary_j, intersection, overlap_ratio_i, overlap_ratio_j))

        for i, j, binary_i, binary_j, intersection, overlap_i, overlap_j in overlapping_pairs:
            if i not in keep_indices or j not in keep_indices:
                continue

            if overlap_i > inclusion_threshold:
                new_j = binary_j.copy()
                new_j[intersection > 0] = 0
                self._update_mask_with_binary(modified_masks, j, new_j.astype(np.float32))
            elif overlap_j > inclusion_threshold:
                new_i = binary_i.copy()
                new_i[intersection > 0] = 0
                self._update_mask_with_binary(modified_masks, i, new_i.astype(np.float32))

        final_keep_indices = []
        for idx in keep_indices:
            current_mask = modified_masks[idx][0] if len(modified_masks[idx].shape) == 3 else modified_masks[idx]
            current_mask = self.refine_mask_edges_adaptive(current_mask)
            self._update_mask_with_binary(modified_masks, idx, current_mask)

            if np.sum(current_mask > 0.5) >= 140:
                final_keep_indices.append(idx)

        if len(final_keep_indices) == 0:
            return {
                'boxes': np.array([]),
                'masks': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }

        final_masks = []
        final_boxes = []
        final_scores = []
        final_labels = []

        for idx in final_keep_indices:
            current_mask = modified_masks[idx][0] if len(modified_masks[idx].shape) == 3 else modified_masks[idx]
            ys, xs = np.where(current_mask > 0.5)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            final_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            final_masks.append(np.expand_dims(current_mask.astype(np.float32), axis=0))
            final_scores.append(scores[idx])
            final_labels.append(labels[idx])

        if len(final_masks) == 0:
            return {
                'boxes': np.array([]),
                'masks': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }

        return {
            'boxes': np.array(final_boxes, dtype=np.float32),
            'masks': np.array(final_masks, dtype=np.float32),
            'scores': np.array(final_scores, dtype=np.float32),
            'labels': np.array(final_labels, dtype=np.int64)
        }

    def _update_mask_with_binary(self, modified_masks, idx, new_binary_mask):
        original_shape = modified_masks[idx][0].shape if len(modified_masks[idx].shape) == 3 else modified_masks[idx].shape

        if new_binary_mask.shape != original_shape:
            new_mask = cv2.resize(new_binary_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            new_mask = new_binary_mask

        if len(modified_masks[idx].shape) == 3:
            modified_masks[idx][0] = new_mask
        else:
            modified_masks[idx] = new_mask

    def process_results(
        self,
        results,
        score_threshold=0.6,
        remove_overlaps=True,
        use_conflict_resolution=True,
        selected_coin_mask=None,
        exclude_coin_region=True,
        remove_misclassified_coin=True,
        fix_fragmented_masks=True,
        resolve_boundary_conflicts=True,
        split_touching_instances=True,
        large_mask_area_threshold=1800,
        min_split_area=140,
        max_overlap_ratio=0.03
    ):
        processed_results = results.copy()

        if remove_misclassified_coin and selected_coin_mask is not None:
            processed_results = self.find_and_remove_coin_misclassified_as_rock(processed_results, selected_coin_mask)

        if exclude_coin_region and selected_coin_mask is not None:
            processed_results = self.exclude_selected_coin(processed_results, selected_coin_mask)

        if use_conflict_resolution:
            processed_results = self.resolve_mask_conflicts(processed_results, score_threshold)
        elif remove_overlaps:
            processed_results = self.remove_overlapping_detections(processed_results, score_threshold)

        if fix_fragmented_masks:
            processed_results = self.fix_fragmented_masks(processed_results)

        if resolve_boundary_conflicts:
            processed_results = self.resolve_mask_boundary_conflicts(processed_results, overlap_threshold=max_overlap_ratio)

        if split_touching_instances:
            print("尝试拆分粘连的大mask...")
            processed_results = self.split_large_masks_in_results(
                processed_results,
                score_threshold=score_threshold,
                area_threshold=large_mask_area_threshold,
                min_split_area=min_split_area
            )

        processed_results = self.refine_results_masks(
            processed_results,
            min_area=min_split_area
        )

        return processed_results