"""
Enhanced Memory Bank Manager v3 for Long-term and Short-term Memory
Author: Enhanced SAMWISE v3
Key improvements over v2:
- Feature-level similarity computation (not just mask IoU)
- Dynamic threshold adjustment based on video difficulty
- Importance scoring mechanism for memory frames
- Enhanced key-frame detection
- Smart memory retrieval with relevance-based selection
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class MemoryBankManagerV3:
    """v3: Enhanced memory bank with feature-level similarity and dynamic thresholds"""

    def __init__(self, args):
        self.use_dual_memory = args.use_dual_memory
        self.short_term_capacity = args.short_term_capacity
        self.long_term_capacity = args.long_term_capacity
        self.object_score_threshold = args.object_score_threshold
        self.iou_threshold = args.iou_threshold
        self.mask_consistency_threshold = args.mask_consistency_threshold
        self.similarity_threshold = args.similarity_threshold
        self.frame_sampling_interval = args.frame_sampling_interval

        # Enhanced parameters
        self.temporal_decay_factor = getattr(args, 'temporal_decay_factor', 0.95)
        self.quality_score_threshold = getattr(args, 'quality_score_threshold', 0.5)
        self.enable_adaptive_sampling = getattr(args, 'enable_adaptive_sampling', True)

        # v3 new parameters
        self.feature_similarity_weight = getattr(args, 'feature_similarity_weight', 0.5)
        self.enable_dynamic_thresholds = getattr(args, 'enable_dynamic_thresholds', True)
        self.importance_score_threshold = getattr(args, 'importance_score_threshold', 0.6)

        # Weights for quality scoring
        self.w_object_score = 0.3
        self.w_iou_score = 0.4
        self.w_consistency = 0.3

        # Memory banks with metadata
        self.long_term_memory = {}
        self.short_term_memory = {}
        self.memory_metadata = {}

        # Frame change tracking
        self.frame_change_history = []
        self.prev_mask = None
        self.prev_features = None

        # Dynamic threshold tracking
        self.difficulty_history = []
        self.adaptive_iou_threshold = self.iou_threshold
        self.adaptive_similarity_threshold = self.similarity_threshold

    def reset(self):
        """Reset both memory banks and metadata"""
        self.long_term_memory = {}
        self.short_term_memory = {}
        self.memory_metadata = {}
        self.frame_change_history = []
        self.prev_mask = None
        self.prev_features = None
        self.difficulty_history = []
        self.adaptive_iou_threshold = self.iou_threshold
        self.adaptive_similarity_threshold = self.similarity_threshold

    def compute_quality_score(self, decoder_out) -> Tuple[float, Dict[str, float]]:
        """Compute comprehensive quality score with enhanced metrics"""
        metrics = {}

        # Metric 1: Object score
        object_score = torch.sigmoid(decoder_out.object_score_logits).item()
        metrics['object_score'] = object_score

        # Metric 2: IoU score
        iou_score = decoder_out.ious.max().item()
        metrics['iou_score'] = iou_score

        # Metric 3: Multimask consistency
        is_consistent, consistency_score = self.check_multimask_consistency(decoder_out)
        metrics['consistency_score'] = consistency_score
        metrics['is_consistent'] = is_consistent

        # Weighted quality score
        quality = (
            self.w_object_score * object_score +
            self.w_iou_score * iou_score +
            self.w_consistency * consistency_score
        )
        metrics['quality_score'] = quality

        return quality, metrics

    def compute_importance_score(self, frame_idx: int, quality_score: float,
                                 is_extreme: bool, novelty_score: float) -> float:
        """
        v3: Compute comprehensive importance score for a frame.

        Args:
            frame_idx: Frame index
            quality_score: Quality score from quality check
            is_extreme: Whether this is an extreme case (inconsistent masks)
            novelty_score: How novel/different this frame is from existing memory

        Returns:
            importance_score: 0-1 score indicating frame importance
        """
        # Factor 1: Quality (30%)
        quality_factor = quality_score

        # Factor 2: Extreme case bonus (25%)
        extreme_factor = 1.0 if is_extreme else 0.3

        # Factor 3: Novelty (35%)
        novelty_factor = novelty_score

        # Factor 4: Temporal coverage (10%)
        # Frames that fill temporal gaps are more important
        temporal_factor = self._compute_temporal_coverage_value(frame_idx)

        importance = (
            0.30 * quality_factor +
            0.25 * extreme_factor +
            0.35 * novelty_factor +
            0.10 * temporal_factor
        )

        return importance

    def _compute_temporal_coverage_value(self, frame_idx: int) -> float:
        """Compute how much this frame improves temporal coverage"""
        if len(self.long_term_memory) == 0:
            return 1.0

        lt_indices = sorted(self.long_term_memory.keys())

        # Find nearest neighbors
        distances = [abs(frame_idx - idx) for idx in lt_indices]
        min_dist = min(distances) if distances else float('inf')

        # Normalize: larger gap = higher value
        coverage_value = min(1.0, min_dist / (self.frame_sampling_interval * 3))

        return coverage_value

    def calculate_feature_similarity(self, current_features, stored_features) -> float:
        """
        v3: Calculate feature-level cosine similarity.

        Args:
            current_features: Features from current frame
            stored_features: Features from stored frame

        Returns:
            similarity: Cosine similarity score
        """
        if current_features is None or stored_features is None:
            return 0.0

        # Flatten features for cosine similarity
        curr_flat = current_features.flatten()
        stored_flat = stored_features.flatten()

        # Cosine similarity
        similarity = F.cosine_similarity(
            curr_flat.unsqueeze(0),
            stored_flat.unsqueeze(0),
            dim=1
        ).item()

        return max(0.0, similarity)  # Ensure non-negative

    def calculate_combined_similarity(self, current_mask, current_features,
                                     frame_idx: int) -> Tuple[float, Optional[int], List[Tuple[int, float]]]:
        """
        v3: Calculate combined similarity using both mask IoU and feature similarity.

        Returns:
            (max_similarity, most_similar_idx, all_similarities)
        """
        if len(self.long_term_memory) == 0:
            return 0.0, None, []

        current_mask_binary = (torch.sigmoid(current_mask) > 0.5).float()

        all_similarities = []

        for mem_idx, mem_dict in self.long_term_memory.items():
            stored_mask = mem_dict['pred_masks']
            stored_mask_binary = (torch.sigmoid(stored_mask) > 0.5).float()

            # Spatial IoU
            spatial_iou = self._calculate_mask_iou(
                current_mask_binary[0, 0],
                stored_mask_binary[0, 0]
            ).item()

            # Feature similarity
            stored_features = mem_dict.get('object_features', None)
            feature_sim = self.calculate_feature_similarity(current_features, stored_features)

            # Combined similarity (mask IoU + feature similarity)
            combined_sim = (
                (1 - self.feature_similarity_weight) * spatial_iou +
                self.feature_similarity_weight * feature_sim
            )

            # Temporal decay
            time_diff = abs(frame_idx - mem_idx)
            temporal_weight = math.exp(-time_diff / (self.long_term_capacity * 2))

            # Final similarity with temporal decay
            similarity = combined_sim * (0.7 + 0.3 * temporal_weight)

            all_similarities.append((mem_idx, similarity))

        # Find maximum similarity
        if all_similarities:
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            most_similar_idx, max_similarity = all_similarities[0]
        else:
            max_similarity, most_similar_idx = 0.0, None

        # Update video difficulty estimation
        self._update_difficulty_estimation(max_similarity)

        return max_similarity, most_similar_idx, all_similarities

    def _update_difficulty_estimation(self, similarity_score: float):
        """
        v3: Update video difficulty estimation based on similarity patterns.
        Used for dynamic threshold adjustment.
        """
        self.difficulty_history.append(similarity_score)

        # Keep last 30 frames
        if len(self.difficulty_history) > 30:
            self.difficulty_history.pop(0)

        if self.enable_dynamic_thresholds and len(self.difficulty_history) >= 10:
            # Compute average similarity
            avg_similarity = sum(self.difficulty_history) / len(self.difficulty_history)

            # If average similarity is low, video is difficult -> lower thresholds
            # If average similarity is high, video is easy -> raise thresholds

            # Adaptive IoU threshold (0.3 to 0.7 range)
            self.adaptive_iou_threshold = self.iou_threshold * (0.6 + 0.4 * avg_similarity)
            self.adaptive_iou_threshold = max(0.3, min(0.7, self.adaptive_iou_threshold))

            # Adaptive similarity threshold (0.6 to 0.9 range)
            self.adaptive_similarity_threshold = self.similarity_threshold * (0.8 + 0.2 * avg_similarity)
            self.adaptive_similarity_threshold = max(0.6, min(0.9, self.adaptive_similarity_threshold))

    def check_quality_thresholds(self, decoder_out) -> Tuple[bool, str, Dict]:
        """Enhanced quality check with dynamic thresholds"""
        quality_score, metrics = self.compute_quality_score(decoder_out)

        # Use adaptive threshold if enabled
        iou_threshold = self.adaptive_iou_threshold if self.enable_dynamic_thresholds else self.iou_threshold

        # Step 1: Check object score
        if metrics['object_score'] < self.object_score_threshold:
            return False, f"object_score={metrics['object_score']:.3f} < {self.object_score_threshold}", metrics

        # Step 2: Check IoU score (with dynamic threshold)
        if metrics['iou_score'] < iou_threshold:
            return False, f"iou={metrics['iou_score']:.3f} < {iou_threshold:.3f} (adaptive)", metrics

        # Check overall quality score
        if quality_score < self.quality_score_threshold:
            return False, f"quality={quality_score:.3f} < {self.quality_score_threshold}", metrics

        return True, "passed", metrics

    def check_multimask_consistency(self, decoder_out) -> Tuple[bool, float]:
        """Enhanced multimask consistency check"""
        if decoder_out.high_res_multimasks is None:
            return True, 1.0

        multimasks = torch.sigmoid(decoder_out.high_res_multimasks[0])
        binary_masks = (multimasks > 0.5).float()

        # Calculate pairwise IoU
        iou_01 = self._calculate_mask_iou(binary_masks[0], binary_masks[1])
        iou_02 = self._calculate_mask_iou(binary_masks[0], binary_masks[2])
        iou_12 = self._calculate_mask_iou(binary_masks[1], binary_masks[2])

        # Weighted average favoring best mask
        avg_consistency = (2.0 * iou_01 + 2.0 * iou_02 + 1.0 * iou_12) / 5.0

        is_consistent = avg_consistency >= self.mask_consistency_threshold

        return is_consistent, avg_consistency.item()

    def should_apply_filtering(self, frame_idx: int, decoder_out=None) -> bool:
        """Enhanced adaptive frame sampling"""
        if not self.use_dual_memory:
            return False

        # Base case: always filter at sampling interval
        if frame_idx % self.frame_sampling_interval == 0:
            return True

        # Adaptive sampling
        if self.enable_adaptive_sampling and decoder_out is not None:
            change = self.update_frame_change_tracking(decoder_out.low_res_masks)

            if len(self.frame_change_history) >= 5:
                recent_avg_change = sum(self.frame_change_history[-5:]) / 5.0

                # Dynamic threshold based on difficulty
                change_threshold = 0.1
                if len(self.difficulty_history) >= 5:
                    avg_difficulty = 1.0 - sum(self.difficulty_history[-5:]) / 5.0
                    change_threshold = 0.05 + 0.15 * avg_difficulty  # 0.05 to 0.20

                if recent_avg_change > change_threshold:
                    return frame_idx % max(1, self.frame_sampling_interval // 2) == 0

        return False

    def update_frame_change_tracking(self, current_mask):
        """Track frame-to-frame changes"""
        if self.prev_mask is None:
            self.prev_mask = current_mask
            return 0.0

        curr_binary = (torch.sigmoid(current_mask) > 0.5).float()
        prev_binary = (torch.sigmoid(self.prev_mask) > 0.5).float()

        change = torch.abs(curr_binary - prev_binary).mean().item()

        self.frame_change_history.append(change)

        if len(self.frame_change_history) > 30:
            self.frame_change_history.pop(0)

        self.prev_mask = current_mask

        return change

    def store_to_memory(self, frame_idx: int, mem_dict: Dict,
                        decoder_out, object_features=None, force_long_term: bool = False):
        """
        v3: Enhanced storage with importance scoring and feature-level similarity.
        """
        if not self.use_dual_memory:
            return mem_dict

        # Store object features for similarity computation
        if object_features is not None:
            mem_dict['object_features'] = object_features.detach() if torch.is_tensor(object_features) else object_features

        # Check if filtering should be applied
        if not self.should_apply_filtering(frame_idx, decoder_out):
            self._store_to_short_term(frame_idx, mem_dict, quality_score=0.0)
            return mem_dict

        # Quality check
        passes_quality, reason, metrics = self.check_quality_thresholds(decoder_out)

        if not passes_quality:
            quality_score = metrics['quality_score']

            if quality_score < 0.3:
                print(f"Frame {frame_idx} DISCARDED: {reason}")
                return mem_dict
            else:
                print(f"Frame {frame_idx} -> SHORT-TERM (low quality): {reason}")
                self._store_to_short_term(frame_idx, mem_dict, quality_score)
                return mem_dict

        # High quality frame
        quality_score = metrics['quality_score']
        is_consistent, consistency_score = metrics['is_consistent'], metrics['consistency_score']

        # Check if extreme case
        is_extreme = not is_consistent or force_long_term

        if is_extreme:
            # Extreme case -> long-term memory
            importance = self.compute_importance_score(frame_idx, quality_score, True, 1.0)
            print(f"Frame {frame_idx} -> LONG-TERM [EXTREME] (cons={consistency_score:.3f}, imp={importance:.3f})")
            self._store_to_long_term_smart(frame_idx, mem_dict, quality_score, importance, force=True)
            return mem_dict

        # Calculate combined similarity (mask + features)
        max_similarity, most_similar_idx, all_similarities = self.calculate_combined_similarity(
            decoder_out.low_res_masks,
            object_features,
            frame_idx
        )

        # Use adaptive threshold if enabled
        similarity_threshold = self.adaptive_similarity_threshold if self.enable_dynamic_thresholds else self.similarity_threshold

        # Compute novelty score (inverse of similarity)
        novelty_score = 1.0 - max_similarity

        # Compute importance
        importance = self.compute_importance_score(frame_idx, quality_score, False, novelty_score)

        if max_similarity > similarity_threshold:
            # High similarity -> short-term
            print(f"Frame {frame_idx} -> SHORT-TERM (sim={max_similarity:.3f} > {similarity_threshold:.3f})")
            self._store_to_short_term(frame_idx, mem_dict, quality_score)
        else:
            # Low similarity, high importance -> long-term
            if importance >= self.importance_score_threshold:
                print(f"Frame {frame_idx} -> LONG-TERM [IMPORTANT] (sim={max_similarity:.3f}, imp={importance:.3f})")
                self._store_to_long_term_smart(frame_idx, mem_dict, quality_score, importance,
                                              all_similarities=all_similarities)
            else:
                print(f"Frame {frame_idx} -> SHORT-TERM (low importance={importance:.3f})")
                self._store_to_short_term(frame_idx, mem_dict, quality_score)

        return mem_dict

    def _store_to_short_term(self, frame_idx: int, mem_dict: Dict, quality_score: float):
        """Store to short-term with FIFO"""
        self.short_term_memory[frame_idx] = mem_dict
        self.memory_metadata[frame_idx] = {
            'memory_type': 'short_term',
            'quality_score': quality_score,
            'timestamp': frame_idx
        }

        if len(self.short_term_memory) > self.short_term_capacity:
            oldest_idx = min(self.short_term_memory.keys())
            del self.short_term_memory[oldest_idx]
            if oldest_idx in self.memory_metadata:
                del self.memory_metadata[oldest_idx]

    def _store_to_long_term_smart(self, frame_idx: int, mem_dict: Dict, quality_score: float,
                                   importance_score: float = 1.0,
                                   all_similarities: Optional[List[Tuple[int, float]]] = None,
                                   force: bool = False):
        """
        v3: Smart storage with importance-based replacement.
        """
        self.long_term_memory[frame_idx] = mem_dict
        self.memory_metadata[frame_idx] = {
            'memory_type': 'long_term',
            'quality_score': quality_score,
            'importance_score': importance_score,
            'timestamp': frame_idx
        }

        # Handle capacity with importance-based removal
        if len(self.long_term_memory) > self.long_term_capacity:
            frame_to_remove = self._find_least_important_frame(frame_idx, all_similarities)

            if frame_to_remove is not None and frame_to_remove != frame_idx:
                removed_importance = self.memory_metadata.get(frame_to_remove, {}).get('importance_score', 0.0)
                print(f"  Removing frame {frame_to_remove} (imp={removed_importance:.3f}) from LONG-TERM")
                del self.long_term_memory[frame_to_remove]
                if frame_to_remove in self.memory_metadata:
                    del self.memory_metadata[frame_to_remove]

    def _find_least_important_frame(self, new_frame_idx: int,
                                    all_similarities: Optional[List[Tuple[int, float]]]) -> Optional[int]:
        """
        v3: Find least important frame for removal based on importance score.
        """
        if len(self.long_term_memory) < self.long_term_capacity:
            return None

        frame_indices = list(self.long_term_memory.keys())

        # Compute removal cost for each frame (lower = more likely to remove)
        removal_costs = {}

        for frame_idx in frame_indices:
            if frame_idx == new_frame_idx:
                continue

            metadata = self.memory_metadata.get(frame_idx, {})

            # Importance score (higher importance = higher cost to remove)
            importance = metadata.get('importance_score', 0.5)

            # Quality score
            quality = metadata.get('quality_score', 0.5)

            # Temporal position (keep boundary frames)
            position = frame_indices.index(frame_idx)
            is_boundary = (position == 0 or position == len(frame_indices) - 1)
            boundary_cost = 1.0 if is_boundary else 0.3

            # Combined cost (lower = easier to remove)
            cost = (
                0.50 * importance +      # Importance is key
                0.30 * quality +          # Quality matters
                0.20 * boundary_cost      # Preserve boundaries
            )

            removal_costs[frame_idx] = cost

        # Remove frame with lowest cost (least important)
        if removal_costs:
            frame_to_remove = min(removal_costs.keys(), key=lambda k: removal_costs[k])
            return frame_to_remove

        return None

    @staticmethod
    def _calculate_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two binary masks"""
        intersection = (mask1 * mask2).sum()
        union = (mask1 + mask2).clamp(0, 1).sum()

        if union == 0:
            return torch.tensor(0.0)

        iou = intersection / union
        return iou

    def get_memory_bank_for_retrieval(self) -> Dict:
        """Get combined memory bank for retrieval"""
        if not self.use_dual_memory:
            return {}

        combined = {}
        combined.update(self.long_term_memory)
        combined.update(self.short_term_memory)
        return combined

    def compute_temporal_diversity(self) -> float:
        """Compute temporal diversity of long-term memory"""
        if len(self.long_term_memory) <= 1:
            return 1.0

        frame_indices = sorted(self.long_term_memory.keys())
        gaps = [frame_indices[i+1] - frame_indices[i] for i in range(len(frame_indices)-1)]

        if not gaps:
            return 1.0

        avg_gap = sum(gaps) / len(gaps)
        gap_variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)

        diversity = 1.0 / (1.0 + gap_variance / (avg_gap + 1e-6))

        return diversity

    def print_memory_status(self):
        """Print enhanced memory status"""
        if self.use_dual_memory:
            lt_qualities = [self.memory_metadata.get(idx, {}).get('quality_score', 0.0)
                           for idx in self.long_term_memory.keys()]
            st_qualities = [self.memory_metadata.get(idx, {}).get('quality_score', 0.0)
                           for idx in self.short_term_memory.keys()]
            lt_importances = [self.memory_metadata.get(idx, {}).get('importance_score', 0.0)
                             for idx in self.long_term_memory.keys()]

            avg_lt_quality = sum(lt_qualities) / len(lt_qualities) if lt_qualities else 0.0
            avg_st_quality = sum(st_qualities) / len(st_qualities) if st_qualities else 0.0
            avg_lt_importance = sum(lt_importances) / len(lt_importances) if lt_importances else 0.0

            diversity = self.compute_temporal_diversity()

            print(f"Memory Status (v3):")
            print(f"  Short-term: {len(self.short_term_memory)}/{self.short_term_capacity} "
                  f"(quality={avg_st_quality:.3f})")
            print(f"  Long-term: {len(self.long_term_memory)}/{self.long_term_capacity} "
                  f"(quality={avg_lt_quality:.3f}, importance={avg_lt_importance:.3f}, diversity={diversity:.3f})")
            if self.enable_dynamic_thresholds:
                print(f"  Adaptive thresholds: IoU={self.adaptive_iou_threshold:.3f}, "
                      f"Similarity={self.adaptive_similarity_threshold:.3f}")

    def get_memory_dict(self, frame_idx: int) -> Optional[Dict]:
        """Retrieve memory for specific frame"""
        if not self.use_dual_memory:
            return None

        if frame_idx in self.short_term_memory:
            return self.short_term_memory[frame_idx]

        if frame_idx in self.long_term_memory:
            return self.long_term_memory[frame_idx]

        return None

    def get_all_memory_indices(self):
        """Get all frame indices from both memory banks"""
        short_term_keys = set(self.short_term_memory.keys())
        long_term_keys = set(self.long_term_memory.keys())
        return short_term_keys.union(long_term_keys)
