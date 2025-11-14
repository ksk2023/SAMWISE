"""
Enhanced Memory Bank Manager for Long-term and Short-term Memory
Author: Enhanced SAMWISE v2
Key improvements:
- Quality scoring system with weighted metrics
- Temporal diversity-aware long-term memory management
- Enhanced similarity calculation with temporal decay
- Adaptive frame sampling based on content change
- Smarter memory replacement strategies
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class MemoryBankManager:
    """Manages long-term and short-term memory banks with enhanced filtering mechanisms"""

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

        # Weights for quality scoring
        self.w_object_score = 0.3
        self.w_iou_score = 0.4
        self.w_consistency = 0.3

        # Memory banks with metadata
        self.long_term_memory = {}
        self.short_term_memory = {}
        self.memory_metadata = {}  # Store quality scores and timestamps

        # Frame change tracking for adaptive sampling
        self.frame_change_history = []
        self.prev_mask = None

    def reset(self):
        """Reset both memory banks and metadata"""
        self.long_term_memory = {}
        self.short_term_memory = {}
        self.memory_metadata = {}
        self.frame_change_history = []
        self.prev_mask = None

    def compute_quality_score(self, decoder_out) -> Tuple[float, Dict[str, float]]:
        """
        Compute comprehensive quality score combining multiple metrics.

        Returns:
            (quality_score, metrics_dict) - Overall quality and individual metrics
        """
        metrics = {}

        # Metric 1: Object score (occlusion/existence confidence)
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

    def check_quality_thresholds(self, decoder_out) -> Tuple[bool, str, Dict]:
        """
        Step 1 & 2: Check object score and IoU thresholds with detailed metrics.

        Returns:
            (pass_check, reason, metrics) - True if passes, False otherwise with reason and metrics
        """
        # Get comprehensive quality metrics
        quality_score, metrics = self.compute_quality_score(decoder_out)

        # Step 1: Check object score (occlusion score)
        if metrics['object_score'] < self.object_score_threshold:
            return False, f"object_score={metrics['object_score']:.3f} < {self.object_score_threshold}", metrics

        # Step 2: Check IoU score
        if metrics['iou_score'] < self.iou_threshold:
            return False, f"iou={metrics['iou_score']:.3f} < {self.iou_threshold}", metrics

        # Check overall quality score
        if quality_score < self.quality_score_threshold:
            return False, f"quality={quality_score:.3f} < {self.quality_score_threshold}", metrics

        return True, "passed", metrics

    def check_multimask_consistency(self, decoder_out) -> Tuple[bool, float]:
        """
        Step 3: Check if all 3 masks represent the same object with enhanced checking.

        Returns:
            (is_consistent, consistency_score) - True if consistent, False if extreme case
        """
        if decoder_out.high_res_multimasks is None:
            # No multimask output, assume consistent
            return True, 1.0

        # Get the 3 masks [B, 3, H, W] - working with B=1
        multimasks = torch.sigmoid(decoder_out.high_res_multimasks[0])  # [3, H, W]

        # Convert to binary masks
        binary_masks = (multimasks > 0.5).float()

        # Calculate pairwise IoU between the 3 masks
        iou_01 = self._calculate_mask_iou(binary_masks[0], binary_masks[1])
        iou_02 = self._calculate_mask_iou(binary_masks[0], binary_masks[2])
        iou_12 = self._calculate_mask_iou(binary_masks[1], binary_masks[2])

        # Weighted average: give more weight to IoU with the best mask (mask 0)
        # The best mask (index 0) should be more consistent with others
        avg_consistency = (2.0 * iou_01 + 2.0 * iou_02 + 1.0 * iou_12) / 5.0

        # If average consistency is below threshold, masks are inconsistent
        # This indicates an extreme case (occlusion, deformation, etc.)
        is_consistent = avg_consistency >= self.mask_consistency_threshold

        return is_consistent, avg_consistency.item()

    def calculate_similarity_with_temporal_decay(self, current_mask, current_frame_idx) -> Tuple[float, Optional[int], List[Tuple[int, float]]]:
        """
        Calculate similarity with long-term memory considering temporal decay.
        Recent frames in long-term memory should have higher influence.

        Returns:
            (max_similarity, most_similar_idx, all_similarities)
        """
        if len(self.long_term_memory) == 0:
            return 0.0, None, []

        current_mask_binary = (torch.sigmoid(current_mask) > 0.5).float()

        all_similarities = []

        for frame_idx, mem_dict in self.long_term_memory.items():
            stored_mask = mem_dict['pred_masks']
            stored_mask_binary = (torch.sigmoid(stored_mask) > 0.5).float()

            # Spatial IoU
            spatial_iou = self._calculate_mask_iou(
                current_mask_binary[0, 0],
                stored_mask_binary[0, 0]
            ).item()

            # Temporal decay: more recent frames have higher weight
            time_diff = abs(current_frame_idx - frame_idx)
            temporal_weight = math.exp(-time_diff / (self.long_term_capacity * 2))

            # Combined similarity with temporal decay
            similarity = spatial_iou * (0.7 + 0.3 * temporal_weight)

            all_similarities.append((frame_idx, similarity))

        # Find maximum similarity
        if all_similarities:
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            most_similar_idx, max_similarity = all_similarities[0]
        else:
            max_similarity, most_similar_idx = 0.0, None

        return max_similarity, most_similar_idx, all_similarities

    def compute_temporal_diversity(self) -> float:
        """
        Compute temporal diversity of long-term memory.
        Higher diversity means frames are well-distributed over time.
        """
        if len(self.long_term_memory) <= 1:
            return 1.0

        frame_indices = sorted(self.long_term_memory.keys())

        # Calculate gaps between consecutive frames
        gaps = [frame_indices[i+1] - frame_indices[i] for i in range(len(frame_indices)-1)]

        if not gaps:
            return 1.0

        # Diversity is higher when gaps are more uniform
        avg_gap = sum(gaps) / len(gaps)
        gap_variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)

        # Normalize diversity score (lower variance = higher diversity)
        diversity = 1.0 / (1.0 + gap_variance / (avg_gap + 1e-6))

        return diversity

    def find_redundant_frame_for_removal(self, new_frame_idx: int, all_similarities: List[Tuple[int, float]]) -> Optional[int]:
        """
        Find the most redundant frame to remove from long-term memory.
        Considers both similarity and temporal distribution.

        Args:
            new_frame_idx: Frame index being added
            all_similarities: List of (frame_idx, similarity) tuples

        Returns:
            Frame index to remove, or None if no removal needed
        """
        if len(self.long_term_memory) < self.long_term_capacity:
            return None

        # Strategy: Remove frame that has high similarity AND won't hurt temporal diversity

        # Get frame indices sorted by time
        frame_indices = sorted(self.long_term_memory.keys())

        # Calculate removal cost for each frame
        removal_costs = {}

        for frame_idx in frame_indices:
            # Cost factor 1: Similarity to new frame (higher similarity = lower cost to remove)
            similarity_to_new = next((sim for idx, sim in all_similarities if idx == frame_idx), 0.0)
            similarity_cost = 1.0 - similarity_to_new

            # Cost factor 2: Quality score (higher quality = higher cost to remove)
            quality_cost = self.memory_metadata.get(frame_idx, {}).get('quality_score', 0.5)

            # Cost factor 3: Temporal importance (frames at temporal boundaries are more important)
            position = frame_indices.index(frame_idx)
            is_temporal_boundary = (position == 0 or position == len(frame_indices) - 1)
            temporal_cost = 1.0 if is_temporal_boundary else 0.5

            # Cost factor 4: Time gap importance (frames in large gaps are more important)
            if position > 0 and position < len(frame_indices) - 1:
                gap_before = frame_indices[position] - frame_indices[position-1]
                gap_after = frame_indices[position+1] - frame_indices[position]
                gap_cost = (gap_before + gap_after) / (2.0 * self.frame_sampling_interval)
            else:
                gap_cost = 1.0

            # Combined cost (lower cost = better candidate for removal)
            total_cost = (
                0.4 * similarity_cost +  # High similarity to new = low cost
                0.3 * quality_cost +      # High quality = high cost (don't remove)
                0.2 * temporal_cost +     # Boundary frame = high cost
                0.1 * gap_cost           # Large gap = high cost
            )

            removal_costs[frame_idx] = total_cost

        # Remove frame with lowest cost
        if removal_costs:
            frame_to_remove = min(removal_costs.keys(), key=lambda k: removal_costs[k])
            return frame_to_remove

        return None

    def update_frame_change_tracking(self, current_mask):
        """
        Track frame-to-frame changes for adaptive sampling.
        """
        if self.prev_mask is None:
            self.prev_mask = current_mask
            return 0.0

        # Calculate change between current and previous frame
        curr_binary = (torch.sigmoid(current_mask) > 0.5).float()
        prev_binary = (torch.sigmoid(self.prev_mask) > 0.5).float()

        # Use symmetric difference as change measure
        change = torch.abs(curr_binary - prev_binary).mean().item()

        self.frame_change_history.append(change)

        # Keep only recent history
        if len(self.frame_change_history) > 30:
            self.frame_change_history.pop(0)

        self.prev_mask = current_mask

        return change

    def should_apply_filtering(self, frame_idx: int, decoder_out=None) -> bool:
        """
        Determine if filtering should be applied with adaptive sampling.

        Args:
            frame_idx: Current frame index
            decoder_out: Optional decoder output for adaptive sampling

        Returns:
            True if filtering should be applied
        """
        if not self.use_dual_memory:
            return False

        # Base case: always filter at sampling interval
        if frame_idx % self.frame_sampling_interval == 0:
            return True

        # Adaptive sampling: check if there's significant change
        if self.enable_adaptive_sampling and decoder_out is not None:
            change = self.update_frame_change_tracking(decoder_out.low_res_masks)

            # If recent changes are high, increase sampling rate
            if len(self.frame_change_history) >= 5:
                recent_avg_change = sum(self.frame_change_history[-5:]) / 5.0

                # Threshold for significant change (adaptive)
                if recent_avg_change > 0.1:  # Significant motion/appearance change
                    # Sample more frequently
                    return frame_idx % max(1, self.frame_sampling_interval // 2) == 0

        return False

    def store_to_memory(self, frame_idx: int, mem_dict: Dict,
                        decoder_out, force_long_term: bool = False):
        """
        Enhanced storage to appropriate bank based on improved filtering logic.

        Args:
            frame_idx: Current frame index
            mem_dict: Memory dictionary to store
            decoder_out: Decoder output for quality checks
            force_long_term: Force storage to long-term memory
        """
        if not self.use_dual_memory:
            # Original behavior - store to unified memory
            return mem_dict

        # Check if filtering should be applied (with adaptive sampling)
        if not self.should_apply_filtering(frame_idx, decoder_out):
            # Skip filtering, store directly to short-term
            self._store_to_short_term(frame_idx, mem_dict, quality_score=0.0)
            return mem_dict

        # Apply enhanced filtering pipeline
        # Step 1 & 2: Quality thresholds with comprehensive metrics
        passes_quality, reason, metrics = self.check_quality_thresholds(decoder_out)

        if not passes_quality:
            # Low quality - decide whether to store or discard
            quality_score = metrics['quality_score']

            if quality_score < 0.3:  # Very low quality, discard completely
                print(f"Frame {frame_idx} DISCARDED: {reason}")
                return mem_dict
            else:  # Medium-low quality, store to short-term for continuity
                print(f"Frame {frame_idx} -> SHORT-TERM (low quality): {reason}")
                self._store_to_short_term(frame_idx, mem_dict, quality_score)
                return mem_dict

        # High quality frame - proceed to consistency check
        quality_score = metrics['quality_score']

        # Step 3: Multimask consistency check
        is_consistent, consistency_score = metrics['is_consistent'], metrics['consistency_score']

        if not is_consistent or force_long_term:
            # Extreme case detected - MUST go to long-term memory
            print(f"Frame {frame_idx} -> LONG-TERM [EXTREME CASE] (consistency={consistency_score:.3f}, quality={quality_score:.3f})")
            self._store_to_long_term_smart(frame_idx, mem_dict, quality_score, force=True)
            return mem_dict

        # Masks are consistent - check similarity with long-term memory
        max_similarity, most_similar_idx, all_similarities = self.calculate_similarity_with_temporal_decay(
            decoder_out.low_res_masks, frame_idx
        )

        if max_similarity > self.similarity_threshold:
            # High similarity - frame is redundant with existing long-term memory
            print(f"Frame {frame_idx} -> SHORT-TERM (similar to LT frame {most_similar_idx}, sim={max_similarity:.3f})")
            self._store_to_short_term(frame_idx, mem_dict, quality_score)
        else:
            # Low similarity - new valuable information for long-term
            print(f"Frame {frame_idx} -> LONG-TERM [NEW INFO] (max_sim={max_similarity:.3f}, quality={quality_score:.3f})")
            self._store_to_long_term_smart(frame_idx, mem_dict, quality_score,
                                          all_similarities=all_similarities)

        return mem_dict

    def _store_to_short_term(self, frame_idx: int, mem_dict: Dict, quality_score: float):
        """Store to short-term memory with FIFO eviction"""
        self.short_term_memory[frame_idx] = mem_dict
        self.memory_metadata[frame_idx] = {
            'memory_type': 'short_term',
            'quality_score': quality_score,
            'timestamp': frame_idx
        }

        # FIFO: Remove oldest if exceeding capacity
        if len(self.short_term_memory) > self.short_term_capacity:
            oldest_idx = min(self.short_term_memory.keys())
            del self.short_term_memory[oldest_idx]
            if oldest_idx in self.memory_metadata:
                del self.memory_metadata[oldest_idx]

    def _store_to_long_term_smart(self, frame_idx: int, mem_dict: Dict, quality_score: float,
                                   all_similarities: Optional[List[Tuple[int, float]]] = None,
                                   force: bool = False):
        """
        Smart storage to long-term memory with intelligent capacity management.

        Args:
            frame_idx: Frame index to store
            mem_dict: Memory dictionary
            quality_score: Quality score of the frame
            all_similarities: Similarities to existing frames
            force: Force storage even if at capacity
        """
        # Store the frame
        self.long_term_memory[frame_idx] = mem_dict
        self.memory_metadata[frame_idx] = {
            'memory_type': 'long_term',
            'quality_score': quality_score,
            'timestamp': frame_idx
        }

        # Handle capacity with smart removal
        if len(self.long_term_memory) > self.long_term_capacity:
            if all_similarities is not None:
                # Find best frame to remove based on redundancy and diversity
                frame_to_remove = self.find_redundant_frame_for_removal(frame_idx, all_similarities)
            else:
                # Fallback: remove oldest frame
                frame_to_remove = min(self.long_term_memory.keys())

            if frame_to_remove is not None and frame_to_remove != frame_idx:
                print(f"  Removing frame {frame_to_remove} from LONG-TERM (redundancy management)")
                del self.long_term_memory[frame_to_remove]
                if frame_to_remove in self.memory_metadata:
                    del self.memory_metadata[frame_to_remove]

    @staticmethod
    def _calculate_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between two binary masks.

        Args:
            mask1, mask2: Binary masks of shape [H, W]

        Returns:
            IoU score as tensor
        """
        intersection = (mask1 * mask2).sum()
        union = (mask1 + mask2).clamp(0, 1).sum()

        if union == 0:
            return torch.tensor(0.0)

        iou = intersection / union
        return iou

    def get_memory_bank_for_retrieval(self) -> Dict:
        """
        Get combined memory bank for retrieval.
        Merges both short-term and long-term with proper prioritization.
        """
        if not self.use_dual_memory:
            return {}

        # Merge both memory banks
        # Long-term provides stable anchors, short-term provides temporal continuity
        combined = {}
        combined.update(self.long_term_memory)
        combined.update(self.short_term_memory)  # Short-term takes precedence for duplicates
        return combined

    def print_memory_status(self):
        """Print current memory bank status for debugging"""
        if self.use_dual_memory:
            # Calculate average quality scores
            lt_qualities = [self.memory_metadata.get(idx, {}).get('quality_score', 0.0)
                           for idx in self.long_term_memory.keys()]
            st_qualities = [self.memory_metadata.get(idx, {}).get('quality_score', 0.0)
                           for idx in self.short_term_memory.keys()]

            avg_lt_quality = sum(lt_qualities) / len(lt_qualities) if lt_qualities else 0.0
            avg_st_quality = sum(st_qualities) / len(st_qualities) if st_qualities else 0.0

            diversity = self.compute_temporal_diversity()

            print(f"Memory Status:")
            print(f"  Short-term: {len(self.short_term_memory)}/{self.short_term_capacity} "
                  f"(avg_quality={avg_st_quality:.3f})")
            print(f"  Long-term: {len(self.long_term_memory)}/{self.long_term_capacity} "
                  f"(avg_quality={avg_lt_quality:.3f}, diversity={diversity:.3f})")

    def get_memory_dict(self, frame_idx: int) -> Optional[Dict]:
        """
        Retrieve memory for a specific frame index.
        First check short-term, then long-term.
        """
        if not self.use_dual_memory:
            return None

        # Check short-term first (most recent)
        if frame_idx in self.short_term_memory:
            return self.short_term_memory[frame_idx]

        # Then check long-term
        if frame_idx in self.long_term_memory:
            return self.long_term_memory[frame_idx]

        return None

    def get_all_memory_indices(self):
        """Get all frame indices from both memory banks"""
        short_term_keys = set(self.short_term_memory.keys())
        long_term_keys = set(self.long_term_memory.keys())
        return short_term_keys.union(long_term_keys)
