"""
Memory Bank Manager for Long-term and Short-term Memory
Author: Enhanced SAMWISE
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MemoryBankManager:
    """Manages long-term and short-term memory banks with filtering mechanisms"""

    def __init__(self, args):
        self.use_dual_memory = args.use_dual_memory
        self.short_term_capacity = args.short_term_capacity
        self.long_term_capacity = args.long_term_capacity
        self.object_score_threshold = args.object_score_threshold
        self.iou_threshold = args.iou_threshold
        self.mask_consistency_threshold = args.mask_consistency_threshold
        self.similarity_threshold = args.similarity_threshold
        self.frame_sampling_interval = args.frame_sampling_interval

        # Memory banks
        self.long_term_memory = {}
        self.short_term_memory = {}

    def reset(self):
        """Reset both memory banks"""
        self.long_term_memory = {}
        self.short_term_memory = {}

    def get_memory_dict(self, frame_idx: int) -> Optional[Dict]:
        """
        Retrieve memory for a specific frame index.
        First check short-term, then long-term.
        """
        if not self.use_dual_memory:
            # Fallback to original behavior
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

    def should_apply_filtering(self, frame_idx: int) -> bool:
        """
        Determine if filtering should be applied based on frame sampling strategy.
        Only apply filtering every N frames (e.g., every 3 frames).
        """
        if not self.use_dual_memory:
            return False
        return frame_idx % self.frame_sampling_interval == 0

    def check_quality_thresholds(self, decoder_out) -> Tuple[bool, str]:
        """
        Step 1 & 2: Check object score and IoU thresholds.

        Returns:
            (pass_check, reason) - True if passes, False otherwise with reason
        """
        # Step 1: Check object score (occlusion score)
        object_score = torch.sigmoid(decoder_out.object_score_logits).item()
        if object_score < self.object_score_threshold:
            return False, f"object_score={object_score:.3f} < threshold={self.object_score_threshold}"

        # Step 2: Check IoU score
        iou_score = decoder_out.ious.max().item()
        if iou_score < self.iou_threshold:
            return False, f"iou={iou_score:.3f} < threshold={self.iou_threshold}"

        return True, "passed"

    def check_multimask_consistency(self, decoder_out) -> Tuple[bool, float]:
        """
        Step 3: Check if all 3 masks represent the same object.

        Returns:
            (is_consistent, consistency_score) - True if consistent, False if extreme case
        """
        if decoder_out.high_res_multimasks is None:
            # No multimask output, assume consistent
            return True, 1.0

        # Get the 3 masks [B, 3, H, W] - but we're working with B=1
        multimasks = torch.sigmoid(decoder_out.high_res_multimasks[0])  # [3, H, W]

        # Convert to binary masks
        binary_masks = (multimasks > 0.5).float()

        # Calculate pairwise IoU between the 3 masks
        iou_01 = self._calculate_mask_iou(binary_masks[0], binary_masks[1])
        iou_02 = self._calculate_mask_iou(binary_masks[0], binary_masks[2])
        iou_12 = self._calculate_mask_iou(binary_masks[1], binary_masks[2])

        # Average IoU across all pairs
        avg_consistency = (iou_01 + iou_02 + iou_12) / 3.0

        # If average consistency is below threshold, masks are inconsistent
        # This indicates an extreme case (occlusion, deformation, etc.)
        is_consistent = avg_consistency >= self.mask_consistency_threshold

        return is_consistent, avg_consistency.item()

    def calculate_similarity_with_long_term(self, current_mask) -> Tuple[float, Optional[int]]:
        """
        Calculate similarity between current mask and all long-term memory masks.

        Returns:
            (max_similarity, most_similar_frame_idx)
        """
        if len(self.long_term_memory) == 0:
            return 0.0, None

        current_mask_binary = (torch.sigmoid(current_mask) > 0.5).float()

        max_similarity = 0.0
        most_similar_idx = None

        for frame_idx, mem_dict in self.long_term_memory.items():
            stored_mask = mem_dict['pred_masks']
            stored_mask_binary = (torch.sigmoid(stored_mask) > 0.5).float()

            similarity = self._calculate_mask_iou(
                current_mask_binary[0, 0],
                stored_mask_binary[0, 0]
            ).item()

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_idx = frame_idx

        return max_similarity, most_similar_idx

    def store_to_memory(self, frame_idx: int, mem_dict: Dict,
                        decoder_out, force_long_term: bool = False):
        """
        Store memory to appropriate bank based on filtering logic.

        Args:
            frame_idx: Current frame index
            mem_dict: Memory dictionary to store
            decoder_out: Decoder output for quality checks
            force_long_term: Force storage to long-term memory
        """
        if not self.use_dual_memory:
            # Original behavior - store to unified memory
            return mem_dict

        # Check if filtering should be applied
        if not self.should_apply_filtering(frame_idx):
            # Skip filtering, store directly to short-term
            self._store_to_short_term(frame_idx, mem_dict)
            return mem_dict

        # Apply filtering pipeline
        # Step 1 & 2: Quality thresholds
        passes_quality, reason = self.check_quality_thresholds(decoder_out)
        if not passes_quality:
            # Low quality, discard
            print(f"Frame {frame_idx} discarded: {reason}")
            # Still store to short-term for continuity
            self._store_to_short_term(frame_idx, mem_dict)
            return mem_dict

        # Step 3: Multimask consistency check
        is_consistent, consistency_score = self.check_multimask_consistency(decoder_out)

        if not is_consistent or force_long_term:
            # Extreme case detected or forced - store to long-term
            print(f"Frame {frame_idx} -> LONG-TERM (consistency={consistency_score:.3f})")
            self._store_to_long_term(frame_idx, mem_dict)
            return mem_dict

        # Masks are consistent - check similarity with long-term memory
        max_similarity, most_similar_idx = self.calculate_similarity_with_long_term(
            decoder_out.low_res_masks
        )

        if max_similarity > self.similarity_threshold:
            # High similarity with long-term memory - store to short-term only
            print(f"Frame {frame_idx} -> SHORT-TERM (similarity={max_similarity:.3f} with frame {most_similar_idx})")
            self._store_to_short_term(frame_idx, mem_dict)
        else:
            # Low similarity - new valuable information, store to long-term
            print(f"Frame {frame_idx} -> LONG-TERM (new info, similarity={max_similarity:.3f})")
            self._store_to_long_term(frame_idx, mem_dict)

            # Remove the most similar frame from long-term if at capacity
            if most_similar_idx is not None and len(self.long_term_memory) > self.long_term_capacity:
                print(f"  Removing frame {most_similar_idx} from long-term (most similar)")
                del self.long_term_memory[most_similar_idx]

        return mem_dict

    def _store_to_short_term(self, frame_idx: int, mem_dict: Dict):
        """Store to short-term memory with FIFO eviction"""
        self.short_term_memory[frame_idx] = mem_dict

        # FIFO: Remove oldest if exceeding capacity
        if len(self.short_term_memory) > self.short_term_capacity:
            oldest_idx = min(self.short_term_memory.keys())
            del self.short_term_memory[oldest_idx]

    def _store_to_long_term(self, frame_idx: int, mem_dict: Dict):
        """Store to long-term memory with capacity management"""
        self.long_term_memory[frame_idx] = mem_dict

        # If exceeding capacity, this will be handled by similarity-based removal
        # in the store_to_memory function

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
        This merges both short-term and long-term for backward compatibility.
        """
        if not self.use_dual_memory:
            return {}

        # Merge both memory banks (short-term takes precedence for duplicate keys)
        combined = {}
        combined.update(self.long_term_memory)
        combined.update(self.short_term_memory)
        return combined

    def print_memory_status(self):
        """Print current memory bank status for debugging"""
        if self.use_dual_memory:
            print(f"Memory Status: Short-term={len(self.short_term_memory)}/{self.short_term_capacity}, "
                  f"Long-term={len(self.long_term_memory)}/{self.long_term_capacity}")
