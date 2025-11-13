"""
Unit tests for Dual Memory Bank functionality
Run: python -m pytest tests/test_dual_memory_bank.py -v
"""

import torch
import sys
sys.path.append('..')
from argparse import Namespace
from models.memory_bank_manager import MemoryBankManager
from models.model_utils import DecoderOutput


def create_mock_args(use_dual_memory=True):
    """Create mock arguments for testing"""
    return Namespace(
        use_dual_memory=use_dual_memory,
        short_term_capacity=5,
        long_term_capacity=10,
        object_score_threshold=0.0,
        iou_threshold=0.7,
        mask_consistency_threshold=0.8,
        similarity_threshold=0.85,
        frame_sampling_interval=3
    )


def create_mock_decoder_output(
    object_score=0.9,
    iou=0.8,
    mask_shape=(1, 1, 256, 256),
    multimask_shape=(1, 3, 256, 256)
):
    """Create mock DecoderOutput for testing"""
    decoder_out = DecoderOutput()
    decoder_out.object_score_logits = torch.logit(torch.tensor([object_score]))
    decoder_out.ious = torch.tensor([[iou, iou-0.1, iou-0.2]])
    decoder_out.low_res_masks = torch.randn(*mask_shape)
    decoder_out.high_res_masks = torch.randn(*mask_shape)
    decoder_out.high_res_multimasks = torch.randn(*multimask_shape)
    return decoder_out


def test_memory_bank_initialization():
    """Test MemoryBankManager initialization"""
    print("\n=== Test 1: Initialization ===")
    args = create_mock_args()
    manager = MemoryBankManager(args)

    assert manager.use_dual_memory == True
    assert manager.short_term_capacity == 5
    assert manager.long_term_capacity == 10
    assert len(manager.short_term_memory) == 0
    assert len(manager.long_term_memory) == 0
    print("✓ Initialization test passed")


def test_frame_sampling():
    """Test frame sampling strategy"""
    print("\n=== Test 2: Frame Sampling ===")
    args = create_mock_args()
    manager = MemoryBankManager(args)

    # With interval=3, should filter on frames 0, 3, 6, 9...
    assert manager.should_apply_filtering(0) == True
    assert manager.should_apply_filtering(1) == False
    assert manager.should_apply_filtering(2) == False
    assert manager.should_apply_filtering(3) == True
    assert manager.should_apply_filtering(6) == True
    print("✓ Frame sampling test passed")


def test_quality_thresholds():
    """Test quality threshold checks"""
    print("\n=== Test 3: Quality Thresholds ===")
    args = create_mock_args()
    manager = MemoryBankManager(args)

    # Case 1: High quality - should pass
    decoder_out = create_mock_decoder_output(object_score=0.9, iou=0.85)
    passed, reason = manager.check_quality_thresholds(decoder_out)
    assert passed == True
    print(f"✓ High quality: {reason}")

    # Case 2: Low IoU - should fail
    decoder_out = create_mock_decoder_output(object_score=0.9, iou=0.5)
    passed, reason = manager.check_quality_thresholds(decoder_out)
    assert passed == False
    print(f"✓ Low IoU detected: {reason}")

    # Case 3: Low object score - should fail
    decoder_out = create_mock_decoder_output(object_score=0.3, iou=0.85)
    args_strict = create_mock_args()
    args_strict.object_score_threshold = 0.5
    manager_strict = MemoryBankManager(args_strict)
    passed, reason = manager_strict.check_quality_thresholds(decoder_out)
    assert passed == False
    print(f"✓ Low object score detected: {reason}")


def test_multimask_consistency():
    """Test multimask consistency check"""
    print("\n=== Test 4: Multimask Consistency ===")
    args = create_mock_args()
    manager = MemoryBankManager(args)

    # Case 1: Consistent masks (all similar)
    decoder_out = DecoderOutput()
    consistent_mask = torch.ones(1, 1, 64, 64)
    # Create 3 very similar masks
    mask1 = consistent_mask.clone()
    mask2 = consistent_mask.clone() * 0.95
    mask3 = consistent_mask.clone() * 0.98
    decoder_out.high_res_multimasks = torch.cat([mask1, mask2, mask3], dim=1)

    is_consistent, score = manager.check_multimask_consistency(decoder_out)
    print(f"  Consistent masks score: {score:.3f}")
    assert is_consistent == True

    # Case 2: Inconsistent masks (very different)
    mask1 = torch.ones(1, 1, 64, 64)
    mask1[:, :, :32, :] = 0  # Top half zero
    mask2 = torch.ones(1, 1, 64, 64)
    mask2[:, :, :, :32] = 0  # Left half zero
    mask3 = torch.ones(1, 1, 64, 64)
    mask3[:, :, 32:, :] = 0  # Bottom half zero
    decoder_out.high_res_multimasks = torch.cat([mask1, mask2, mask3], dim=1)

    is_consistent, score = manager.check_multimask_consistency(decoder_out)
    print(f"  Inconsistent masks score: {score:.3f}")
    assert is_consistent == False
    print("✓ Multimask consistency test passed")


def test_short_term_fifo():
    """Test short-term memory FIFO behavior"""
    print("\n=== Test 5: Short-term FIFO ===")
    args = create_mock_args()
    manager = MemoryBankManager(args)

    # Add 7 frames (capacity is 5)
    for i in range(7):
        mem_dict = {'frame': i, 'pred_masks': torch.randn(1, 1, 64, 64)}
        manager._store_to_short_term(i, mem_dict)

    # Should only have the last 5 frames
    assert len(manager.short_term_memory) == 5
    assert 0 not in manager.short_term_memory  # Frame 0 evicted
    assert 1 not in manager.short_term_memory  # Frame 1 evicted
    assert 6 in manager.short_term_memory      # Frame 6 present
    print(f"  Short-term memory contains frames: {list(manager.short_term_memory.keys())}")
    print("✓ Short-term FIFO test passed")


def test_similarity_calculation():
    """Test similarity calculation with long-term memory"""
    print("\n=== Test 6: Similarity Calculation ===")
    args = create_mock_args()
    manager = MemoryBankManager(args)

    # Add some frames to long-term memory
    for i in range(3):
        mask = torch.ones(1, 1, 64, 64) * (i + 1) / 4  # Different intensities
        mem_dict = {'pred_masks': mask}
        manager.long_term_memory[i] = mem_dict

    # Test similarity with a mask similar to frame 1
    test_mask = torch.ones(1, 1, 64, 64) * 0.5 + torch.randn(1, 1, 64, 64) * 0.1
    max_sim, most_similar_idx = manager.calculate_similarity_with_long_term(test_mask)

    print(f"  Most similar frame: {most_similar_idx}, similarity: {max_sim:.3f}")
    assert most_similar_idx is not None
    assert 0.0 <= max_sim <= 1.0
    print("✓ Similarity calculation test passed")


def test_full_pipeline():
    """Test full storage pipeline"""
    print("\n=== Test 7: Full Storage Pipeline ===")
    args = create_mock_args()
    manager = MemoryBankManager(args)

    # Simulate processing 10 frames
    for frame_idx in range(10):
        # Create decoder output with varying quality
        if frame_idx % 3 == 0:  # Filtering frames
            # High quality, inconsistent masks (extreme case)
            is_consistent = (frame_idx % 6 != 0)
            decoder_out = create_mock_decoder_output(
                object_score=0.95,
                iou=0.85
            )
            # Make masks inconsistent for frames 0, 6
            if not is_consistent:
                mask1 = torch.ones(1, 1, 64, 64)
                mask2 = torch.zeros(1, 1, 64, 64)
                mask3 = torch.randn(1, 1, 64, 64)
                decoder_out.high_res_multimasks = torch.cat([mask1, mask2, mask3], dim=1)
        else:
            decoder_out = create_mock_decoder_output()

        mem_dict = {
            'frame': frame_idx,
            'pred_masks': decoder_out.low_res_masks,
            'obj_ptr': torch.randn(1, 256)
        }

        manager.store_to_memory(frame_idx, mem_dict, decoder_out)

    # Check results
    manager.print_memory_status()
    combined = manager.get_memory_bank_for_retrieval()
    print(f"  Total frames in combined memory: {len(combined)}")
    print(f"  Short-term frames: {list(manager.short_term_memory.keys())}")
    print(f"  Long-term frames: {list(manager.long_term_memory.keys())}")
    print("✓ Full pipeline test passed")


def test_backward_compatibility():
    """Test backward compatibility when dual memory is disabled"""
    print("\n=== Test 8: Backward Compatibility ===")
    args = create_mock_args(use_dual_memory=False)
    manager = MemoryBankManager(args)

    assert manager.use_dual_memory == False
    assert manager.should_apply_filtering(0) == False
    assert manager.should_apply_filtering(3) == False

    # Store should not use dual memory logic
    decoder_out = create_mock_decoder_output()
    mem_dict = {'frame': 0}
    result = manager.store_to_memory(0, mem_dict, decoder_out)
    assert result == mem_dict
    print("✓ Backward compatibility test passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Dual Memory Bank Unit Tests")
    print("=" * 60)

    try:
        test_memory_bank_initialization()
        test_frame_sampling()
        test_quality_thresholds()
        test_multimask_consistency()
        test_short_term_fifo()
        test_similarity_calculation()
        test_full_pipeline()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
