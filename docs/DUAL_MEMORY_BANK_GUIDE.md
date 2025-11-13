# Long-term and Short-term Memory Bank Guide

## 概述

这个增强版本的SAMWISE引入了一个创新的**双记忆库系统**（Long-term和Short-term Memory Bank），旨在提高模型在极端情况下的分割性能，如遮挡、变形、相似物体干扰等。

## 核心思想

### 记忆库设计

1. **长期记忆库 (Long-term Memory)**
   - 存储高质量、有价值的关键帧
   - 目的：在极端情况下提供可靠的参考信息
   - 容量有限（默认20帧）
   - 基于相似度动态更新

2. **短期记忆库 (Short-term Memory)**
   - 存储最近的帧（类似原始SAM2）
   - 目的：提供时间连续性的参考
   - FIFO更新策略（默认保留10帧）
   - 保证时序信息的流畅性

### 筛选机制

每一帧从mask decoder输出后，会经过三步筛选：

#### **步骤1: 物体存在分数检查**
```python
object_score = sigmoid(object_score_logits)
if object_score < threshold:  # 默认0.0
    丢弃或存入短期记忆
```

#### **步骤2: IoU质量检查**
```python
iou_score = max(ious)
if iou_score < threshold:  # 默认0.7
    丢弃或存入短期记忆
```

#### **步骤3: 多掩码一致性检查**
- SAM2的mask decoder输出3个候选掩码
- 计算3个掩码之间的IoU
- 如果一致性低（< 0.8），说明遇到极端情况 → **存入长期记忆库**
- 如果一致性高，继续下一步：

#### **步骤4: 与长期记忆的相似度检查**
- 计算当前掩码与长期记忆库中所有帧的相似度
- 如果相似度高（> 0.85）→ **存入短期记忆库**（避免冗余）
- 如果相似度低 → **存入长期记忆库**（新的有价值信息）
  - 同时移除长期记忆库中最相似的那一帧（如果超出容量）

### 帧采样策略

为了避免视觉特征冗余：
- 每隔N帧（默认3帧）才应用上述筛选机制
- 中间帧直接存入短期记忆库，不经过筛选

## 使用方法

### 1. 训练时启用双记忆库

```bash
python main.py \
    --dataset_file ytvos \
    --use_dual_memory \
    --short_term_capacity 10 \
    --long_term_capacity 20 \
    --object_score_threshold 0.0 \
    --iou_threshold 0.7 \
    --mask_consistency_threshold 0.8 \
    --similarity_threshold 0.85 \
    --frame_sampling_interval 3
```

### 2. 推理时启用双记忆库

```bash
python inference_ytvos.py \
    --resume checkpoints/samwise_dual_memory.pth \
    --use_dual_memory \
    --short_term_capacity 10 \
    --long_term_capacity 20 \
    --iou_threshold 0.7 \
    --mask_consistency_threshold 0.8 \
    --similarity_threshold 0.85 \
    --frame_sampling_interval 3
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_dual_memory` | False | 启用双记忆库系统 |
| `--short_term_capacity` | 10 | 短期记忆库容量（帧数） |
| `--long_term_capacity` | 20 | 长期记忆库容量（帧数） |
| `--object_score_threshold` | 0.0 | 物体存在分数阈值 |
| `--iou_threshold` | 0.7 | IoU质量阈值 |
| `--mask_consistency_threshold` | 0.8 | 多掩码一致性阈值 |
| `--similarity_threshold` | 0.85 | 与长期记忆相似度阈值 |
| `--frame_sampling_interval` | 3 | 帧采样间隔 |

## 参数调优建议

### 场景1: 高质量视频（无明显遮挡）
```bash
--iou_threshold 0.8 \
--mask_consistency_threshold 0.85 \
--similarity_threshold 0.9 \
--frame_sampling_interval 5
```
- 提高阈值，减少长期记忆库存储频率
- 增大采样间隔，提高效率

### 场景2: 复杂场景（频繁遮挡、变形）
```bash
--iou_threshold 0.6 \
--mask_consistency_threshold 0.75 \
--similarity_threshold 0.8 \
--frame_sampling_interval 2
```
- 降低阈值，更容易捕获极端情况
- 减小采样间隔，更密集地监控

### 场景3: 相似物体干扰
```bash
--mask_consistency_threshold 0.7 \
--long_term_capacity 30 \
--similarity_threshold 0.85
```
- 降低一致性阈值，更敏感地检测混淆
- 增大长期记忆容量，保留更多关键帧

## 代码架构

### 文件结构
```
models/
├── memory_bank_manager.py  # 新增：记忆库管理器
├── model_utils.py          # 修改：DecoderOutput添加multimask字段
├── samwise.py              # 修改：集成双记忆库
└── sam2/
    └── modeling/
        └── sam2_base.py    # 修改：保存multimask输出

opts.py                     # 修改：添加新参数
```

### 关键类和方法

#### `MemoryBankManager` (memory_bank_manager.py)
- `should_apply_filtering(frame_idx)`: 判断是否应用筛选
- `check_quality_thresholds(decoder_out)`: 步骤1和2
- `check_multimask_consistency(decoder_out)`: 步骤3
- `calculate_similarity_with_long_term(mask)`: 步骤4
- `store_to_memory(frame_idx, mem_dict, decoder_out)`: 存储逻辑
- `get_memory_bank_for_retrieval()`: 获取合并后的记忆库

#### `SAMWISE` (samwise.py)
- `__init__`: 初始化MemoryBankManager
- `forward`: 在每帧后调用存储逻辑
- `compute_decoder_out_w_mem`: 根据需要请求multimask输出

## 调试和可视化

### 打印记忆库状态
```python
# 在训练/推理循环中添加
model.memory_bank_manager.print_memory_status()
```

输出示例：
```
Frame 0 -> SHORT-TERM (consistency=0.92)
Frame 3 -> LONG-TERM (consistency=0.65)
Frame 6 -> SHORT-TERM (similarity=0.91 with frame 3)
Frame 9 -> LONG-TERM (new info, similarity=0.70)
Memory Status: Short-term=10/10, Long-term=5/20
```

### 监控筛选过程
筛选过程会自动打印日志：
- 丢弃的帧：`Frame X discarded: iou=0.65 < threshold=0.7`
- 存入长期记忆：`Frame X -> LONG-TERM (consistency=0.65)`
- 存入短期记忆：`Frame X -> SHORT-TERM (similarity=0.91 with frame Y)`

## 实验结果预期

使用双记忆库系统，预期在以下场景有显著提升：

1. **遮挡场景** (Occlusion)
   - 长期记忆保留遮挡前的清晰帧
   - 遮挡后恢复时能快速重新识别

2. **形变场景** (Deformation)
   - 捕获形变关键时刻
   - 提供多样化的形状参考

3. **相似物体** (Similar Object Distraction)
   - 长期记忆保留目标物体的独特特征
   - 减少混淆

## 向后兼容性

如果不启用 `--use_dual_memory`，系统将使用原始的单一记忆库，完全向后兼容。

## 注意事项

1. **计算开销**: 启用multimask输出会增加约15-20%的计算时间
2. **内存占用**: 长期记忆库会额外占用显存，建议根据GPU容量调整`long_term_capacity`
3. **训练稳定性**: 建议先用原始方法预训练，再用双记忆库微调

## 引用

如果这个功能对你的研究有帮助，请引用SAMWISE论文并提及本增强版本。

## 联系方式

如有问题或建议，请提交issue或联系作者。
