# SAMWISE with Dual Memory Bank System

## 🎯 概述

这是对SAMWISE模型的创新性增强，实现了**长期记忆库**（Long-term Memory）和**短期记忆库**（Short-term Memory）的双记忆系统，旨在提高模型在极端情况下的视频分割性能。

### 主要创新点

1. **智能记忆分层**: 根据掩码质量和场景复杂度，自动将帧分配到长期或短期记忆库
2. **三级筛选机制**: 物体存在分数 → IoU质量 → 多掩码一致性检查
3. **相似度管理**: 避免冗余信息，保持记忆库的多样性
4. **帧采样策略**: 避免视觉特征冗余，提高效率

## 📁 修改的文件

### 新增文件
- `models/memory_bank_manager.py` - 记忆库管理器核心实现
- `docs/DUAL_MEMORY_BANK_GUIDE.md` - 详细使用指南
- `tests/test_dual_memory_bank.py` - 单元测试
- `examples/train_with_dual_memory.sh` - 训练示例脚本
- `examples/inference_with_dual_memory.sh` - 推理示例脚本

### 修改的文件
- `opts.py` - 添加8个新的超参数
- `models/model_utils.py` - DecoderOutput 添加 multimask 字段
- `models/samwise.py` - 集成记忆库管理器
- `models/sam2/modeling/sam2_base.py` - 保存 multimask 输出

## 🚀 快速开始

### 1. 安装依赖

```bash
# 无需额外依赖，使用原有环境即可
# 确保已安装SAMWISE的基础依赖
```

### 2. 训练示例

```bash
# 基础训练（启用双记忆库）
bash examples/train_with_dual_memory.sh

# 或者直接使用Python
python main.py \
    --dataset_file ytvos \
    --use_dual_memory \
    --short_term_capacity 10 \
    --long_term_capacity 20 \
    --iou_threshold 0.7 \
    --mask_consistency_threshold 0.8 \
    --similarity_threshold 0.85 \
    --frame_sampling_interval 3
```

### 3. 推理示例

```bash
# 在Ref-YouTube-VOS上推理
bash examples/inference_with_dual_memory.sh

# 或者直接使用Python
python inference_ytvos.py \
    --resume checkpoints/samwise_dual_memory.pth \
    --use_dual_memory \
    --short_term_capacity 10 \
    --long_term_capacity 20
```

### 4. 运行测试

```bash
# 创建测试目录
mkdir -p tests

# 运行单元测试
cd tests
python test_dual_memory_bank.py

# 输出应显示所有测试通过
# ✓ ALL TESTS PASSED!
```

## 📊 参数说明

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

## 🔬 工作原理

### 筛选流程图

```
Frame N from Mask Decoder
    ↓
[Step 1] Object Score > threshold?
    ↓ Yes
[Step 2] IoU > threshold?
    ↓ Yes
[Step 3] Check 3 masks consistency
    ↓
    ├─ Inconsistent (extreme case) → LONG-TERM MEMORY
    │
    └─ Consistent → [Step 4] Calculate similarity with long-term
        ↓
        ├─ High similarity (>0.85) → SHORT-TERM MEMORY (FIFO)
        │
        └─ Low similarity (<0.85) → LONG-TERM MEMORY
                                     (Replace most similar frame)
```

### 记忆库检索

在进行分割时，模型会：
1. 首先查找**短期记忆库**（最新的信息）
2. 如果短期记忆库没有，再查找**长期记忆库**（关键帧）
3. 合并两个记忆库的信息进行分割

## 📈 预期效果

基于设计原理，双记忆库系统在以下场景应有显著提升：

| 场景 | 原因 | 预期提升 |
|------|------|----------|
| 遮挡恢复 | 长期记忆保留遮挡前的清晰帧 | J&F +3~5% |
| 物体变形 | 捕获关键形变时刻 | J&F +2~4% |
| 相似物体 | 保留目标物体的独特特征 | J&F +2~3% |
| 长视频 | 长期记忆避免特征漂移 | J&F +1~2% |

## 🐛 调试技巧

### 1. 查看记忆库状态

在训练/推理脚本中添加：
```python
# 在主循环中
model.memory_bank_manager.print_memory_status()
```

输出示例：
```
Frame 3 -> LONG-TERM (consistency=0.65)
Frame 6 -> SHORT-TERM (similarity=0.91 with frame 3)
Memory Status: Short-term=10/10, Long-term=5/20
```

### 2. 可视化记忆库内容

```python
# 查看哪些帧在长期记忆中
print("Long-term frames:", list(model.memory_bank_manager.long_term_memory.keys()))

# 查看哪些帧在短期记忆中
print("Short-term frames:", list(model.memory_bank_manager.short_term_memory.keys()))
```

### 3. 调整阈值

如果发现：
- **长期记忆库填充太快**: 提高 `mask_consistency_threshold` 和降低 `similarity_threshold`
- **长期记忆库填充太慢**: 降低 `mask_consistency_threshold` 和提高 `similarity_threshold`
- **筛选过于频繁**: 增大 `frame_sampling_interval`

## 🔧 故障排除

### 问题1: 显存不足

```bash
# 减少记忆库容量
--short_term_capacity 5 \
--long_term_capacity 10
```

### 问题2: 速度太慢

```bash
# 增大采样间隔，减少multimask计算
--frame_sampling_interval 5
```

### 问题3: 效果不明显

```bash
# 尝试更敏感的阈值
--mask_consistency_threshold 0.75 \
--iou_threshold 0.65
```

## 📚 技术细节

### 记忆库数据结构

```python
{
    'maskmem_features': Tensor,  # 编码后的记忆特征
    'maskmem_pos_enc': Tensor,   # 位置编码
    'pred_masks': Tensor,        # 预测掩码
    'obj_ptr': Tensor,           # 对象指针
}
```

### 相似度计算

使用IoU作为相似度度量：
```python
similarity = intersection / union
```

### 多掩码一致性

计算3个掩码两两之间的IoU，取平均值：
```python
consistency = (IoU_01 + IoU_02 + IoU_12) / 3
```

## 🎓 实验建议

### 消融实验

1. **关闭双记忆库**（基线）
```bash
python main.py --dataset_file ytvos  # 不加 --use_dual_memory
```

2. **只用短期记忆**
```bash
--use_dual_memory --long_term_capacity 0
```

3. **只用长期记忆**
```bash
--use_dual_memory --short_term_capacity 0
```

4. **不同容量配比**
```bash
# 配比1: 短期多，长期少
--short_term_capacity 15 --long_term_capacity 5

# 配比2: 短期少，长期多
--short_term_capacity 5 --long_term_capacity 30
```

### 不同场景的最佳参数

详见 `docs/DUAL_MEMORY_BANK_GUIDE.md`

## 🤝 贡献

如有问题或改进建议，欢迎提交issue或PR。

## 📖 引用

如果这个增强版本对你的研究有帮助，请引用SAMWISE原始论文。

## 📝 更新日志

### v1.0 (2025-01)
- ✅ 实现长期/短期记忆库分离
- ✅ 实现三级筛选机制
- ✅ 实现帧采样策略
- ✅ 实现相似度管理
- ✅ 完整的单元测试
- ✅ 详细的使用文档

## 💡 未来工作

- [ ] 自适应阈值调整
- [ ] 基于attention的相似度计算
- [ ] 长期记忆的聚类压缩
- [ ] 跨视频的记忆迁移

---

**Happy Segmenting with Dual Memory! 🎉**
