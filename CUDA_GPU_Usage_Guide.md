# LSTM风电功率预测 - CUDA GPU加速使用指南

## 🚀 GPU版本改进特性

### 1. 自动GPU检测和配置
- 自动检测CUDA可用性
- 显示GPU名称和内存信息
- 自动回退到CPU（如果GPU不可用）

### 2. 针对GPU优化的超参数
```python
LSTM_HIDDEN_SIZE = 512      # 更大的隐藏层（CPU版本: 256）
FC_LAYERS = [256, 128, 64]  # 更大的全连接层
BATCH_SIZE = 64             # 更大的批次大小（CPU版本: 32）
MAX_EPOCHS = 200            # 更多训练轮数（CPU版本: 100）
```

### 3. GPU内存管理
- 实时GPU内存使用监控
- 自动内存清理（每个epoch后）
- 防止内存泄漏

### 4. 性能优化特性
- 混合精度训练（AMP）支持
- 模型编译优化（PyTorch 2.0+）
- CUDA随机种子设置

## 📋 系统要求

### 硬件要求
- NVIDIA GPU（支持CUDA）
- 推荐：RTX 3060或更高
- 最低GPU内存：6GB
- 推荐GPU内存：8GB+

### 软件要求
```bash
# 检查CUDA版本
nvidia-smi

# 安装PyTorch GPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔧 使用方法

### 1. 运行GPU版本
```bash
cd Python/AI/Lab2
python LSTM_cuda_multi_steps.py
```

### 2. 预期输出示例
```
--- Wind Power Prediction using Improved LSTM (CUDA GPU Version) ---
Using device: cuda
GPU Name: NVIDIA GeForce RTX 4080
GPU Memory: 16.0 GB
CUDA random seeds set for reproducibility
Mixed precision training enabled (AMP)
Model compiled for better performance

ImprovedLSTMModel initialized.
  Input features: 1
  Hidden size: 512, Num layers: 3, Bidirectional: True
  Using attention mechanism and step-specific predictors

Epoch [1/200], Train Loss: 0.234567, Val Loss: 0.198765, Duration: 2.34s, GPU Mem: 3.2/4.1GB
```

## 📊 性能对比

### 训练速度提升
- **CPU版本**: ~15-20秒/epoch
- **GPU版本**: ~2-5秒/epoch
- **加速比**: 3-8倍

### 模型容量提升
- **隐藏层大小**: 256 → 512
- **全连接层**: [128,64,32] → [256,128,64]
- **批次大小**: 32 → 64

### 训练效果提升
- 更大的模型容量
- 更多的训练轮数
- 更好的收敛性能

## 🛠️ 故障排除

### 1. CUDA不可用
```
CUDA not available, using device: cpu
```
**解决方案**:
- 检查GPU驱动安装
- 重新安装PyTorch GPU版本
- 确认CUDA版本兼容性

### 2. GPU内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**:
- 减小BATCH_SIZE（64 → 32 → 16）
- 减小LSTM_HIDDEN_SIZE（512 → 256）
- 减小WINDOW_SIZE（96 → 48）

### 3. 混合精度训练问题
```
Mixed precision training not available
```
**解决方案**:
- 更新PyTorch版本
- 检查GPU是否支持Tensor Cores
- 手动禁用混合精度训练

## 📈 性能监控

### GPU使用率监控
```bash
# 实时监控GPU使用情况
nvidia-smi -l 1

# 或使用
watch -n 1 nvidia-smi
```

### 内存使用优化
```python
# 在代码中添加内存监控
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    torch.cuda.empty_cache()  # 清理缓存
```

## 🎯 最佳实践

### 1. 批次大小调优
- 从64开始，根据GPU内存调整
- 监控GPU利用率，保持在80-95%

### 2. 模型大小平衡
- 隐藏层大小与GPU内存成正比
- 避免过大导致内存溢出

### 3. 训练策略
- 使用早停防止过拟合
- 监控验证损失趋势
- 保存最佳模型检查点

### 4. 数据加载优化
```python
# 使用多进程数据加载
DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

## 📝 配置建议

### 不同GPU配置推荐

#### RTX 3060 (8GB)
```python
LSTM_HIDDEN_SIZE = 256
BATCH_SIZE = 32
FC_LAYERS = [128, 64, 32]
```

#### RTX 3080/4070 (12GB)
```python
LSTM_HIDDEN_SIZE = 512
BATCH_SIZE = 64
FC_LAYERS = [256, 128, 64]
```

#### RTX 4080/4090 (16GB+)
```python
LSTM_HIDDEN_SIZE = 768
BATCH_SIZE = 128
FC_LAYERS = [512, 256, 128]
```

## 🔍 结果分析

### 预期改进效果
1. **训练速度**: 3-8倍提升
2. **模型精度**: 更大模型容量带来更好性能
3. **滞后减少**: 改进的架构减少长期预测滞后
4. **收敛稳定性**: GPU上可以训练更多轮数

### 对比指标
- 训练时间缩短
- C_R准确率提升
- MSE误差降低
- 长期预测滞后减少

运行GPU版本后，请对比CPU版本的结果，验证改进效果！
