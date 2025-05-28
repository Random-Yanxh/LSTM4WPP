# RTX 3050Ti 4GB 显存优化指南

## 🎯 针对您的GPU的专门优化

### GPU规格
- **型号**: NVIDIA GeForce RTX 3050Ti (笔记本版)
- **显存**: 4GB GDDR6
- **CUDA核心**: 2560个
- **架构**: Ampere (支持Tensor Cores)

## ⚙️ 自动参数调整

代码已经添加了自动检测功能，会根据您的4GB显存自动调整参数：

```python
# 自动检测到RTX 3050Ti后的参数设置
LSTM_HIDDEN_SIZE = 256      # 适合4GB显存
BATCH_SIZE = 24             # 保守的批次大小
FC_LAYERS = [128, 64]       # 精简的全连接层
LSTM_NUM_LAYERS = 2         # 2层LSTM
MAX_EPOCHS = 150            # 适中的训练轮数
```

## 📊 预期性能表现

### 训练速度对比
| 配置 | 每轮训练时间 | 相比CPU提升 |
|------|-------------|-------------|
| CPU版本 | ~15-20秒 | 基准 |
| RTX 3050Ti | ~3-6秒 | 3-5倍 |

### 显存使用预估
```
模型加载: ~1.2GB
训练数据: ~0.8GB
梯度缓存: ~0.6GB
PyTorch开销: ~0.4GB
总计: ~3.0GB (留有1GB余量)
```

## 🚨 显存不足处理策略

### 1. 自动降级参数
如果仍然遇到显存不足，代码会自动尝试：
```python
# 紧急降级参数
BATCH_SIZE = 16             # 进一步减小
LSTM_HIDDEN_SIZE = 192      # 减小隐藏层
WINDOW_SIZE = 64            # 减小窗口大小
```

### 2. 手动调整建议
如果自动调整仍不够，可以手动修改：

```python
# 最保守设置 (适用于显存紧张时)
PREDICTION_STEPS = 16
WINDOW_SIZE = 48            # 减小到12小时历史
LSTM_HIDDEN_SIZE = 128      # 最小可用隐藏层
LSTM_NUM_LAYERS = 2         
FC_LAYERS = [64, 32]        # 最小全连接层
BATCH_SIZE = 16             # 最小批次大小
```

## 🔧 优化技巧

### 1. 禁用混合精度训练
对于4GB显存，已自动禁用AMP以节省显存：
```python
# 代码中已自动处理
print("Mixed precision training disabled for 4GB GPU to save memory")
```

### 2. 启用梯度累积（如需要）
如果批次大小太小影响训练效果，可以启用梯度累积：
```python
# 在训练循环中添加
accumulation_steps = 2  # 累积2个批次再更新
if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 3. 数据加载优化
```python
# 减少数据加载器的内存使用
DataLoader(dataset, batch_size=24, num_workers=2, pin_memory=False)
```

## 📈 监控和调试

### 1. 实时显存监控
```bash
# 在另一个终端运行
nvidia-smi -l 1
```

### 2. 代码中的显存监控
训练过程中会显示：
```
Epoch [1/150], Train Loss: 0.234567, Val Loss: 0.198765, Duration: 4.23s, GPU Mem: 2.8/3.2GB
```

### 3. 显存不足错误处理
如果遇到 `CUDA out of memory` 错误：

1. **立即处理**:
   ```python
   torch.cuda.empty_cache()  # 清理缓存
   ```

2. **重启程序并调整参数**:
   ```python
   BATCH_SIZE = 16  # 减半
   ```

## 🎯 最佳实践

### 1. 渐进式调参
```python
# 从保守参数开始
BATCH_SIZE = 16
LSTM_HIDDEN_SIZE = 128

# 如果运行正常，逐步增加
BATCH_SIZE = 24
LSTM_HIDDEN_SIZE = 192

# 最终目标
BATCH_SIZE = 24
LSTM_HIDDEN_SIZE = 256
```

### 2. 训练策略
- 使用早停防止过拟合
- 监控验证损失，避免过度训练
- 保存最佳模型检查点

### 3. 性能优化
```python
# 关闭不必要的功能
torch.backends.cudnn.benchmark = True  # 加速训练
torch.backends.cudnn.deterministic = False  # 如果不需要完全可重现
```

## 🔍 预期结果

### 训练效果
- **训练速度**: 比CPU快3-5倍
- **模型精度**: 与CPU版本相当或更好
- **滞后减少**: 改进的架构应能减少长期预测滞后

### 资源使用
- **显存使用**: 约75% (3GB/4GB)
- **GPU利用率**: 80-95%
- **功耗**: 约60-80W

## 🚀 运行命令

```bash
# 确保在正确目录
cd Python/AI/Lab2

# 运行GPU优化版本
python LSTM_cuda_multi_steps.py

# 预期输出
Using device: cuda
GPU Name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
GPU Memory: 4.0 GB
Detected GPU memory: 4.0 GB
Adjusted parameters for 4GB GPU (RTX 3050Ti)
Final parameters: Hidden=256, Batch=24, Layers=2
Mixed precision training disabled for 4GB GPU to save memory
```

## 🆘 故障排除

### 常见问题及解决方案

1. **显存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：减小 `BATCH_SIZE` 到 16 或 12

2. **训练太慢**
   ```
   每轮训练超过10秒
   ```
   解决：检查GPU利用率，确保数据在GPU上

3. **精度下降**
   ```
   验证损失不下降
   ```
   解决：适当增加 `LSTM_HIDDEN_SIZE` 或训练轮数

现在您的RTX 3050Ti应该能够高效运行改进的LSTM模型了！
