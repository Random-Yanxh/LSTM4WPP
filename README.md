# 风电功率预测系统技术文档

## 项目概述

本项目实现了基于深度学习的风电功率多步长预测系统，包含五种不同的神经网络架构：**GRU**、**LSTM**、**LSTM-CPU优化版**、**TCN**和**CNN-LSTM混合模型**。系统能够预测未来4小时（16个15分钟时间步）的风电功率输出，为风电场运营和电网调度提供技术支持。

## 系统架构总览

### 核心特性
- **多步长预测**: 同时预测未来16个时间步（4小时）
- **多模型支持**: 五种不同的深度学习架构
- **GPU/CPU支持**: 支持CUDA加速训练和CPU优化版本
- **统一评估**: 标准化的MSE和C_R准确率评估体系
- **可视化输出**: 完整的预测对比图和训练曲线
- **结果保存**: 自动保存C_R准确率分析报告

### 数据流程
```
原始数据 → 数据清洗 → 标准化 → 序列构建 → 模型训练 → 预测评估 → 结果可视化
```


## 共同技术组件

### 1. 数据处理模块

#### 数据加载与清洗
```python
def load_and_clean_data(file_path: str) -> pd.Series:
    # Excel文件加载
    # 数值转换和NaN处理
    # 前向填充和后向填充
```

#### 数据标准化
```python
class DataScaler:
    # 支持MinMax和Standard两种标准化方法
    # 可逆变换，支持预测结果反标准化
```

#### 序列构建
```python
def create_sequences_tensor(data, window_size, prediction_steps):
    # 滑动窗口构建输入序列
    # 多步长目标构建
    # PyTorch张量转换
```

### 2. 训练策略

#### 损失函数
```python
class WeightedMSELoss(nn.Module):
    # 加权MSE损失，对远期预测给予更高权重
    # 权重衰减机制，平衡近期和远期预测
```

#### 优化器配置
```python
optimizer = optim.AdamW(model.parameters(), 
                       lr=LEARNING_RATE, 
                       weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)
```

#### 早停机制
```python
# 验证损失不改善时提前停止训练
# 防止过拟合，节省训练时间
```

### 3. 评估体系

#### MSE评估
```python
# 每个预测步长的均方误差
# 整体预测性能评估
```

#### C_R准确率
```python
def calculate_cr_accuracy(P_M, P_P):
    # 风电功率预测专用准确率指标
    # 考虑功率值大小的相对误差计算
```

### 4. 可视化系统

#### 预测对比图
- 绘制所有数据点（无500点限制）
- 使用等差数列作为横轴（1, 2, 3...）
- 显示MSE、C_R、模型参数等信息

#### C_R准确率报告
- 自动保存为文本文件
- 包含每个预测步长的详细分析
- 统计信息：平均值、最佳值、最差值

## 文件结构与输出

### 程序文件
```
├── GRU_GPU_multi_steps.py          # GRU模型
├── LSTM_GPU_multi_steps.py         # LSTM模型
├── LSTM_cpu_multi_steps.py         # LSTM-CPU优化模型
├── TCN_GPU_multi_steps.py          # TCN模型（优化版）
├── LSTM_CNN_GPU_multi_steps.py     # CNN-LSTM混合模型
├── train_set.xlsx                  # 训练数据
└── test_set.xlsx                   # 测试数据
```

### 输出目录结构
```
├── figures_gru_gpu/                # GRU模型输出
├── figures_gpu/                    # LSTM模型输出
├── figures_cpu/                    # LSTM-CPU优化模型输出
├── figures_tcn_gpu/                # TCN模型输出
└── figures_cnn_lstm/               # CNN-LSTM模型输出
```

### 输出文件类型
```
每个模型目录包含：
├── test_CR_{timestamp}.txt                    # C_R准确率报告
├── loss_curve_{timestamp}.png                 # 训练损失曲线
├── train_predictions_comparison_15min_{timestamp}.png
├── train_predictions_comparison_1h_{timestamp}.png
├── train_predictions_comparison_2h_{timestamp}.png
├── train_predictions_comparison_3h_{timestamp}.png
├── train_predictions_comparison_4h_{timestamp}.png
├── test_predictions_comparison_15min_{timestamp}.png
├── test_predictions_comparison_1h_{timestamp}.png
├── test_predictions_comparison_2h_{timestamp}.png
├── test_predictions_comparison_3h_{timestamp}.png
└── test_predictions_comparison_4h_{timestamp}.png
```

## 使用指南

### 环境要求

#### 硬件要求
- **GPU**: 推荐NVIDIA RTX 3050Ti或更高（4GB+ VRAM）
- **内存**: 8GB+ RAM
- **存储**: 2GB可用空间

#### 软件环境
```bash
# Python 3.8+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy matplotlib scikit-learn openpyxl
```

### 快速开始

#### 1. 数据准备
确保以下文件在项目根目录：
- `train_set.xlsx`: 训练数据集
- `test_set.xlsx`: 测试数据集

#### 2. 运行训练
```bash
# GRU模型
python GRU_GPU_multi_steps.py

# LSTM模型
python LSTM_GPU_multi_steps.py

# LSTM-CPU优化模型（速度优先）
python LSTM_cpu_multi_steps.py

# TCN模型（快速训练）
python TCN_GPU_multi_steps.py

# CNN-LSTM混合模型
python LSTM_CNN_GPU_multi_steps.py
```

#### 3. 结果查看
- 训练过程会实时显示损失和指标
- 图片和报告自动保存到对应的figures目录
- C_R准确率报告保存为txt文件


#### 性能优化建议

**内存不足时的调整策略：**
```python
# 减少批次大小
BATCH_SIZE = 16  # 或 8

# 减少隐藏层大小
HIDDEN_SIZE = 128  # 或 64

# 减少层数
NUM_LAYERS = 2  # 或 1

# 减少全连接层
FC_LAYERS = [64]  # 简化结构
```

**训练速度优化：**
```python
# 提高学习率
LEARNING_RATE = 0.001  # 或更高

# 减少训练轮数
MAX_EPOCHS = 50

# 增加批次大小（GPU内存允许时）
BATCH_SIZE = 64
```

## 技术要点与注意事项

### 1. 数据预处理要点

#### 数据质量检查
```python
# 检查缺失值
data.isnull().sum()

# 检查异常值
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
```

#### 标准化选择
- **Standard Scaler**: 适用于正态分布数据
- **MinMax Scaler**: 适用于有明确边界的数据
- **Robust Scaler**: 适用于有异常值的数据

#### 序列长度设置
```python
WINDOW_SIZE = 96  # 24小时历史数据 (96 * 15min)
PREDICTION_STEPS = 16  # 4小时预测 (16 * 15min)
```

### 2. 模型训练要点

#### 损失函数设计
```python
class WeightedMSELoss(nn.Module):
    # 权重衰减：远期预测权重更高
    # 避免模型过度关注近期预测
    # 平衡短期和长期预测性能
```

#### 学习率调度
```python
# ReduceLROnPlateau: 验证损失停止改善时降低学习率
# 参数：patience=5, factor=0.5
# 有助于模型收敛到更好的局部最优
```

#### 早停策略
```python
# 监控验证损失
# patience=10-15轮
# min_delta=0.0001
# 防止过拟合，节省训练时间
```

### 3. 评估指标解读

#### MSE (均方误差)
- 衡量预测值与真实值的平方差
- 对大误差敏感
- 单位与原数据一致

#### C_R准确率
```python
# 风电功率预测专用指标
# 考虑功率值大小的相对误差
# 值越高表示预测越准确
# 通常期望 > 80%
```



### 5. 模型部署考虑

#### 推理优化
```python
# 模型量化
torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# 模型剪枝
torch.nn.utils.prune.global_unstructured(parameters, pruning_method=torch.nn.utils.prune.L1Unstructured, amount=0.2)

# ONNX导出
torch.onnx.export(model, dummy_input, "model.onnx")
```







