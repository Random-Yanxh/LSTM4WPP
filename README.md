# 基于LSTM的风电功率多步时间序列预测项目

本项目旨在使用长短期记忆网络 (LSTM) 实现风电功率的多步时间序列预测。项目提供了两个主要的Python脚本：`LSTM_cuda.py`（支持GPU加速）和 `LSTM_WPP.py`（仅CPU运行）。

## 项目目标

*   利用历史风电功率数据预测未来多个时间点的功率值。
*   对比GPU加速和纯CPU环境下的模型训练与评估流程。
*   可视化训练过程中的损失变化以及在训练集和测试集上的预测结果。

## 核心技术栈

*   **PyTorch**:主要的深度学习框架，用于构建和训练LSTM模型。
*   **Pandas**:用于高效地加载和处理原始数据（Excel文件）。
*   **NumPy**:用于数值计算，特别是在数据转换和序列构建中。
*   **Matplotlib**:用于结果可视化，包括损失曲线和预测对比图。
*   **Scikit-learn**:用于数据标准化（`MinMaxScaler`, `StandardScaler`）和性能评估（`mean_squared_error`）。

## 文件结构

*   `LSTM_cuda.py`: 支持CUDA加速的PyTorch LSTM模型实现。
*   `LSTM_WPP.py`: 仅使用CPU运行的PyTorch LSTM模型实现。
*   `train_set.xlsx`: 训练数据集（单列功率数据，无表头）。
*   `test_set.xlsx`: 测试数据集（单列功率数据，无表头）。
*   `figures/` (或 `figures_cpu/`): 存放生成的图像文件（损失曲线、预测对比图）。
*   `lstm_wpp_model.pth` (或 `lstm_wpp_model_cpu.pth`): 保存的训练好的模型状态字典。
*   `README.md`: 本技术文档。

## 程序主要思路与流程

两个脚本 (`LSTM_cuda.py` 和 `LSTM_WPP.py`) 遵循相似的核心逻辑流程，主要区别在于设备（GPU/CPU）的选择和部分CUDA相关的特定设置。以下是通用的流程概述：

### I. 项目设置与配置

1.  **环境初始化**:
    *   导入所有必要的库。
    *   **设备检测与设置**:
        *   `LSTM_cuda.py`: 自动检测可用的CUDA设备，如果CUDA可用则使用GPU，否则回退到CPU。通过 `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")` 实现。
        *   `LSTM_WPP.py`: 强制将设备设置为CPU，`DEVICE = torch.device("cpu")`。
2.  **定义全局超参数**:
    *   `PREDICTION_STEPS`: 预测未来多少个时间步。
    *   `WINDOW_SIZE`: 输入历史序列的窗口大小。
    *   LSTM模型配置: `LSTM_HIDDEN_SIZE` (隐藏单元数), `LSTM_NUM_LAYERS` (层数), `LSTM_DROPOUT` (Dropout率)。
    *   可选全连接层配置: `FC_LAYERS` (一个定义各FC层单元数的列表)。
    *   训练配置: `BATCH_SIZE`, `LEARNING_RATE`, `MAX_EPOCHS`。
    *   早停配置: `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_MIN_DELTA`。
    *   数据标准化方法: `NORMALIZATION_METHOD` ("minmax" 或 "standard")。
    *   文件路径: `MODEL_SAVE_PATH`, `FIGURE_SAVE_DIR`, `TRAIN_DATA_PATH`, `TEST_DATA_PATH`。

### 超参数详解

下表详细说明了脚本中定义的关键超参数及其作用：

| 超参数                      | 脚本中变量名                | 类型        | 默认值示例         | 描述                                                                                                                               |
| :-------------------------- | :-------------------------- | :---------- | :----------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **数据与序列相关**          |                             |             |                    |                                                                                                                                    |
| 预测未来步数                | `PREDICTION_STEPS`          | `int`       | `16`               | 模型需要预测未来多少个时间步。                                                                                                         |
| 输入历史窗口大小            | `WINDOW_SIZE`               | `int`       | `96`               | 模型进行一次预测时回顾的过去历史时间点数量。                                                                                               |
| 数据标准化方法              | `NORMALIZATION_METHOD`      | `str`       | `"minmax"`         | 可选 "minmax" (缩放到0-1) 或 "standard" (Z-score标准化)。                                                                           |
| **LSTM 模型结构相关**       |                             |             |                    |                                                                                                                                    |
| LSTM隐藏单元数              | `LSTM_HIDDEN_SIZE`          | `int`       | `128`              | LSTM层中隐藏状态的维度。影响模型容量。                                                                                                  |
| LSTM层数                    | `LSTM_NUM_LAYERS`           | `int`       | `2`                | LSTM网络的层数。                                                                                                                      |
| LSTM Dropout率             | `LSTM_DROPOUT`              | `float`     | `0.2`              | 应用于LSTM层之间（若层数>1）及LSTM输出后的Dropout比率，防止过拟合。                                                                      |
| 全连接层配置                | `FC_LAYERS`                 | `List[int]` | `[64, 32]`         | 定义LSTM层后的全连接层结构，列表内数字为各层单元数。空列表 `[]` 表示无额外FC层。                                                                 |
| **训练过程相关**            |                             |             |                    |                                                                                                                                    |
| 批处理大小                  | `BATCH_SIZE`                | `int`       | `64`               | 每次模型参数更新时使用的训练样本数量。                                                                                                   |
| 学习率                      | `LEARNING_RATE`             | `float`     | `0.001`            | 控制模型参数更新的幅度。                                                                                                                |
| 最大训练轮数                | `MAX_EPOCHS`                | `int`       | `100`              | 模型训练的最大轮次。                                                                                                                  |
| 早停耐心轮数                | `EARLY_STOPPING_PATIENCE`   | `int`       | `10`               | 验证集损失连续多少轮未改善则提前停止训练。                                                                                                 |
| 早停最小改善阈值            | `EARLY_STOPPING_MIN_DELTA`  | `float`     | `0.0001`           | 验证损失改善小于此阈值时不计为有效改善。                                                                                                   |
| **文件路径相关**            |                             |             |                    |                                                                                                                                    |
| 模型保存路径                | `MODEL_SAVE_PATH`           | `str`       | `"./lstm_wpp_model.pth"` | 训练好的模型状态字典保存路径。CPU版本会添加 `_cpu` 后缀。                                                                                |
| 图像保存目录                | `FIGURE_SAVE_DIR`           | `str`       | `"./figures"`      | 生成的图像文件保存目录。CPU版本会添加 `_cpu` 后缀。                                                                                     |
| 训练数据路径                | `TRAIN_DATA_PATH`           | `str`       | `"train_set.xlsx"` | 训练数据文件的路径。                                                                                                                  |
| 测试数据路径                | `TEST_DATA_PATH`            | `str`       | `"test_set.xlsx"`  | 测试数据文件的路径。                                                                                                                  |
| **其他**                    |                             |             |                    |                                                                                                                                    |
| 全局随机种子                | `SEED`                      | `int`       | `42`               | 用于初始化随机数生成器，确保实验可复现性。                                                                                                 |

3.  **设置全局随机种子**:
    *   通过 `set_seed` 函数为 `random`, `numpy`, `torch` 设置随机种子，以保证实验的可复现性。
    *   `LSTM_cuda.py` 中的 `set_seed` 包含CUDA相关的种子设置 (`torch.cuda.manual_seed_all` 和 `cudnn` 后端设置)，而 `LSTM_WPP.py` 中已移除这些CUDA特定行。
4.  **创建输出目录**: 检查并创建用于保存图像的目录 (`FIGURE_SAVE_DIR`)。

### II. 数据处理模块

1.  **原始数据加载与清洗 (`load_and_clean_data` 函数)**:
    *   从指定的Excel文件（`train_set.xlsx` 或 `test_set.xlsx`）加载数据。文件应为单列，无表头。
    *   将加载的数据转换为Pandas Series。
    *   使用 `pd.to_numeric` 将数据转换为数值类型，无法转换的值设为NaN。
    *   处理缺失值（NaN）：首先尝试向前填充 (`ffill`)，然后向后填充 (`bfill`)，最后如果仍有NaN，则用0填充。
2.  **数据标准化与反标准化 (`DataScaler` 类)**:
    *   该类封装了数据的标准化和反标准化逻辑。
    *   构造函数接收 `method` 参数（"minmax" 或 "standard"），并初始化相应的Scikit-learn scaler (`MinMaxScaler` 或 `StandardScaler`)。
    *   `fit(data)`: 根据训练数据计算并存储标准化所需的参数（如min/max或mean/std）。
    *   `transform(data)`: 使用已存储的参数对输入数据进行标准化。
    *   `fit_transform(data)`: 结合 `fit` 和 `transform`。
    *   `inverse_transform(data)`: 使用已存储的参数将标准化数据还原到原始尺度。
    *   **关键**: 在训练集上 `fit` 得到的 `DataScaler` 实例必须用于后续所有数据（验证集、测试集）的 `transform` 和 `inverse_transform`，以保证数据一致性。
3.  **序列数据构建 (`create_sequences` 函数)**:
    *   将标准化后的1D时间序列数据（NumPy数组）转换为适用于LSTM输入的 (X, y) PyTorch张量。
    *   X (输入) 张量形状: `(样本数, WINDOW_SIZE, 1)`。每个样本包含 `WINDOW_SIZE` 个历史时间点的数据。
    *   y (目标) 张量形状: `(样本数, PREDICTION_STEPS)`。每个样本对应未来 `PREDICTION_STEPS` 个时间点的真实值。
    *   函数通过滑动窗口的方式从输入数据中提取序列。
    *   生成的张量会被移动到已配置的 `DEVICE` 上。

### III. LSTM模型定义 (`LSTMModel` 类)

*   定义一个继承自 `torch.nn.Module` 的LSTM模型类。
*   **构造函数 `__init__`**:
    *   接收模型配置参数：`input_features` (固定为1，因为是单变量时间序列), `hidden_size` (LSTM隐藏单元数), `num_layers` (LSTM层数), `dropout_rate`, `fc_layers_config` (全连接层配置), `output_steps` (预测步数)。
    *   包含一个 `nn.LSTM` 层 (`batch_first=True`)。如果 `num_layers > 1`，则LSTM层内部的 `dropout` 参数生效。
    *   一个 `nn.Dropout` 层，在LSTM层之后应用。
    *   一个可选的 `nn.Sequential` 模块，用于构建由 `fc_layers_config` 定义的多个全连接层（每个 `nn.Linear` 后接一个 `nn.ReLU` 激活函数）。
    *   一个最终的 `nn.Linear` 输出层，将特征映射到 `output_steps` 维度。
*   **`forward(x)` 方法**:
    *   定义数据通过模型的前向传播路径。
    *   初始化LSTM的隐藏状态 (h0) 和细胞状态 (c0)。
    *   数据通过 `nn.LSTM` 层。取LSTM层最后一个时间步的输出作为后续层的输入。
    *   通过 `nn.Dropout` 层。
    *   如果定义了全连接层，则数据通过 `self.fc_layers`。
    *   最后通过输出层 `self.output_layer` 得到预测结果。

### IV. 模型训练与验证模块

1.  **数据准备 (`prepare_dataloaders` 函数)**:
    *   加载并清洗原始训练数据 (`TRAIN_DATA_PATH`)。
    *   使用传入的 `DataScaler` 实例对全部训练数据进行 `fit_transform` 标准化。
    *   按时间顺序将标准化后的训练数据划分为训练子集和验证子集（基于 `val_split_ratio`）。
    *   为训练子集和验证子集分别调用 `create_sequences` 构建PyTorch序列数据。
    *   创建对应的 `TensorDataset` 和 `DataLoader`。训练集的 `DataLoader` 会打乱数据 (`shuffle=True`)，验证集则不打乱。
    *   返回训练 `DataLoader` 和验证 `DataLoader`。
2.  **训练主函数 (`train_model` 函数)**:
    *   接收模型实例、训练和验证 `DataLoader`、损失函数 (`criterion`)、优化器 (`optimizer`)、训练配置（最大轮数、早停参数等）、设备和模型保存路径。
    *   **训练主循环 (Epochs)**:
        *   **训练阶段**:
            *   设置模型为训练模式 (`model.train()`)。
            *   遍历训练 `DataLoader` 中的每个批次。
            *   将数据批次移至 `DEVICE`。
            *   执行优化器梯度清零 (`optimizer.zero_grad()`)。
            *   模型前向传播得到输出 (`outputs = model(X_batch)`)。
            *   计算损失 (`loss = criterion(outputs, y_batch)`)。
            *   反向传播计算梯度 (`loss.backward()`)。
            *   优化器更新参数 (`optimizer.step()`)。
            *   累积批次损失，计算平均训练损失。
        *   **验证阶段**:
            *   设置模型为评估模式 (`model.eval()`)。
            *   在 `torch.no_grad()` 上下文中执行，禁用梯度计算。
            *   遍历验证 `DataLoader`，计算平均验证损失。
        *   打印当前epoch的训练损失、验证损失和持续时间。
        *   **早停逻辑**:
            *   比较当前验证损失与历史最佳验证损失。
            *   如果验证损失在连续 `EARLY_STOPPING_PATIENCE` 轮内没有改善超过 `EARLY_STOPPING_MIN_DELTA`，则提前终止训练。
            *   若验证损失改善，则保存当前模型的状态字典到临时文件 (`best_model_temp_cpu.pth` 或 `best_model_temp.pth`)。
    *   **模型保存**:
        *   训练结束后，如果早停被触发且最佳模型已保存，则加载性能最佳的模型状态。
        *   保存最终的模型状态字典到 `MODEL_SAVE_PATH`。
        *   移除临时保存的最佳模型文件。
    *   返回训练损失和验证损失的历史记录。

### V. 模型评估与预测模块

1.  **通用预测函数 (`make_predictions` 函数)**:
    *   接收已训练模型、数据加载器和设备。
    *   设置模型为评估模式 (`model.eval()`) 并在 `torch.no_grad()` 下执行。
    *   遍历数据加载器（通常只使用输入X），收集模型对所有批次的完整多步预测。
    *   将所有批次的预测结果连接起来，并转换为NumPy数组。
    *   返回形状为 `(总样本数, PREDICTION_STEPS)` 的预测结果数组。
2.  **在指定数据集上进行评估 (`evaluate_on_dataset` 函数)**:
    *   接收数据集类型标识（"Train"或"Test"）、模型、原始数据路径、已学习的 `DataScaler` 实例、窗口大小、预测步数、设备和批大小。
    *   加载并清洗相应数据集的原始数据。
    *   使用已学习的 `DataScaler` 对数据进行 `transform` 标准化。
    *   调用 `create_sequences` 构建PyTorch序列数据 (X_eval, y_true_scaled_tensor)。
    *   创建评估用的 `DataLoader`（不打乱数据）。
    *   调用 `make_predictions` 获取标准化尺度的预测结果 (`y_pred_scaled`)。
    *   **提取第一步预测**: 从 `y_pred_scaled` (形状 `(N, PREDICTION_STEPS)`) 中提取所有样本的第一步预测 `y_pred_scaled[:, 0]`。同样从 `y_true_scaled_tensor` 中提取对应的真实值。
    *   **反标准化**: 使用 `DataScaler` 的 `inverse_transform` 方法对第一步的预测值和真实值进行反标准化，还原到原始数据尺度。
    *   **计算MSE**: 使用 `sklearn.metrics.mean_squared_error` 计算反标准化后的真实值和预测值之间的均方误差 (MSE)。
    *   打印该数据集上的MSE。
    *   返回反标准化后的真实值、预测值（均为1D NumPy数组，用于绘图）以及计算出的MSE。

### VI. 结果可视化模块

1.  **损失曲线绘制函数 (`plot_loss_curves` 函数)**:
    *   接收训练损失历史、验证损失历史、保存路径以及用于在图上显示的附加信息（网络参数字符串、学习率、实际训练轮数）。
    *   使用 `matplotlib.pyplot` 在同一图上绘制训练损失和验证损失曲线。
    *   添加标题、坐标轴标签、图例和网格。
    *   使用 `plt.figtext` 在图表左下角添加网络参数、学习率和实际训练轮数的文本信息。
    *   调整布局以适应文本信息。
    *   保存图像到指定路径，并使用 `plt.show()` 直接显示图像。
2.  **预测对比图绘制函数 (`plot_predictions_comparison` 函数)**:
    *   接收反标准化后的实际值、预测值、数据集名称、保存路径、该数据集的MSE、网络参数字符串、学习率、实际训练轮数以及最大绘图点数。
    *   如果数据点过多，则截取最后 `max_points` 个点进行绘制。
    *   X轴根据15分钟的时间间隔生成时间标签 (HH:MM)。
    *   使用 `matplotlib.pyplot` 绘制实际值和第一步预测值的对比曲线。
    *   添加标题、坐标轴标签、图例和网格。
    *   使用 `plt.figtext` 在图表左下角添加当前数据集的MSE、网络参数、学习率和实际训练轮数的文本信息。
    *   调整布局以适应文本信息。
    *   保存图像到指定路径，并使用 `plt.show()` 直接显示图像。

### VII. 主执行流程 (`if __name__ == "__main__":`)

1.  **打印程序开始信息** (区分CPU/GPU版本)。
2.  **生成时间戳**: 创建一个 `YYYYMMDD_HHMM` 格式的时间戳字符串，用于后续图像文件的命名，防止覆盖。
3.  **执行全局配置和环境初始化**: 调用 `set_seed`，确保 `FIGURE_SAVE_DIR` 存在，打印使用的设备。
4.  **检查数据文件是否存在**: 确保 `TRAIN_DATA_PATH` 和 `TEST_DATA_PATH` 指向的文件存在。
5.  **实例化 `DataScaler`**。
6.  **训练模型**:
    *   调用 `prepare_dataloaders` 获取训练和验证的 `DataLoader`。`DataScaler` 实例在此过程中会被 `fit`。
    *   实例化 `LSTMModel`，并将其移至 `DEVICE`。
    *   定义损失函数 (`nn.MSELoss`) 和优化器 (`torch.optim.Adam`)。
    *   调用 `train_model` 函数进行模型训练，获取训练和验证损失的历史记录。
    *   计算实际训练的轮数，并准备网络参数的字符串描述。
7.  **可视化训练过程**:
    *   构造带有时间戳的损失曲线保存路径。
    *   调用 `plot_loss_curves` 绘制并保存/显示损失曲线，并传入网络参数、学习率和实际轮数。
8.  **加载最佳模型进行评估**:
    *   从 `MODEL_SAVE_PATH` 加载训练好的模型状态字典到 `lstm_model`。
9.  **在训练集上评估与可视化**:
    *   调用 `evaluate_on_dataset` 函数，在训练集上获取反标准化的实际值、预测值以及MSE。
    *   构造带有时间戳的训练集预测对比图保存路径。
    *   调用 `plot_predictions_comparison` 绘制并保存/显示训练集上的预测对比图，并传入MSE及其他参数信息。
10. **在测试集上评估与可视化**:
    *   调用 `evaluate_on_dataset` 函数，在测试集上获取反标准化的实际值、预测值以及MSE。
    *   构造带有时间戳的测试集预测对比图保存路径。
    *   调用 `plot_predictions_comparison` 绘制并保存/显示测试集上的预测对比图，并传入MSE及其他参数信息。
11. **打印脚本结束信息**。

## 如何运行

1.  确保已安装所有必要的库 (PyTorch, Pandas, NumPy, Matplotlib, Scikit-learn)。
2.  将训练数据文件 `train_set.xlsx` 和测试数据文件 `test_set.xlsx` 放置在与脚本相同的目录下。
3.  根据需要选择运行的脚本：
    *   **GPU版本**: `python LSTM_cuda.py`
    *   **CPU版本**: `python LSTM_WPP.py`
4.  脚本运行时，会输出各个阶段的信息，包括设备使用、数据加载、模型初始化、训练过程中的损失、评估结果（MSE）等。
5.  训练完成后，模型将保存到指定的 `.pth` 文件，相关的可视化图像将保存到 `figures/` 或 `figures_cpu/` 目录，并会直接显示出来。

## 注意事项

*   `MODEL_SAVE_PATH` 和 `FIGURE_SAVE_DIR` 在CPU版本 (`LSTM_WPP.py`) 中已添加 "_cpu" 后缀，以区分GPU版本生成的文件。
*   超参数（如 `WINDOW_SIZE`, `LSTM_HIDDEN_SIZE`, `LEARNING_RATE`, `MAX_EPOCHS` 等）可以在脚本顶部的全局配置部分进行调整以优化模型性能。
*   数据文件应为单列Excel文件，不包含表头。