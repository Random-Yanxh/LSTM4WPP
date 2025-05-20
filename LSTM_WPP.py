# I. 全局配置与环境

# 库导入
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Optional, Union

# 设备指定
device = torch.device('cpu')

# 核心超参数定义
PREDICTION_STEPS: int = 16  # 预测未来步数
WINDOW_SIZE: int = 96  # 输入历史窗口大小
TRAIN_FILE_PATH: str = "Python/AI/Lab2/train_set.xlsx"
TEST_FILE_PATH: str = "Python/AI/Lab2/test_set.xlsx"
LSTM_UNITS: int = 50  # LSTM单元数
LSTM_LAYERS: int = 1  # LSTM层数
FC_LAYERS: Optional[List[int]] = None  # 全连接层配置, 例如 [64, 32] 或 None
BATCH_SIZE: int = 32  # 批量大小
LEARNING_RATE: float = 0.001  # 学习率
EPOCHS: int = 100  # 训练轮次
PATIENCE_EPOCHS: int = 10  # 早停耐心轮次 (当前无验证集，暂不生效)
TARGET_COLUMN_NAME: str = 'Power'  # 目标列名

# 随机性控制
torch.manual_seed(42)
np.random.seed(42)

# II. 数据处理模块

def load_and_clean_data(file_path: str, target_column_name: str) -> pd.Series:
    """
    从指定的Excel文件加载数据，进行清洗并返回Pandas Series。

    参数:
    file_path (str): Excel文件的路径。假设文件为单列，无表头。
    target_column_name (str): 用于命名加载的单列数据的列名。

    返回:
    pd.Series: 清洗后的数据。
    """
    try:
        # 从Excel加载数据，假设无表头，单列数据
        data_df = pd.read_excel(file_path, header=None, names=[target_column_name])
        data_series = data_df[target_column_name]

        # 转换为数值类型，无法转换的设置为NaN
        data_series = pd.to_numeric(data_series, errors='coerce')

        # 缺失值处理：先向前填充，再向后填充，最后用0填充
        data_series = data_series.fillna(method='ffill')
        data_series = data_series.fillna(method='bfill')
        data_series = data_series.fillna(0)

        return data_series
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        # 返回一个空的Series或者抛出异常，根据实际需求
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"加载和清洗数据时发生错误 {file_path}: {e}")
        return pd.Series(dtype=float)

def scale_data(data_series: pd.Series, scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    对输入数据进行Min-Max标准化。

    参数:
    data_series (pd.Series): 待标准化的Pandas Series数据。
    scaler (Optional[MinMaxScaler]): 可选的MinMaxScaler对象。
                                     如果为None（训练数据），则创建并拟合新的scaler。
                                     如果提供（测试/验证数据），则使用已有的scaler进行转换。

    返回:
    Tuple[np.ndarray, MinMaxScaler]: 标准化后的NumPy数组和（可能新建的）scaler对象。
    """
    data_values = data_series.values.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_values)
    else:
        scaled_data = scaler.transform(data_values)

    return scaled_data.flatten(), scaler # 返回1D数组

def create_sequences(input_data: np.ndarray, window_size: int, prediction_steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将标准化的1D时间序列转换为适用于LSTM的 (X, y) 样本对。

    参数:
    input_data (np.ndarray): 标准化后的一维NumPy数组。
    window_size (int): 输入历史窗口大小。
    prediction_steps (int): 预测未来步数。
    device (torch.device): PyTorch设备 (e.g., torch.device('cpu') or torch.device('cuda'))。

    返回:
    Tuple[torch.Tensor, torch.Tensor]:
        X: 输入序列张量，形状 (样本数, window_size, 1)。
        y: 目标序列张量，形状 (样本数, prediction_steps)。
    """
    X_list, y_list = [], []
    data_len = len(input_data)

    for i in range(data_len - window_size - prediction_steps + 1):
        X_list.append(input_data[i : i + window_size])
        y_list.append(input_data[i + window_size : i + window_size + prediction_steps])

    if not X_list: # 如果数据太短，无法创建任何序列
        # 返回空的张量，维度需要匹配后续代码的期望
        # X: (0, window_size, 1), y: (0, prediction_steps)
        X_tensor = torch.empty((0, window_size, 1), dtype=torch.float32, device=device)
        y_tensor = torch.empty((0, prediction_steps), dtype=torch.float32, device=device)
        return X_tensor, y_tensor

    X_np = np.array(X_list)
    y_np = np.array(y_list)

    # 将X的形状调整为 (样本数, window_size, 1) 以适应LSTM的输入要求 (features=1)
    X_np_reshaped = X_np.reshape(X_np.shape[0], X_np.shape[1], 1)

    # 转换为PyTorch张量并移动到指定设备
    X_tensor = torch.tensor(X_np_reshaped, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).to(device)

    return X_tensor, y_tensor

if __name__ == '__main__':
    # 这是一个简单的测试/演示代码块，用于验证上述函数
    print("开始数据处理模块测试...")

    # 1. 测试 load_and_clean_data
    print(f"\n--- 测试 load_and_clean_data ---")
    # 创建一个临时的Excel文件用于测试
    temp_train_data = {'col1': [10, 20, np.nan, 40, 50, '60', 70, None, 90]}
    temp_train_df = pd.DataFrame(temp_train_data)
    temp_train_file = "temp_train_data.xlsx"
    temp_train_df.to_excel(temp_train_file, index=False, header=False)

    train_data_series = load_and_clean_data(temp_train_file, TARGET_COLUMN_NAME)
    print(f"加载并清洗后的训练数据 (前5行):\n{train_data_series.head()}")
    print(f"数据类型: {train_data_series.dtype}")
    print(f"是否有NaN: {train_data_series.isnull().sum()}")

    # 2. 测试 scale_data
    print(f"\n--- 测试 scale_data ---")
    if not train_data_series.empty:
        scaled_train_data, train_scaler = scale_data(train_data_series)
        print(f"标准化后的训练数据 (前5个): {scaled_train_data[:5]}")
        print(f"Scaler: {train_scaler}")

        # 模拟测试数据
        temp_test_data = {'col1': [15, 25, 35, 45, 'abc', 65]} # 包含一个非数值
        temp_test_df = pd.DataFrame(temp_test_data)
        temp_test_file = "temp_test_data.xlsx"
        temp_test_df.to_excel(temp_test_file, index=False, header=False)

        test_data_series = load_and_clean_data(temp_test_file, TARGET_COLUMN_NAME)
        print(f"\n加载并清洗后的测试数据 (前5行):\n{test_data_series.head()}")

        if not test_data_series.empty:
            scaled_test_data, _ = scale_data(test_data_series, scaler=train_scaler)
            print(f"使用训练集的scaler标准化后的测试数据 (前5个): {scaled_test_data[:5]}")
    else:
        print("训练数据为空，跳过scale_data测试。")
        scaled_train_data = np.array([]) # 为后续测试提供空数组

    # 3. 测试 create_sequences
    print(f"\n--- 测试 create_sequences ---")
    if scaled_train_data.size > 0:
        X_train, y_train = create_sequences(scaled_train_data, WINDOW_SIZE, PREDICTION_STEPS, device)
        print(f"X_train 形状: {X_train.shape if X_train is not None else 'None'}")
        print(f"y_train 形状: {y_train.shape if y_train is not None else 'None'}")

        if X_train.numel() > 0 and y_train.numel() > 0:
             print(f"X_train 示例 (第一个序列):\n{X_train[0]}")
             print(f"y_train 示例 (第一个序列):\n{y_train[0]}")
        else:
            print("数据不足以创建序列，或序列创建失败。")

        # 测试数据太短无法创建序列的情况
        short_data = np.array([0.1, 0.2, 0.3])
        X_short, y_short = create_sequences(short_data, WINDOW_SIZE, PREDICTION_STEPS, device)
        print(f"\n对于短数据:")
        print(f"X_short 形状: {X_short.shape}")
        print(f"y_short 形状: {y_short.shape}")

    else:
        print("标准化训练数据为空，跳过create_sequences测试。")

    # 清理临时文件
    import os
    if os.path.exists(temp_train_file):
        os.remove(temp_train_file)
    if os.path.exists(temp_test_file):
        os.remove(temp_test_file)

    print("\n数据处理模块测试结束。")
# III. LSTM模型架构

class LSTMForecastModel(nn.Module):
    """
    用于时间序列预测的LSTM模型。

    该模型包含一个LSTM层，可选的Dropout层，可选的多个全连接层，
    以及一个最终的线性输出层，用于预测未来多个时间步的值。
    """
    def __init__(self,
                 input_features: int,
                 lstm_units: int,
                 lstm_layers: int,
                 prediction_steps: int,
                 fc_layers_config: Optional[List[int]] = None,
                 dropout_rate: float = 0.0):
        """
        初始化LSTMForecastModel。

        参数:
        input_features (int): 输入特征的数量。对于单变量时间序列，此值为1。
        lstm_units (int): LSTM层中的单元数。
        lstm_layers (int): LSTM层数。
        prediction_steps (int): 模型需要预测的未来时间步数。
        fc_layers_config (Optional[List[int]]): 一个整数列表，定义每个全连接层的单元数。
                                                如果为None或空列表，则不使用全连接层。
                                                例如: [64, 32] 表示两个全连接层，分别有64和32个单元。
        dropout_rate (float): Dropout比率。如果大于0，则在LSTM层之后（如果lstm_layers > 1）
                              或单独添加Dropout层。
        """
        super(LSTMForecastModel, self).__init__()
        self.input_features = input_features
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.prediction_steps = prediction_steps
        self.fc_layers_config = fc_layers_config
        self.dropout_rate = dropout_rate

        # LSTM层
        # 注意: nn.LSTM的dropout参数仅在num_layers > 1时生效。
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True, # 输入和输出张量提供为 (batch, seq, feature)
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        # 可选的Dropout层 (如果dropout_rate > 0且lstm_layers == 1，则在此处添加)
        self.dropout = None
        if dropout_rate > 0.0: # 统一在LSTM后加Dropout，无论层数
            self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc_layers = None
        last_fc_output_features = lstm_units # 默认输出层的输入来自LSTM

        if fc_layers_config and len(fc_layers_config) > 0:
            layers = []
            current_input_features = lstm_units
            for units in fc_layers_config:
                layers.append(nn.Linear(current_input_features, units))
                layers.append(nn.ReLU())
                # 如果dropout_rate > 0，可以在每个FC层后也添加Dropout，但任务描述只提到LSTM后
                # if dropout_rate > 0.0:
                #     layers.append(nn.Dropout(dropout_rate))
                current_input_features = units
            self.fc_layers = nn.Sequential(*layers)
            last_fc_output_features = fc_layers_config[-1]

        # 输出层
        self.output_layer = nn.Linear(last_fc_output_features, prediction_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, window_size, input_features)。

        返回:
        torch.Tensor: 预测结果张量，形状为 (batch_size, prediction_steps)。
        """
        # x 形状: (batch_size, window_size, input_features)
        # LSTM的初始隐藏状态和细胞状态默认为零

        lstm_out, (hn, cn) = self.lstm(x)
        # lstm_out 形状: (batch_size, window_size, lstm_units)
        # hn 形状: (lstm_layers, batch_size, lstm_units)
        # cn 形状: (lstm_layers, batch_size, lstm_units)

        # 我们通常取LSTM最后一个时间步的输出
        # 对于batch_first=True, lstm_out[:, -1, :] 是最后一个时间步的输出
        last_time_step_out = lstm_out[:, -1, :]
        # last_time_step_out 形状: (batch_size, lstm_units)

        out = last_time_step_out

        # 应用Dropout层 (如果在__init__中定义了)
        if self.dropout is not None:
            out = self.dropout(out)

        # 通过全连接层 (如果在__init__中定义了)
        if self.fc_layers is not None:
            out = self.fc_layers(out)
        
        # 通过输出层
        predictions = self.output_layer(out)
        # predictions 形状: (batch_size, prediction_steps)

        return predictions

# IV. 训练模块

def train_model(model: LSTMForecastModel, X_train: torch.Tensor, y_train: torch.Tensor,
                epochs: int, batch_size: int, learning_rate: float, device: torch.device,
                model_save_path: str = "lstm_wpp_model.pth") -> List[float]:
    """
    训练LSTM模型。

    参数:
    model (LSTMForecastModel): 实例化的LSTMForecastModel对象。
    X_train (torch.Tensor): 训练数据的特征张量。
    y_train (torch.Tensor): 训练数据的目标张量。
    epochs (int): 训练轮次。
    batch_size (int): 批量大小。
    learning_rate (float): 学习率。
    device (torch.device): PyTorch设备。
    model_save_path (str): 模型保存路径，默认为 "lstm_wpp_model.pth"。

    返回:
    List[float]: 每个epoch的平均训练损失列表。
    """
    # 数据加载器
    # 确保 X_train 和 y_train 与模型在同一设备上。
    # create_sequences 应该已经处理了 X_train, y_train 到 device 的移动。
    # model.to(device) 会处理模型到 device 的移动。
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型配置
    model.to(device) # 确保模型在正确的设备上
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses: List[float] = []

    print(f"\n--- 开始模型训练 ---")
    print(f"训练将在设备: {device} 上进行。")
    print(f"总轮次: {epochs}, 每批大小: {batch_size}, 学习率: {learning_rate}")
    print(f"模型保存路径: {model_save_path}")
    if X_train.shape[0] > 0:
        print(f"训练数据 X形状: {X_train.shape}, y形状: {y_train.shape}")
    else:
        print("警告: 训练数据X为空，无法开始训练。")
        return train_losses # 如果没有数据，则提前返回

    for epoch in range(epochs):
        model.train() # 设置模型为训练模式
        epoch_loss: float = 0.0
        batch_count: int = 0

        for X_batch, y_batch in train_loader:
            # 将批数据移至 device (如果尚未在device上)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad() # 清零梯度
            predictions = model(X_batch) # 前向传播
            loss = criterion(predictions, y_batch) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            epoch_loss += loss.item() #累积批次损失
            batch_count += 1
        
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            train_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.6f}")
        else:
            # 这种情况理论上不应该发生，如果 train_loader 有数据的话
            print(f"Epoch [{epoch+1}/{epochs}], 警告: 没有处理任何批次。")
            train_losses.append(float('inf')) 

    # 早停机制 (占位符)
    # TODO: Implement early stopping based on validation loss when validation set is available.
    # 早停机制通常需要一个验证集来监控模型在未见过数据上的性能。
    # 如果连续N个epochs验证损失没有改善（或达到某个阈值），则停止训练。

    # 模型保存
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"模型状态字典已成功保存至: {model_save_path}")
    except Exception as e:
        print(f"保存模型时发生错误: {e}")
        
    print(f"--- 模型训练结束 ---")

    return train_losses
# V. 评估与预测模块
from sklearn.metrics import mean_squared_error # 确保导入

def evaluate_model(
    model_class: type, # LSTMForecastModel 类本身
    model_params: dict, # 用于实例化模型的参数字典
    test_data_scaled: np.ndarray, # 标准化后的完整测试集 (1D NumPy array)
    scaler: MinMaxScaler, # 用于反标准化的、已在训练集上拟合的scaler
    window_size: int,
    prediction_steps: int,
    device: torch.device,
    model_load_path: str = "lstm_wpp_model.pth"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    评估已训练的LSTM模型在测试集上的性能。

    参数:
    model_class (type): LSTMForecastModel 类。
    model_params (dict): 实例化 LSTMForecastModel 所需的参数字典。
    test_data_scaled (np.ndarray): 标准化后的测试集数据 (1D NumPy array)。
    scaler (MinMaxScaler): 在训练数据上拟合的 MinMaxScaler 对象。
    window_size (int): 输入历史窗口大小。
    prediction_steps (int): 模型预测的未来步数 (尽管我们主要关注第一步)。
    device (torch.device): PyTorch 设备。
    model_load_path (str): 已保存的模型文件路径。

    返回:
    Tuple[np.ndarray, np.ndarray, float]:
        - actual_inverted: 反标准化后的真实值 (t+1)。
        - predicted_inverted: 反标准化后的第一步预测值 (t+1)。
        - mse: 真实值与第一步预测值之间的均方误差。
    """
    print(f"\n--- 开始模型评估 ---")
    print(f"从路径加载模型: {model_load_path}")
    print(f"测试数据长度 (scaled): {len(test_data_scaled)}")

    # 1. 模型加载
    model = model_class(**model_params)
    try:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
    except FileNotFoundError:
        print(f"错误: 模型文件未找到 {model_load_path}")
        # 返回空数组和NaN，或者抛出异常
        return np.array([]), np.array([]), float('nan')
    except Exception as e:
        print(f"加载模型状态字典时发生错误: {e}")
        return np.array([]), np.array([]), float('nan')

    model.to(device)
    model.eval() # 设置为评估模式

    # 2. 滚动预测
    predictions_step1_scaled: List[float] = []
    actuals_step1_scaled: List[float] = []

    # 实际预测从测试集的第 window_size 个点之后开始，即预测索引为 window_size 的点 (对应原始数据的第 window_size+1 个点)
    # 第一个输入窗口是 test_data_scaled[0:window_size]，用于预测 test_data_scaled[window_size]
    # 循环迭代 i 从 0 到 len(test_data_scaled) - window_size - 1
    # 确保至少有一个完整的 (input_seq, target_val) 对可以形成
    if len(test_data_scaled) < window_size + 1:
        print("错误: 测试数据太短，无法进行至少一次预测。")
        return np.array([]), np.array([]), float('nan')

    with torch.no_grad(): # 关闭梯度计算
        for i in range(len(test_data_scaled) - window_size):
            # 形成输入序列
            input_seq_np = test_data_scaled[i : i + window_size]
            # 对应的真实目标值 (t+1)
            target_val_scaled = test_data_scaled[i + window_size]

            # 转换为PyTorch张量
            input_tensor = torch.tensor(input_seq_np, dtype=torch.float32).reshape(1, window_size, 1).to(device)

            # 模型预测 (所有步长)
            predicted_all_steps = model(input_tensor) # 输出形状 (1, prediction_steps)

            # 提取第一步的预测值
            first_step_pred_scaled = predicted_all_steps[0, 0].item()

            predictions_step1_scaled.append(first_step_pred_scaled)
            actuals_step1_scaled.append(target_val_scaled)

    if not predictions_step1_scaled:
        print("警告: 未能生成任何预测。")
        return np.array([]), np.array([]), float('nan')

    # 3. 结果后处理与指标计算
    predictions_step1_scaled_np = np.array(predictions_step1_scaled).reshape(-1, 1)
    actuals_step1_scaled_np = np.array(actuals_step1_scaled).reshape(-1, 1)

    # 反标准化
    # scaler期望的输入是2D的，即使只有一个特征
    predicted_inverted = scaler.inverse_transform(predictions_step1_scaled_np).flatten()
    actual_inverted = scaler.inverse_transform(actuals_step1_scaled_np).flatten()

    # 计算均方误差 (MSE)
    if len(actual_inverted) > 0 and len(predicted_inverted) > 0:
        mse = mean_squared_error(actual_inverted, predicted_inverted)
        print(f"Test MSE (t+1): {mse:.4f}")
    else:
        mse = float('nan')
        print("无法计算MSE，因为反标准化后的数组为空。")
        
    print(f"--- 模型评估结束 ---")
    return actual_inverted, predicted_inverted, mse
# VI. 可视化模块
# import matplotlib.pyplot as plt # 已在文件顶部导入
# from typing import List # 已在文件顶部导入
# import numpy as np # 已在文件顶部导入

def plot_loss_curve(train_losses: List[float], save_path: str = "train_loss_curve.png") -> None:
    """
    绘制并保存训练损失曲线图。

    参数:
    train_losses (List[float]): 每个epoch的训练损失列表。
    save_path (str): 损失曲线图的保存路径。
    """
    if not train_losses:
        print("警告: 训练损失列表为空，无法绘制损失曲线。")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(save_path)
        print(f"训练损失曲线已保存至 {save_path}")
    except Exception as e:
        print(f"保存训练损失曲线时发生错误: {e}")
    plt.close() # 关闭图形，避免在多次调用时重叠或消耗过多内存

def plot_predictions(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    title: str = "Actual vs. Predicted Values (t+1)",
    save_path: str = "predictions_comparison.png"
) -> None:
    """
    绘制并保存实际值与预测值的对比图。

    参数:
    actual_values (np.ndarray): 反标准化后的真实值 (1D NumPy array)。
    predicted_values (np.ndarray): 反标准化后的第一步预测值 (1D NumPy array)。
    title (str): 图表标题。
    save_path (str): 预测对比图的保存路径。
    """
    if actual_values.size == 0 or predicted_values.size == 0:
        print("警告: 实际值或预测值数组为空，无法绘制预测对比图。")
        return
    if actual_values.shape != predicted_values.shape:
        print(f"警告: 实际值 (shape {actual_values.shape}) 和预测值 (shape {predicted_values.shape}) 的形状不匹配，无法绘制。")
        return

    plt.figure(figsize=(15, 6))
    plot_points = min(len(actual_values), 500) # 最多绘制500个点
    
    plt.plot(actual_values[:plot_points], label='Actual Values', color='blue', marker='.', linestyle='-', markersize=4, alpha=0.7)
    plt.plot(predicted_values[:plot_points], label='Predicted Values (t+1)', color='red', linestyle='--', marker='x', markersize=4, alpha=0.7)
    
    plt.title(title)
    plt.xlabel(f'Time Step (or Sample Index, first {plot_points} shown)')
    plt.ylabel(TARGET_COLUMN_NAME if 'TARGET_COLUMN_NAME' in globals() else 'Power Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"预测对比图已保存至 {save_path}")
    except Exception as e:
        print(f"保存预测对比图时发生错误: {e}")
    plt.close() # 关闭图形
if __name__ == '__main__':
    print("--- 开始主程序执行流程 ---")

    # VII. 主程序执行流程
    # 环境配置: 全局参数已在文件顶部定义

    # --- 1. 加载和预处理训练数据 ---
    print("\n--- 1. Loading and Preprocessing Training Data ---")
    train_raw_data_series = load_and_clean_data(TRAIN_FILE_PATH, TARGET_COLUMN_NAME)
    
    if train_raw_data_series.empty:
        print(f"错误: 训练数据加载失败或为空 ({TRAIN_FILE_PATH}). 脚本终止。")
        exit()

    # 将 Series 转换为 DataFrame 以匹配 scale_data 的旧用法（如果需要）
    # 或者直接使用 Series.values
    # 这里假设 load_and_clean_data 返回 Series，而 scale_data 可以处理 Series.values
    train_data_scaled, scaler = scale_data(train_raw_data_series) # scale_data 已更新为接受 Series
    
    if train_data_scaled.size == 0:
        print("错误: 训练数据标准化后为空. 脚本终止。")
        exit()
        
    X_train, y_train = create_sequences(train_data_scaled.flatten(), WINDOW_SIZE, PREDICTION_STEPS, device)

    if X_train.shape[0] == 0:
        print("错误: 未能从训练数据创建序列. 脚本终止。")
        print(f"请检查 WINDOW_SIZE ({WINDOW_SIZE}), PREDICTION_STEPS ({PREDICTION_STEPS}) 和训练数据长度 ({len(train_data_scaled)}).")
        exit()
    
    print(f"训练数据准备完毕: X_train shape {X_train.shape}, y_train shape {y_train.shape}")

    # --- 2. 初始化模型 ---
    print("\n--- 2. Initializing Model ---")
    # 注意: input_features 通常为1（单变量时间序列）
    # dropout_rate 可以设为全局变量，或在此处指定。使用0.1作为示例。
    # 如果 FC_LAYERS 为 None，则模型中不包含额外的全连接层。
    model_params = {
        'input_features': 1,
        'lstm_units': LSTM_UNITS,
        'lstm_layers': LSTM_LAYERS,
        'prediction_steps': PREDICTION_STEPS,
        'fc_layers_config': FC_LAYERS, # 从全局配置获取
        'dropout_rate': 0.1 # 示例值，可以调整或设为全局变量
    }
    model = LSTMForecastModel(**model_params).to(device)
    print(f"模型已初始化，参数: {model_params}")
    print(model)

    # --- 3. 训练模型 ---
    print("\n--- 3. Training Model ---")
    final_model_save_path = "lstm_wpp_final_model.pth"
    train_loss_history = train_model(
        model,
        X_train,
        y_train,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        device,
        model_save_path=final_model_save_path
    )
    if not train_loss_history:
        print("警告: 模型训练未返回损失历史。")

    # --- 4. 加载和预处理测试数据 ---
    print("\n--- 4. Loading and Preprocessing Test Data ---")
    test_raw_data_series = load_and_clean_data(TEST_FILE_PATH, TARGET_COLUMN_NAME)

    if test_raw_data_series.empty:
        print(f"错误: 测试数据加载失败或为空 ({TEST_FILE_PATH}). 后续评估可能失败或不准确。")
        # 决定是否终止或继续（可能导致评估失败）
        # 为简单起见，这里继续，但 evaluate_model 应该能处理空数据
        test_data_scaled_for_eval = np.array([]) # 创建一个空数组
    else:
        # 使用训练时得到的 scaler
        test_data_scaled_for_eval, _ = scale_data(test_raw_data_series, scaler=scaler)
        if test_data_scaled_for_eval.size == 0:
            print("错误: 测试数据标准化后为空。后续评估可能失败。")
            test_data_scaled_for_eval = np.array([])


    # --- 5. 评估模型 ---
    print("\n--- 5. Evaluating Model ---")
    # evaluate_model 需要模型类、模型参数、测试数据、scaler等
    # model_load_path 应与训练时保存的路径一致
    if test_data_scaled_for_eval.size > WINDOW_SIZE : # 确保有足够数据进行至少一次预测
        actual_inverted, predicted_inverted, mse = evaluate_model(
            model_class=LSTMForecastModel, # 传递类本身
            model_params=model_params,     # 传递实例化参数
            test_data_scaled=test_data_scaled_for_eval.flatten(), # 确保是1D
            scaler=scaler,
            window_size=WINDOW_SIZE,
            prediction_steps=PREDICTION_STEPS,
            device=device,
            model_load_path=final_model_save_path
        )
        if mse is not None and not np.isnan(mse):
             print(f"模型评估完成. Test MSE (t+1): {mse:.4f}")
        else:
            print("模型评估未能计算有效的MSE。")
            actual_inverted, predicted_inverted = np.array([]), np.array([]) # 确保定义
    else:
        print(f"测试数据不足 ({len(test_data_scaled_for_eval)} points) 无法进行评估。需要至少 {WINDOW_SIZE + 1} 点。")
        actual_inverted, predicted_inverted, mse = np.array([]), np.array([]), float('nan')


    # --- 6. 可视化结果 ---
    print("\n--- 6. Visualizing Results ---")
    if train_loss_history:
        plot_loss_curve(train_loss_history, save_path="final_train_loss_curve.png")
    else:
        print("无训练损失历史可供绘制。")

    if actual_inverted.size > 0 and predicted_inverted.size > 0:
        plot_predictions(
            actual_inverted,
            predicted_inverted,
            title="Final Model: Actual vs Predicted (t+1) on Test Set",
            save_path="final_predictions_comparison.png"
        )
    else:
        print("无有效评估结果可供绘制预测对比图。")

    print("\n--- Script Finished ---")