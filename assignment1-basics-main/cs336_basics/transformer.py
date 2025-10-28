import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Linear(nn.Module):
    """
    我们自己实现的线性变换模块 (y = xW^T + b 中的 y = xW^T 部分)。
    
    参数 (Parameters):
        weight (Parameter): 模块的可学习权重，形状为 (out_features, in_features)。
                            这符合 PyTorch 惯例，即存储 W 的转置 (W^T)。
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device=None, 
                 dtype=None) -> None:
        """
        构造一个线性变换模块。

        参数:
            in_features: int - 输入特征的维度
            out_features: int - 输出特征的维度
            device: torch.device | None - 参数存放的设备
            dtype: torch.dtype | None - 参数的数据类型
        """
        
        # 1. 调用超类 (nn.Module) 的构造函数
        # 这是继承 nn.Module 的强制要求
        super().__init__()

        # 2. 存储维度信息，这对于打印模块信息 (repr) 和调试很有用
        self.in_features = in_features
        self.out_features = out_features

        # 3. 构造参数 W^T
        # factory_kwargs 是一个标准做法，用于统一处理 device 和 dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 创建一个正确形状的空张量。
        # 形状为 (out_features, in_features)，符合题目 W^T 的要求。
        weight_tensor = torch.empty((out_features, in_features), **factory_kwargs)
        
        # 4. 将张量包装在 nn.Parameter 中
        # 这会告诉 PyTorch：这是一个模型参数，需要在反向传播时计算梯度，
        # 并且在调用 model.parameters() 或 state_dict() 时应包含它。
        self.weight = Parameter(weight_tensor)

        # 5. 初始化权重
        # 题目要求使用 trunc_normal_ (截断正态分布)
        # 我们使用 PyTorch 提供的初始化函数，并采用 "in-place" (原地) 操作
        # `mean=0.0`, `std=1.0`, `a=-2.0`, `b=2.0` 是 trunc_normal_ 的默认值
        # 注意：在实际应用中，std=1.0 通常太大了，
        # 我们会使用 Kaiming 或 Xavier 初始化来计算更合适的 std。
        # 但在这里，我们严格遵循题目的要求。
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用线性变换。

        参数:
            x: torch.Tensor - 输入张量，形状为 (..., in_features)
        
        返回:
            torch.Tensor - 输出张量，形状为 (..., out_features)
        """
        
        # 6. 执行前向传播
        # 我们存储的是 W^T (out_features, in_features)
        # 数学上需要 W (in_features, out_features)
        # 所以我们使用 self.weight.t() 来获取 W
        # torch.matmul (或 @ 运算符) 会执行正确的矩阵乘法
        # (..., in_features) @ (in_features, out_features) -> (..., out_features)
        
        # 严格遵守 "不要使用 nn.functional.linear" 的要求
        return torch.matmul(x, self.weight.t())

    def extra_repr(self) -> str:
        """
        这是一个辅助函数，用于在 print(model) 时提供更丰富的描述信息
        模仿 PyTorch 内置的 nn.Linear
        """
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=False'
    
class Embedding(nn.Module):
    """
    一个简单的词嵌入查找表。
    
    参数 (Parameters):
        weight (Parameter): 模块的可学习权重 (嵌入矩阵)，
                            形状为 (num_embeddings, embedding_dim)。
    """

    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device=None, 
                 dtype=None) -> None:
        """
        构造一个嵌入模块。

        参数:
            num_embeddings: int - 词汇表的大小 (例如 50257)
            embedding_dim: int - 嵌入向量的维度 (例如 d_model=768)
            device: torch.device | None - 参数存放的设备
            dtype: torch.dtype | None - 参数的数据类型
        """
        
        # 1. 调用超类构造函数
        super().__init__()

        # 2. 存储维度信息
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 3. 构造嵌入矩阵 (W)
        # 形状为 (num_embeddings, embedding_dim)
        # 题目要求 "d_model (embedding_dim) 是最后一维"，这符合要求。
        weight_tensor = torch.empty((num_embeddings, embedding_dim), **factory_kwargs)

        # 4. 包装在 nn.Parameter 中
        self.weight = Parameter(weight_tensor)

        # 5. 初始化权重
        # 同样使用 trunc_normal_
        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        查找给定 token ID 对应的嵌入向量。

        参数:
            token_ids: torch.Tensor - 一个整数张量 (dtype=torch.long),
                                       形状为 (...)。
        
        返回:
            torch.Tensor - 查找到的嵌入向量，
                           形状为 (..., embedding_dim)。
        """
        
        # 6. 执行查找操作
        # 这是 Embedding 模块的精髓。我们不需要矩阵乘法，
        # 而是利用 PyTorch 强大的 "张量索引 (tensor indexing)" 功能。
        
        # PyTorch 允许我们直接使用一个整数张量 (token_ids) 
        # 来索引另一个张量 (self.weight) 的特定维度。
        
        # 示例:
        # self.weight.shape  -> (50000, 768)
        # token_ids.shape   -> (2, 10)  (一个 batch_size=2, seq_len=10 的输入)
        # output.shape      -> (2, 10, 768)
        
        # 严格遵守 "不要使用 nn.functional.embedding" 的要求
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        """
        辅助函数，用于 print(model)
        """
        return f'{self.num_embeddings}, {self.embedding_dim}'

class Rmsnorm(nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)。

    参数 (Parameters):
        weight (Parameter): 可学习的增益 (gamma)，形状为 (d_model,)。
    """
    def __init__(self,
             d_model:int,
             eps:float = 1e-5,
             device=None,
             dtype=None) -> None:
        """
        构造 RMSNorm 模块。

        参数:
            d_model: int - 模型的隐藏维度 (即最后一个维度)
            eps: float - 用于数值稳定性的 Epsilon 值
            device: torch.device | None - 参数存放的设备
            dtype: torch.dtype | None - 参数的数据类型
        """
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device':device, 'dtype': dtype}
        
        # 1. 初始化可学习的增益参数 (gamma)
        # 形状为 (d_model,)
        # 对于归一化层，增益参数 (gamma) 通常初始化为 1
        weight_tensor = torch.ones(d_model, **factory_kwargs)
        
        # 2. 包装在 nn.Parameter 中
        self.weight = Parameter(weight_tensor)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        应用 RMS 归一化。
        参数:
            x: torch.Tensor - 输入张量, 形状为 (..., d_model)
        返回:
            torch.Tensor - 归一化后的张量, 形状相同
        """
        # 1. 存储原始数据类型 (例如 bfloat16)
        original_dtype = x.dtype
        # 2. 向上转换为 float32 进行高精度计算 
        # 形状仍为 (..., d_model)
        x_fp32 = x.to(torch.float32)
        # 3. 计算均方 (Mean Square)
        # torch.mean(input, dim, keepdim)
        # (..., d_model) -> (..., 1)
        mean_sq = torch.mean(x_fp32.pow(2), dim=-1, keepdim=True)
        # 4. 计算均方根 (Root Mean Square)
        # (..., 1)
        # 我们使用 torch.rsqrt() (平方根的倒数) 来提高计算效率
        # rsqrt(x) = 1 / sqrt(x)
        rrms = torch.rsqrt(mean_sq + self.eps)
        # 5. 归一化 (x * 1/RMS) 并应用增益 (gamma)
        # 广播: (..., d_model) * (..., 1) * (d_model,)
        normalized_x = x_fp32 * rrms * self.weight
        # 6. 转换回原始数据类型
        return normalized_x.to(original_dtype)
    
    def extra_repr(self) -> str:
        """
        辅助函数，用于 print(model)
        """
        return f'{self.d_model}, eps={self.eps}'
    
class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络，在 Llama 等模型中使用。

    公式: SwiGLU(x) = W_2( (SiLU(xW_1)) * (xW_3) )
          其中 SiLU(z) = z * sigmoid(z)
    """
    def __init__(self,
             d_model:int,
             d_ff:int ,
             device=None,
             dtype=None) -> None:
        """
        构造一个 SwiGLU 模块。

        参数:
            d_model: int - 模型的隐藏维度
            device: torch.device | None - 参数存放的设备
            dtype: torch.dtype | None - 参数的数据类型
        """
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = d_ff
        factory_kwargs = {'device':device, 'dtype': dtype}
        
        # #1.计算中间维度
        # d_ffn = int((8 / 3) * self.d_model)
        # self.hidden_dim = (d_ffn + 63) // 64 * 64  # 向上取整到 64 的倍数
        
        # 2. 实例化三个 Linear 层 (使用我们自己实现的 Linear 类）
        # W_1
        self.W1 = Linear(d_model, self.hidden_dim, **factory_kwargs)
        # W_3
        self.W3 = Linear(d_model, self.hidden_dim, **factory_kwargs)
        # W_2
        self.W2 = Linear(self.hidden_dim, d_model, **factory_kwargs)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        应用 SwiGLU 变换。

        参数:
            x: torch.Tensor - 输入张量, 形状 (..., d_model)
        
        返回:
            torch.Tensor - 输出张量, 形状 (..., d_model)    
        """
        # 1. 计算门（gate）
        # x_W1 = x @ self.W1.weight.t()
        x_W1 = self.W1(x)
        gate = x_W1 * torch.sigmoid(x_W1)
        
        # 2. 计算投影
        # x_W3 = x @ self.W3.weight.t()
        x_W3 = self.W3(x)
         
        # 3. 逐元素相乘
        # (..., hidden_dim) * (..., hidden_dim)
        gated_up = gate * x_W3
        
        # 4. 最终线性变换
        # output = gated_up @ self.W2.weight.t()
        output = self.W2(gated_up)
        
        return output
   
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, hidden_dim={self.hidden_dim}'
    
class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置嵌入 (Rotary Positional Embedding, RoPE)。
    
    参数 (Buffers):
        cos_cache (torch.Tensor): 预先计算的 cos 值
        sin_cache (torch.Tensor): 预先计算的 sin 值
    """

    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device=None):
        """
        构造 RoPE 模块并创建缓冲区。

        参数:
            theta: float - RoPE 的 Θ 参数 (例如 10000)
            d_k: int - Query 和 Key 向量的维度 (必须是偶数)
            max_seq_len: int - 预先计算的最大序列长度
            device: torch.device | None - 存储缓冲区的设备
        """
        super().__init__()
        # factory_kwargs 用于统一设置 device 和 dtype
        factory_kwargs = {'device': device, 'dtype': torch.float32}
        
        # 确保 d_k 是偶数
        assert d_k % 2 == 0, f"d_k 必须是偶数，但收到了 {d_k}"
        
        # --- 1. 预计算 ---
        # 计算旋转频率 inv_freq = 1.0 / (theta^{(2k / d_k)})
        # k 的索引是 [0, 2, 4, ..., d_k-2]
        # (d_k // 2) 个频率
        k_indices = torch.arange(0, d_k, 2, **factory_kwargs)
        inverse_freq = 1.0 / (theta ** (k_indices / d_k))
        
        # 计算位置 i
        # time_indices表示序列中的位置i，也即[0,1,2,...,max_seq_len-1]
        time_indices = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        
        # 计算角度 angle_{i,k} = i * freq_k (位置 * 频率)
        # 使用外积 (outer product) 得到 (max_seq_len, d_k/2) 的矩阵
        angles = torch.outer(time_indices, inverse_freq)
        
        # 将 (max_seq_len, d_k/2) -> (max_seq_len, d_k)
        # 使得 `[ang_0, ang_1, ...]` 变成 `[ang_0, ang_0, ang_1, ang_1, ...]`
        angles_repeated = angles.repeat_interleave(2, dim=-1)

        # --- 2. 存储 Buffer ---
        # 计算 cos 和 sin
        cos_cache = torch.cos(angles_repeated)
        sin_cache = torch.sin(angles_repeated)
        
        # 注册为 buffer，并设置 persistent=False 
        # 这样它们不会被保存在 state_dict 中 (因为它们可以被重新计算)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        应用 RoPE 旋转。

        参数:
            x (torch.Tensor): 输入张量, 形状 (..., seq_len, d_k)
            token_positions (torch.Tensor): 词元位置, 形状 (..., seq_len)
        
        返回:
            torch.Tensor: 旋转后的张量, 形状 (..., seq_len, d_k)
        """
        
        # --- 1. 从“查找表”中获取 cos 和 sin ---
        # token_positions shape (..., seq_len)
        # self.cos_cache shape (max_seq_len, d_k)
        # cos shape (..., seq_len, d_k)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        
        # --- 2. 构造 x_paired ---
        # x_paired = (-x_1, x_0, -x_3, x_2, ...)
        # x_even: (..., seq_len, d_k/2)
        x_even = x[..., ::2]
        # x_odd: (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]
        
        x_paired_half1 = -x_odd
        x_paired_half2 = x_even
        
        # 堆叠 (..., seq_len, d_k/2, 2)
        stacked = torch.stack([x_paired_half1, x_paired_half2], dim=-1)
        
        # 展平 (..., seq_len, d_k)
        # 得到 (-x_1, x_0, -x_3, x_2, ...)
        x_paired = torch.flatten(stacked, start_dim=-2)
        
        # --- 3. 应用旋转 ---
        # x' = x * cos + x_paired * sin
        x_rotated = x * cos + x_paired * sin
        
        return x_rotated