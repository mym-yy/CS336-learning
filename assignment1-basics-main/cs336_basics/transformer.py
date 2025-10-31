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
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    沿指定维度计算数值稳定的 softmax。

    参数:
        x (torch.Tensor): 输入张量 (logits)
        dim (int): 要应用 softmax 的维度
    
    返回:
        torch.Tensor: 归一化后的概率张量，形状与 x 相同
    """
    # # 1. 减去最大值以保证数值稳定性
    # #    torch.max(x, dim) 会返回 (values, indices)
    # #    我们只需要 values。
    # #    !! 关键: 必须设置 keepdim=True !!
    # #    这使得 max_val 的形状与 x 兼容 (例如 x 是 [B, S, D], max_val 是 [B, S, 1])
    # #    这样 "x - max_val" 才能正确广播 (broadcasting)。
    # max_val = torch.max(x, dim=dim, keepdim=True).values
    # x_shifted = x - max_val
    
    # # 2. 计算指数
    # #    由于 x_shifted 中的最大值现在是 0，exps 中的最大值就是 e^0 = 1
    # #    这就避免了上溢 (overflow) 变成 inf
    # exps = torch.exp(x_shifted)
    
    # # 3. 计算分母 (所有指数的总和)
    # #    !! 关键: 这里也必须设置 keepdim=True !!
    # #    这使得分母的形状与 exps 兼容 (例如 [B, S, 1])
    # #    以便下一步的除法能够正确广播。
    # sum_of_exps = torch.sum(exps, dim=dim, keepdim=True)
    
    # # 4. 相除得到最终概率
    # probabilities = exps / sum_of_exps
    """
    在写transformerBlock时进行第一次修改
    需要提高精度
    沿指定维度计算数值稳定的 softmax。
    (在高精度 float32 中执行计算以保证稳定性
    """
    # 1. 存储原始数据类型 (例如 bfloat16)
    original_dtype = x.dtype
    
    # 2. 向上转换为 float32 进行高精度计算 (!! 关键步骤 !!)
    x_fp32 = x.to(torch.float32)

    # 3. 在 float32 上执行所有计算
    max_val = torch.max(x_fp32, dim=dim, keepdim=True).values
    x_shifted = x_fp32 - max_val
    exps = torch.exp(x_shifted)
    sum_of_exps = torch.sum(exps, dim=dim, keepdim=True)
    probabilities = exps / sum_of_exps

    return probabilities.to(original_dtype)

def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
      计算缩放点积注意力。
    
    公式: Attention(Q, K, V) = softmax( (QK^T / sqrt(d_k)) + M ) @ V

    参数:
        q (torch.Tensor): Queries 张量  形状 (..., seq_len_q, d_k)
        k (torch.Tensor): Keys 张量  形状 (..., seq_len_k, d_k)
        v (torch.Tensor): Values 张量  形状 (..., seq_len_k, d_v)
        mask (torch.Tensor | None): 布尔掩码.
                                    形状 (..., seq_len_q, seq_len_k)
                                    True = 保留, False = 屏蔽 (设为 -inf)
                                    (注意：... 维度必须与 QK^T 的 ... 维度兼容)
    返回:
        torch.Tensor: 注意力输出张量, 形状 (..., seq_len_q, d_v)
    """
    # 1. 获取 d_k (K 的最后一个维度)
    d_k = k.size(-1)
    
    # 2. 计算 QK^T (缩放前的分数)
    # q 形状: (..., seq_len_q, d_k)
    # k.transpose(-2, -1) 形状: (..., d_k, seq_len_k)
    # scores 形状: (..., seq_len_q, seq_len_k)
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    # 3. 应用缩放 (Scale)
    # 使用 torch.tensor 来确保在正确设备上
    scale_factor = torch.sqrt(torch.tensor(d_k, dtype=q.dtype, device=q.device))
    scaled_scores = scores / scale_factor
    
    # 4. 应用掩码 (Mask) - 核心技巧
    if mask is not None:
        # 我们需要在 `mask == False` 的地方填充一个非常小的数 (负无穷)
        # `torch.finfo` 提供了该 dtype 能表示的最小值
        # .masked_fill_ (in-place) 或 .masked_fill (out-of-place)
        scaled_scores = scaled_scores.masked_fill(
            mask == False, torch.finfo(scaled_scores.dtype).min
        )
        
    # 5. 应用 Softmax (沿最后一个维度)
    # `attention_weights` 形状: (..., seq_len_q, seq_len_k)
    # (使用我们自己实现的 softmax 函数)
    attention_weights = softmax(scaled_scores, dim=-1)
    
    # 6. 乘以 V (Value)
    # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v)
    # output 形状: (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, v)
    
    return output

class CausalMultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力 (Causal Multi-Head Self-Attention) 模块。
    
    它首先将输入 `x` 投影到 Q, K, V，然后将它们分成 `num_heads` 个头。
    接着，它应用 RoPE (如果提供了) 和缩放点积注意力 (带因果掩码)。
    最后，它将所有头的输出合并并通过一个最终的线性层。
    
    参数 (Parameters):
        w_q (Linear): 投影到 Q (所有头) 的大线性层
        w_k (Linear): 投影到 K (所有头) 的大线性层
        w_v (Linear): 投影到 V (所有头) 的大线性层
        w_o (Linear): 最终的输出投影层
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 rope: RotaryPositionalEmbedding | None = None,
                 device= None,
                 dtype= None) -> None:
        """
        构造一个 CausalMultiHeadSelfAttention 模块。

        参数:
            d_model: int - 模型的隐藏维度
            num_heads: int - 注意力头的数量 (h)
            device: torch.device | None - 参数存放的设备
            dtype: torch.dtype | None - 参数的数据类型
        """
        super().__init__()
        # 1.存储核心函数
        self.d_model = d_model
        self.num_heads = num_heads

        # 2. 确保 d_model 可以被 num_heads 整除
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        # 3. 计算每个头的维度 (d_k, d_v)
        # 遵循 Vaswani 等人 [2017] 的论文, d_k = d_v = d_model / h
        self.d_k = d_model // num_heads
        self.d_v = self.d_k # d_v = d_k
        self.rope = rope

        factory_kwargs = {'device':device, 'dtype':dtype}

        # 4. 创建 "投影" 层 (Projection Layers)
        # 我们使用 3 个 "大" 线性层来一次性计算所有头的 Q, K, V       
        # W_q: (d_model) -> (h * d_k) = (d_model)
        self.w_q = Linear(d_model, d_model, **factory_kwargs)
        # W_k: (d_model) -> (h * d_k) = (d_model)
        self.w_k = Linear(d_model, d_model, **factory_kwargs)
        # W_v: (d_model) -> (h * d_v) = (d_model)
        self.w_v = Linear(d_model, d_model, **factory_kwargs)
        # W_o: (h * d_v) = (d_model) -> (d_model)
        self.w_o = Linear(d_model, d_model, **factory_kwargs)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        执行因果多头自注意力的前向传播。

        参数:
            x (torch.Tensor): 输入张量, 形状 (batch_size, seq_len, d_model)
        
        返回:
            torch.Tensor: 注意力输出, 形状 (batch_size, seq_len, d_model)
        """
        # 获取输入的形状信息
        # B = batch_size, S = seq_len, D = d_model
        (B, S, D) = x.shape

        # --- 1. 计算 Q, K, V 的 "大投影" ---
        # (B, S, D) -> (B, S, h*d_k) = (B, S, D)
        q_proj = self.w_q(x)
        k_proj = self.w_k(x)
        v_proj = self.w_v(x)
        
        # --- 2. 将 "大投影" 分割成 "多头" ---
        # (B, S, D) -> (B, S, h, d_k) -> (B, h, S, d_k)
        
        # q_multihead 形状: (B, h, S, d_k)
        q_multihead = q_proj.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # k_multihead 形状: (B, h, S, d_k)
        k_multihead = k_proj.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # v_multihead 形状: (B, h, S, d_v)
        v_multihead = v_proj.view(B, S, self.num_heads, self.d_v).transpose(1, 2)
        
        # (注意：RoPE 会在这一步被应用到 q_multihead 和 k_multihead 上,
        # 在这里应用 RoPE (如果提供了) ---
        if self.rope is not None:
            # 确保 token_positions 也被提供了
            assert token_positions is not None, \
                "RoPE 模块存在, 但 token_positions 未提供"
            
            # RoPE 期望 (..., S, d_k) 形状
            # q_multihead 是 (B, h, S, d_k), token_positions 是 (B, S)
            
            # token_positions (B, S) -> (B, 1, S) -> (B, h, S)
            # 以便广播到 (B, h, S, d_k)
            pos_emb = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1)
            
            q_multihead = self.rope(q_multihead, pos_emb)
            k_multihead = self.rope(k_multihead, pos_emb)
        # --- RoPE 应用结束 ---

        # --- 3. 创建 "因果掩码" (Causal Mask) ---
        # 这是一个形状为 (S, S) 的下三角矩阵
        # `True` = 保留 (j <= i), `False` = 屏蔽 (j > i)
        # `torch.tril` (下三角) 正好做这个
        causal_mask = torch.tril(
            torch.ones((S, S), device=x.device, dtype=torch.bool)
        )
        # 这个 (S, S) 的掩码会自动广播到 (B, h, S, S)
        
        # --- 4. 应用 "缩放点积注意力" ---
        # (B, h, S, d_k) @ (B, h, S, d_v) -> (B, h, S, d_v)
        # (我们调用自己实现的函数)
        attn_output = scaled_dot_product_attention(
            q_multihead, k_multihead, v_multihead, mask=causal_mask
        )
        
        # --- 5. "合并" (Concatenate) 所有头的输出 ---
        # (B, h, S, d_v) -> (B, S, h, d_v) [转置]
        attn_output_transposed = attn_output.transpose(1, 2)
        
        # (B, S, h, d_v) -> (B, S, h*d_v) = (B, S, D) [合并]
        # .contiguous() 确保张量在内存中是连续的，以便 .view() 可以工作
        attn_output_concat = attn_output_transposed.contiguous().view(B, S, D)
        
        # --- 6. 应用 "最终输出投影" (W_o) ---
        # (B, S, D) -> (B, S, D)
        final_output = self.w_o(attn_output_concat)
        
        return final_output

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}'
    
class TransformerBlock(nn.Module):
    """
    一个预规范 (Pre-Norm) 的 Transformer 模块 (遵照 §3.5 和 图 2)
    
    它包含两个子层:
    1. 多头自注意力 (Multi-Head Self-Attention)
    2. 前馈网络 (Feed-Forward Network)
    
    在每个子层中，都使用 "RMSNorm -> 主操作 -> 残差连接" 的流程。
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 max_seq_len: int,
                 theta: int,
                 device = None,
                 dtype = None) -> None:
        """
        初始化 Transformer 模块

        参数:
        d_model: 模型的维度 (int)
        num_heads: 多头注意力的头的数量 (int)
        d_ff: 前馈网络 (FFN) 的内部维度 (int)
        device: torch.device | None - 参数存放的设备
        dtype: torch.dtype | None - 参数的数据类型
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        factory_kwargs = {'device': device, 'dtype': dtype}

        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_k,
            max_seq_len=max_seq_len,
            device=device
        )

        # --- 第一个子层：MHA ---
        # 1. 第一个子层的归一化 (Pre-Normalization)
        self.attn_norm = Rmsnorm(d_model=d_model, **factory_kwargs)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=self.rope,  # 基础测试不需要 RoPE
            **factory_kwargs
        )
        # 2. 前馈网络子层 (FFN Sub-layer)
        self.ffn_norm = Rmsnorm(d_model=d_model, **factory_kwargs)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            **factory_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 Transformer 块的前向传播。
        
        公式:
        y = x + MHA(RMSNorm(x))
        z = y + FFN(RMSNorm(y))

        参数:
            x (torch.Tensor): 输入张量, 形状 (batch_size, seq_len, d_model)
        
        返回:
            torch.Tensor: 块的输出, 形状 (batch_size, seq_len, d_model)
        """
        (B, S, D) = x.shape
        # (S,)
        token_positions_seq = torch.arange(S, device=x.device)
        # (S,) -> (1, S) -> (B, S)
        token_positions = token_positions_seq.unsqueeze(0).expand(B, -1)

        # --- 第一个子层：MHA + 残差连接 ---
        # 对应公式: y = x + MultiHeadSelfAttention(RMSNorm(x))
        y = x + self.attn(self.attn_norm(x), token_positions=token_positions)

        # --- 第二个子层：FFN + 残差连接 ---
        # z = y + FFN(RMSNorm(y))
        z = y + self.ffn(self.ffn_norm(y))

        return z
