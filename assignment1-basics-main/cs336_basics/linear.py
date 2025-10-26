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