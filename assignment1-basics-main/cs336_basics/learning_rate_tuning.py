from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"无效的学习率: {lr}")
        
        # 'defaults' 字典包含了所有超参数
        defaults = {"lr": lr}
        
        # 调用父类的构造函数
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        # closure (闭包) 是一个可选函数，用于重新计算损失
        # 我们这里用不到，但为了 API 兼容性而保留
        loss = None if closure is None else closure()

        # 遍历所有参数组 (param_groups)
        for group in self.param_groups:
            # 从组中获取学习率 (lr)
            lr = group["lr"] 
            
            # 遍历该组中的所有参数 (p)
            for p in group["params"]:
                # 如果梯度不存在 (例如，层被冻结)，则跳过
                if p.grad is None:
                    continue
                
                # 获取与参数 p 关联的“状态” (state)
                # 这是一个字典，用于存储 t (迭代次数) 等信息
                state = self.state[p] 
                
                # 从状态中获取迭代次数 t，如果不存在则默认为 0
                t = state.get("t", 0) 
                
                # 获取损失 L 相对于 p 的梯度 (∇L)
                grad = p.grad.data 
                
                # -----------------------------------------------
                # 核心更新公式 (公式 20)：原地更新权重张量
                p.data -= lr / math.sqrt(t + 1) * grad 
                # -----------------------------------------------
                
                # 将迭代次数 t 递增 1，并存回状态中
                state["t"] = t + 1 

        return loss

    # 创建一个可学习的权重参数
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

# 用我们的新 SGD 优化器来优化这个 'weights' 参数
opt = SGD([weights], lr=1)

# 模拟 100 次训练迭代
for t in range(10):
    # 1. 重置所有可学习参数的梯度
    opt.zero_grad() 
    
    # 2. 计算一个标量损失值 (这里是 L2 范数)
    loss = (weights**2).mean() 
    print(loss.cpu().item())
    
    # 3. 运行反向传播，这会计算梯度 (∇L)
    #    梯度值将被存储在 weights.grad 中
    loss.backward() 
    
    # 4. 运行优化器步骤
    #    这会调用我们实现的 step() 方法，更新 weights.data
    opt.step()