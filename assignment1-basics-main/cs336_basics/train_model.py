import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import argparse
from contextlib import nullcontext
from transformer import *
# -----------------------------------------------------------------------------
# 导入我们从 0 开始实现的所有组件！
# -----------------------------------------------------------------------------
try:
    from transformer import (
        TransformerLM,
        AdamW,
        learning_rate_schedule,
        get_batch,
        cross_entropy,
        save_checkpoint,
        load_checkpoint,
        gradient_clipping
    )
except ImportError:
    print("="*80)
    print("错误：无法从 'transfomer.py' 导入你的实现。")
    print("请确保 'train.py' 和 'transformer.py' 在同一个目录中。")
    print("="*80)
    sys.exit(1)

# 尝试导入 Weights & Biases (wandb) 用于日志记录
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# -----------------------------------------------------------------------------
# 命令行参数解析 (Hyperparameter Configuration)
# -----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="从 0 开始训练 TransformerLM")

    # --- 数据和 IO ---
    parser.add_argument('--data_dir', type=str, default='../outputfile/data_bin/TinyStories', help='存放 train.bin 和 val.bin 的数据目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='存放检查点 (checkpoints) 的目录')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='从这个检查点路径恢复训练')
    
    # --- 模型超参数 (Model Hyperparameters) ---
    parser.add_argument('--vocab_size', type=int, default=10000, help='词汇表大小 (例如 GPT-2)')
    parser.add_argument('--context_length', type=int, default=1024, help='上下文长度 (S)')
    parser.add_argument('--num_layers', type=int, default=12, help='Transformer 块的数量 (L)')
    parser.add_argument('--d_model', type=int, default=768, help='模型维度 (D)')
    parser.add_argument('--num_heads', type=int, default=12, help='注意力头的数量 (H)')
    parser.add_argument('--d_ff', type=int, default=3072, help='FFN 内部维度 (F), 通常 4*D')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE 的 theta 值')

    # --- 训练超参数 (Training Hyperparameters) ---
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小 (B)')
    parser.add_argument('--max_iters', type=int, default=5000, help='总训练迭代步数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪的最大 L2 范数')
    
    # --- 优化器 (Optimizer) 和调度器 (Scheduler) ---
    parser.add_argument('--lr', type=float, default=3e-4, help='最大学习率 (alpha_max)')
    parser.add_argument('--min_lr', type=float, default=3e-5, help='最小学习率 (alpha_min)')
    parser.add_argument('--warmup_iters', type=int, default=100, help='预热 (Warm-up) 步数 (T_w)')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='AdamW 权重衰减 (lambda)')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW beta2 (LLaMA 使用 0.95)')
    parser.add_argument('--eps', type=float, default=1e-8, help='AdamW epsilon')

    # --- 性能和日志 (Performance & Logging) ---
    parser.add_argument('--device', type=str, default='cuda', help='设备 (e.g., "cpu", "cuda", "cuda:0")')
    parser.add_argument('--log_interval', type=int, default=10, help='每隔 N 步打印一次训练日志')
    parser.add_argument('--eval_interval', type=int, default=250, help='每隔 N 步运行一次验证')
    parser.add_argument('--save_interval', type=int, default=1000, help='每隔 N 步保存一次检查点')
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B 项目名称 (如果提供了，则启用日志)')

    return parser.parse_args()

# -----------------------------------------------------------------------------
# 验证 (Evaluation) 辅助函数
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, context_length, batch_size, device, eval_iters=10):
    """
    在验证集上评估模型。
    """
    model.eval() # 将模型设置为评估模式 (例如，关闭 dropout)
    losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch(data, batch_size, context_length, device)
        
        logits = model(inputs)
        
        # 计算损失 (我们需要将 (B, S, V) -> (B*S, V) 来匹配 (B*S))
        B, S, V = logits.shape
        loss = cross_entropy(
            logits.view(B * S, V), 
            targets.view(B * S)
        )
        losses.append(loss.item())
        
    model.train() # 将模型设置回训练模式
    return np.mean(losses)

# -----------------------------------------------------------------------------
# 主训练循环 (Main Training Loop)
# -----------------------------------------------------------------------------
def main():
    args = get_args()
    
    # --- 1. 设置 (Setup) ---
    print(f"Starting training run with args:\n{args}")
    torch.manual_seed(1337) # 为了可复现性
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用, 回退到 CPU。")
        args.device = 'cpu'
    
    # 确保检查点目录存在
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # --- 2. 日志 (Logging) ---
    use_wandb = (args.wandb_project is not None) and HAS_WANDB
    if use_wandb:
        wandb.init(project=args.wandb_project, config=args)
    
    # --- 3. 加载数据 (Data Loading) ---
    print("Loading data...")
    # 使用 np.memmap 高效加载 (假设 token 已被预处理为 uint16)
    train_data = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(args.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # --- 4. 初始化模型 (Model Initialization) ---
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32  # (我们可以稍后使用 autocast)
    )
    model.to(args.device)
    print(f"Model initialized. Parameter count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # --- 5. 初始化优化器 (Optimizer Initialization) ---
    # (使用我们自己实现的 AdamW!)
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # --- 6. 加载检查点 (Checkpoint Loading) ---
    start_iter = 0
    if args.load_checkpoint_path:
        print(f"Resuming from checkpoint: {args.load_checkpoint_path}")
        start_iter = load_checkpoint(args.load_checkpoint_path, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    # --- 7. 训练循环 (The Training Loop) ---
    print(f"Starting training loop from iteration {start_iter}...")
    t0 = time.time()
    
    for iter_num in range(start_iter, args.max_iters):
        
        # --- a. 获取学习率 (LR Schedule) ---
        lr = learning_rate_schedule(
            it=iter_num,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters # T_c 是总迭代步数
        )
        # 将 LR 应用到优化器
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # --- b. 评估 (Evaluation) ---
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            val_loss = evaluate(model, val_data, args.context_length, args.batch_size, args.device)
            print(f"Step {iter_num:6d} | Val Loss {val_loss:.4f}")
            if use_wandb:
                wandb.log({
                    "iter": iter_num,
                    "val_loss": val_loss,
                })

        # --- c. 训练步骤 (Training Step) ---
        
        # 1. 获取批次
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        # 2. 前向传播
        # (我们使用 autocast 来实现混合精度, 如果 GPU 支持的话)
        ctx = torch.autocast(device_type=args.device, dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()
        with ctx:
            logits = model(inputs)
            # 3. 计算损失 (我们需要 B*S, V 和 B*S)
            B, S, V = logits.shape
            loss = cross_entropy(
                logits.view(B * S, V), 
                targets.view(B * S)
            )

        # 4. 反向传播
        loss.backward()
        
        # 5. 梯度裁剪 (Gradient Clipping)
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        
        # 6. 优化器步骤
        optimizer.step()
        
        # 7. 清零梯度
        optimizer.zero_grad(set_to_none=True)
        
        # --- d. 日志 (Logging) ---
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt_ms = (t1 - t0) * 1000 / args.log_interval
            tokens_per_sec = (args.batch_size * args.context_length) / (dt_ms / 1000)
            t0 = t1
            
            print(f"Step {iter_num:6d} | Train Loss {loss.item():.4f} | LR {lr:.2e} | Time {dt_ms:7.2f} ms/step | Tok/s {tokens_per_sec:7.0f}")
            if use_wandb:
                wandb.log({
                    "iter": iter_num,
                    "train_loss": loss.item(),
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec
                })

        # --- e. 保存检查点 (Checkpointing) ---
        if iter_num % args.save_interval == 0 and iter_num > 0:
            save_path = os.path.join(args.checkpoint_dir, f"ckpt_{iter_num}.pth")
            print(f"Saving checkpoint to {save_path}...")
            save_checkpoint(model, optimizer, iter_num, save_path)
            print("Done.")

    print("Training finished.")

if __name__ == "__main__":
    main()