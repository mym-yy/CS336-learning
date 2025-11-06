import torch
import argparse
import time
import os
import sys
import torch.nn.functional as F
from torch.nn import Module

# --- 1. 导入你的核心组件 ---
try:
    # 导入你的 Transformer 模型
    from transformer import TransformerLM 
    
    # 导入你的 Tokenizer
    from Bpe_tokenizer import Tokenizer
    
    # 导入你修复过的 Tokenizer 加载器
    # (这个函数在 tokenize_data.py 中，你可能需要把它复制过来)
    from tokenize_data import load_tokenizer_from_trained_files 

except ImportError:
    print("="*80)
    print("错误: 无法导入 'TransformerLM', 'Tokenizer', 或 'load_tokenizer_from_trained_files'。")
    print("请确保 'generate.py', 'transformer.py', 'Bpe_tokenizer.py' 和 'tokenize_data.py' 都在同一个目录中。")
    print("="*80)
    sys.exit(1)

# --- 2. 你的 Generate 函数 (来自我们之前的讨论) ---
@torch.no_grad()
def generate(
    model: Module, 
    tokenizer: object,
    prompt: str, 
    max_new_tokens: int, 
    temperature: float = 1.0, 
    top_p: float = 0.9
) -> str:
    """
    从语言模型解码的函数。
    """
    
    # --- 准备工作 ---
    model.eval() # 将模型设为评估模式
    try:
        eot_id = tokenizer.special_tokens["<|endoftext|>"]
    except Exception:
        # (你需要根据你的 tokenizer 调整)
        eot_id = tokenizer.vocab_inv[b"<|endoftext|>"] 
        
    device = next(model.parameters()).device
    
    # --- 编码提示 ---
    print(f"--- 开始生成 (提示: '{prompt}') ---")
    try:
        token_ids = tokenizer.encode(prompt)
    except KeyError as e:
        print(f"错误: 提示 '{prompt}' 中包含词汇表未知的 token。")
        print(f"具体的错误是: {e}")
        return ""
        
    # --- 自回归循环 ---
    for i in range(max_new_tokens):
        current_ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        
        # 1. 获取 Logits
        logits_from_model = model(current_ids_tensor)
        next_token_logits = logits_from_model[0, -1, :]
        
        # 2. (可交付成果 3) 应用温度
        if temperature == 0.0:
            next_token_id = torch.argmax(next_token_logits).item()
        else:
            scaled_logits = next_token_logits / temperature
            
            # 3. (可交付成果 4) 应用 Top-p
            if top_p > 0.0 and top_p < 1.0:
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled_logits[indices_to_remove] = -float('Inf')
            
            final_probs = F.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(final_probs, num_samples=1).item()

        # 4. (可交付成果 1) 检查停止条件
        if next_token_id == eot_id:
            print("\n--- (遇到 <|endoftext|>) ---")
            break
            
        # 5. (可交付成果 1) 追加
        token_ids.append(next_token_id)
        
        # 实时打印输出
        print(tokenizer.decode([next_token_id]), end="", flush=True)

    if i == max_new_tokens - 1:
        print("\n--- (达到 max_new_tokens) ---")
        
    # --- 循环结束：解码并返回 ---
    return tokenizer.decode(token_ids)

# --- 3. 主函数 (加载并运行) ---
def main():
    parser = argparse.ArgumentParser(description="从训练好的模型生成文本")
    
    # --- 必填参数 ---
    parser.add_argument('--checkpoint_path', type=str, required=True, help='指向 .pth 检查点文件的路径')
    
    # --- 生成参数 ---
    parser.add_argument('--prompt', type=str, default="Once upon a time,", help='初始提示')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='要生成的最大 token 数量')
    parser.add_argument('--temperature', type=float, default=0.7, help='Softmax 温度 (0.0 = 贪心)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (核) 采样 (1.0 = 禁用)')
    
    # --- 模型参数 (!! 必须匹配你的训练 !!) ---
    # 这些是 17M 模型的参数
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer 块的数量 (L)')
    parser.add_argument('--d_model', type=int, default=384, help='模型维度 (D)')
    parser.add_argument('--num_heads', type=int, default=6, help='注意力头的数量 (H)')
    
    # --- 词汇表和数据路径 (!! 必须匹配你的训练 !!) ---
    parser.add_argument('--vocab_size', type=int, default=10000, help='你的词汇表大小 (来自 BPE 训练)')
    parser.add_argument('--vocab_path', type=str, default='../outputfile/tinystories_vocab.json', help='你的 vocab.json 路径')
    parser.add_argument('--merges_path', type=str, default='../outputfile/tinystories_merges.txt', help='你的 merges.txt 路径')

    args = parser.parse_args()

    # --- 1. 加载分词器 ---
    print(f"正在加载分词器从: {args.vocab_path} 和 {args.merges_path}")
    tokenizer = load_tokenizer_from_trained_files(args.vocab_path, args.merges_path)
    
    # --- 2. 加载模型 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用设备: {device}")

    # (A) 实例化模型 (使用你的参数)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=1024,  # (生成时 context_length 可以是任意的，但模型必须知道)
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4, # 假设 d_ff = 4 * d_model
        theta=10000.0,
        device=device,
        dtype=torch.float32
    )
    model.to(device)
    
    # (B) 加载检查点权重
    print(f"正在从 {args.checkpoint_path} 加载模型权重...")
    try:

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # 2. 从字典中提取 *模型* 的 state_dict
        #    (新的错误日志告诉我们键是 'model_state_dict')
        model_weights = checkpoint['model_state_dict']
        
        # 3. 加载模型的 state_dict
        model.load_state_dict(model_weights)
        
    except Exception as e:
        print(f"错误: 无法加载模型权重。")
        print(f"请确保 '{args.checkpoint_path}' 是一个有效的检查点文件，")
        print(f"并且模型参数 (L={args.num_layers}, H={args.num_heads}, D={args.d_model}) 与检查点完全匹配。")
        print(f"具体错误: {e}")
        sys.exit(1)

    # --- 3. 运行生成 ---
    # (为了可复现性，设置一个种子)
    torch.manual_seed(42)
    
    start_time = time.time()
    generated_text = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p
    )
    end_time = time.time()
    
    print("\n--- 最终输出 ---")
    print(generated_text)
    print("="*80)
    print(f"总计生成 {len(tokenizer.encode(generated_text)) - len(tokenizer.encode(args.prompt))} 个新 token，")
    print(f"耗时 {end_time - start_time:.2f} 秒。")

if __name__ == "__main__":
    main()