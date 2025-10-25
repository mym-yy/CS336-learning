# train_tokenizer.py
# (请将此文件放置在你的项目根目录, e.g., 'assignment1-basics-main/')

import sys
import pathlib
import json
import os

# --- 1. 设置路径 ---
# SCRIPT_DIR 指向脚本所在的目录:
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# PROJECT_ROOT 指向项目根目录:
PROJECT_ROOT = SCRIPT_DIR.parent
# WORKSPACE_ROOT 指向你的工作区根目录:
# .../CS336/
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
# --- 2. 导入你写好的函数 ---
# 
# !!! 重要 !!!
# 假设你包含所有 'run_...' 函数的文件路径是 'cs336_basics/Bpe_optimize.py'
# 如果你的文件名不同 (比如是 basics.py 或 submission.py),
# 请修改下面这行代码！
try:
    from Bpe_optimize_Parallelized import train_bpe
except ImportError:
    print("="*50)
    print(f"错误：无法导入 'run_train_bpe'")
    sys.exit(1)

# --- 3. 设置训练参数 ---
#
# !!! 重要 !!!
# 你需要自己下载 TinyStories 数据集。
# 假设你已将其下载到 'data/TinyStories_all_data.txt'
# 如果你的路径不同, 请修改下面这个变量！
#
# (例如，Hugging Face 地址: https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories_all_data.txt)
#
INPUT_FILE_PATH = WORKSPACE_ROOT / "CS336data" / "Owt" / "owt_train.txt"

# 你的要求:
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]

# 定义输出文件的路径
OUTPUT_VOCAB_PATH = PROJECT_ROOT / "outputfile" / "openwebtext_vocab.json"
OUTPUT_MERGES_PATH = PROJECT_ROOT / "outputfile" / "openwebtext_merges.txt"


def main():
    print(f"开始 BPE 训练...")
    print(f"  > 输入数据: {INPUT_FILE_PATH}")
    print(f"  > 词汇表大小: {VOCAB_SIZE}")
    print(f"  > 特殊 tokens: {SPECIAL_TOKENS}")

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"\n[错误] 输入文件未找到: {INPUT_FILE_PATH}")
        print("请从 Hugging Face 或其他来源下载 Owt 数据集,")
        print("并将其路径更新到此脚本中的 'INPUT_FILE_PATH' 变量。")
        return

    # --- 4. 运行训练 ---
    # 这会调用你已经写好并优化的 BPE 训练函数
    print("\n正在调用 train_bpe... (这可能需要几分钟时间)")
    vocab, merges = train_bpe(
        input_path=INPUT_FILE_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    print("\n训练完成！")
    print(f"  > 最终词汇表大小: {len(vocab)}")
    print(f"  > 总计合并次数: {len(merges)}")

    # --- 5. 保存结果 ---
    
    # 保存词汇表 (vocab)
    # vocab 是 {int: bytes} 格式. JSON 无法直接保存 bytes.
    # 我们使用 'latin-1' 编码将其转换为字符串，'latin-1' 可以
    # 完美地将 0-255 的所有字节一对一映射为 Unicode 字符。
    print(f"正在保存词汇表到 {OUTPUT_VOCAB_PATH}...")
    json_safe_vocab = {token_id: b.decode('latin-1') for token_id, b in vocab.items()}
    with open(OUTPUT_VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(json_safe_vocab, f, ensure_ascii=False, indent=2)

    # 保存合并规则 (merges)
    # merges 是 list[tuple[bytes, bytes]]
    print(f"正在保存合并规则到 {OUTPUT_MERGES_PATH}...")
    with open(OUTPUT_MERGES_PATH, 'w', encoding='utf-8') as f:
        f.write("# BPE Merges (trained on Owt)\n")
        f.write("# Format: token1 token2\n")
        for b1, b2 in merges:
            # 同样使用 'latin-1' 编码
            s1 = b1.decode('latin-1')
            s2 = b2.decode('latin-1')
            f.write(f"{s1} {s2}\n")

    print("\n全部完成！")


if __name__ == "__main__":
    main()