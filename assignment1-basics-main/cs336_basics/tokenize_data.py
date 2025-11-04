import os
import pathlib
import sys
import numpy as np
import json
from tqdm import tqdm # 用于显示漂亮的进度条 (pip install tqdm)

# --- 导入你的 Tokenizer 类 ---
# (假设它保存在 'tokenizer.py' 中)
try:
    from Bpe_tokenizer import Tokenizer
except ImportError:
    print("="*50)
    print(f"错误：无法导入 'Tokenizer' 类。")
    print("请确保你粘贴的 Tokenizer 类代码保存在 'tokenizer.py' 文件中,")
    print("并且与此脚本在同一个目录下。")
    print("="*50)
    sys.exit(1)

# --- 1. 定义所有路径 ---
# (这部分与你的 train_tokenizer.py 脚本中的路径逻辑保持一致)

# SCRIPT_DIR 指向脚本所在的目录:
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# PROJECT_ROOT 指向项目根目录: (假设此脚本在根目录)
PROJECT_ROOT = SCRIPT_DIR.parent
# WORKSPACE_ROOT 指向你的工作区根目录 (e.g., .../CS336/)
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent.parent

# --- 输入 (你已经有的文件) ---

# 1. 原始文本
TRAIN_TXT_PATH = WORKSPACE_ROOT / "CS336data" / "TinyStories" / "TinyStoriesV2-GPT4-train.txt"
VALID_TXT_PATH = WORKSPACE_ROOT / "CS336data" / "TinyStories" / "TinyStoriesV2-GPT4-valid.txt"

# 2. 训练好的分词器文件 (由 train_tokenizer.py 生成)
VOCAB_PATH = PROJECT_ROOT / "outputfile" / "tinystories_vocab.json"
MERGES_PATH = PROJECT_ROOT / "outputfile" / "tinystories_merges.txt"

# --- 输出 (我们即将创建的文件) ---

# 我们将 .bin 文件保存在一个新目录中，以便 train.py 加载
OUTPUT_DIR = PROJECT_ROOT / "outputfile" / "data_bin"
TRAIN_BIN_PATH = OUTPUT_DIR / "TinyStories" / "train.bin"
VALID_BIN_PATH = OUTPUT_DIR / "TinyStories" / "val.bin"

# -----------------------------------------------------------------------------
# 辅助函数 (!!! 最终修复 !!!)
# -----------------------------------------------------------------------------
def load_tokenizer_from_trained_files(vocab_path, merges_path) -> Tokenizer:
    """
    加载由 train_tokenizer.py 保存的【特定格式】的 vocab 和 merges 文件。
    
    我们必须使用 'latin-1' 来编码字符串，
    以完美逆转 'latin-1' 的解码过程。
    """
    print(f"正在从 {vocab_path} 和 {merges_path} 加载分词器...")

    # 1. 加载词汇表 (JSON, 值为 latin-1 字符串)
    print("  > 加载 vocab.json...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        json_vocab_str_vals = json.load(f)
        
    vocab = {}
    for k_str, v_str in json_vocab_str_vals.items():
        vocab[int(k_str)] = v_str.encode('latin-1')

    # 2. 加载合并规则 (Text 文件, 空格分隔, latin-1 字符串)
    print("  > 加载 merges.txt...")
    merges = []
    with open(merges_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#"): # 跳过注释行
                continue
            
            # --- 这是唯一的修改 ---
            # 旧的、有 bug 的代码:
            # parts = line.strip().split()
            
            # 新的、健壮的代码:
            # 1. 只移除行尾的换行符，而不是所有空白
            # 2. 只在第一个空格处分割一次
            try:
                # 这会将 "a \n" 分割为 ['a', '\n']
                # 这会将 "a b" 分割为 ['a', 'b']
                part1, part2 = line.rstrip('\n').split(' ', 1)
                parts = [part1, part2]
            except ValueError:
                # 处理空行或没有空格的行
                parts = [] 
            # --- 修改结束 ---

            if len(parts) == 2:
                merges.append((parts[0].encode('latin-1'), parts[1].encode('latin-1')))
    
    # 3. 获取特殊 tokens (硬编码, 与 train_tokenizer.py 一致)
    special_tokens = ["<|endoftext|>"]

    # 4. 实例化 Tokenizer
    print(f"  > 实例化 Tokenizer (词表大小: {len(vocab)}, 合并规则: {len(merges)})...")
    return Tokenizer(vocab, merges, special_tokens)

# -----------------------------------------------------------------------------
# 主处理函数
# -----------------------------------------------------------------------------
def process_and_save(tokenizer: Tokenizer, txt_path: pathlib.Path, bin_path: pathlib.Path):
    """
    使用分词器读取一个 .txt 文件, 将其编码, 并保存为 .bin 文件。
    """
    print(f"\n开始处理: {txt_path.name}")
    
    # 检查输入文件是否存在
    if not os.path.exists(txt_path):
        print(f"[错误] 输入文件未找到: {txt_path}")
        print("请确保路径正确。")
        return

    # 我们将分批读取文件并编码，而不是一次性读入所有文本
    # 这可以处理G几十B的大文件，而不会耗尽内存
    
    # 1. 打开 .bin 文件准备写入
    # (我们使用一个临时的 Python 列表来收集 ID)
    all_token_ids = []
    
    # 2. 逐行读取 .txt 文件
    with open(txt_path, 'r', encoding='utf-8') as f:
        # 使用 tqdm 自动显示进度条
        for line in tqdm(f, desc=f"  -> 编码 {txt_path.name}"):
            if line.strip(): # 忽略空行
                # 3. 编码每一行并添加到列表中
                ids = tokenizer.encode(line)
                all_token_ids.extend(ids)

    print(f"编码完成。共找到 {len(all_token_ids):,} 个 tokens。")
    
    # 4. 将 Python 列表转换为 NumPy 数组
    # 我们使用 'np.uint16' (0-65535)，这对于 50k 的词汇表来说足够了
    # 并且 train.py 中的 np.memmap 期望的是这个
    token_array = np.array(all_token_ids, dtype=np.uint16)
    
    # 5. 将 NumPy 数组保存为纯二进制文件
    print(f"正在保存到 {bin_path}...")
    token_array.tofile(bin_path)
    print(f"成功保存 {bin_path.name}！")

# -----------------------------------------------------------------------------
# 运行
# -----------------------------------------------------------------------------
def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载我们训练好的分词器
    if not (os.path.exists(VOCAB_PATH) and os.path.exists(MERGES_PATH)):
        print(f"[错误] 分词器文件未找到 (例如 {VOCAB_PATH})")
        print("请在运行此脚本之前，先成功运行 `train_tokenizer.py`！")
        return
        
    tokenizer = load_tokenizer_from_trained_files(VOCAB_PATH, MERGES_PATH)
    
    # 2. 处理训练文件
    process_and_save(tokenizer, TRAIN_TXT_PATH, TRAIN_BIN_PATH)
    
    # 3. 处理验证文件
    process_and_save(tokenizer, VALID_TXT_PATH, VALID_BIN_PATH)
    
    print("\n--- 预处理第 2 步全部完成！ ---")
    print(f"你的 .bin 文件现在位于: {OUTPUT_DIR}")
    print(f"你现在可以运行 train.py 并使用: --data_dir=\"{OUTPUT_DIR}\"")

if __name__ == "__main__":
    main()