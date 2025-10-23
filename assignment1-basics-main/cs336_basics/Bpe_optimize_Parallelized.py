import sys
import pathlib
import json
import os
import regex as re
from collections import defaultdict, Counter
from pathlib import Path
import multiprocessing  # <--- 导入多进程库

# --- 你的 PAT 和 update_word_counts_and_get_delta 函数 (无需修改) ---
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def update_word_counts_and_get_delta(word_counts:dict[tuple, int], best_pair:tuple, new_id:int):

    new_word_counts = defaultdict(int)
    pair_delta = Counter()
    b1, b2 = best_pair
    for word, freq in word_counts.items():
        has_pair = False
        for i in range(len(word) - 1):
            if word[i] == b1 and word[i+1] == b2:
                has_pair = True
                break
        if has_pair:
            for i in range(len(word) - 1):
                pair_delta[(word[i], word[i+1])] -= freq
            new_word = []
            skip = False
            for i in range(len(word)):
                if skip:
                    skip = False
                    continue
                if i < len(word) - 1 and word[i] == b1 and word[i+1] == b2:
                    new_word.append(new_id)
                    skip = True
                else:
                    new_word.append(word[i])
            new_word = tuple(new_word)
            new_word_counts[new_word] += freq
            for i in range(len(new_word) - 1):
                pair_delta[(new_word[i], new_word[i+1])] += freq
        else:
            new_word_counts[word] += freq
    return new_word_counts, pair_delta

# --- 1. 修改 pretok，使其返回 Counter ---
# Counter 在并行聚合时性能更好
def pretok(text:str, special_tokens:list[str], special_token_ids:dict) -> Counter:
    tokens = []
    pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    chunks = re.split(pattern, text)
    for chunk in chunks:
        if chunk in special_tokens:
            tokens.append(chunk)
        else:
            for m in re.finditer(PAT, chunk):
                tokens.append(m.group(0))
    
    # 将 defaultdict(int) 改为 Counter()
    word_counts = Counter()
    for tok in tokens:
        if tok in special_token_ids:
            key = (special_token_ids[tok],)
        else:
            key = tuple(tok.encode("utf-8"))
        word_counts[key] += 1

    return word_counts

# --- 2. 这是一个新的 "worker" 函数 ---
# 它会在一个单独的 CPU 核心上运行
# 它接收的参数由主进程的 `starmap` 提供
def pretok_worker(text_chunk: str, special_tokens: list[str], special_token_ids: dict) -> Counter:
    """单个进程的工作函数，调用 pretok 处理一个文本块"""
    return pretok(text_chunk, special_tokens, special_token_ids)

# --- 3. 这是一个新的 "数据块读取器" (生成器) ---
def read_file_in_chunks(file_path: str, special_token_pattern: re.Pattern, chunk_size_bytes: int = 64 * 1024 * 1024):
    """
    按块读取文件，并确保在特殊标记的边界处分割。
    这是一个生成器，它会逐块 'yield' (产生) 文本。
    """
    buffer = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                # 读取一大块数据
                data = f.read(chunk_size_bytes)
                if not data:
                    # 文件读完，处理最后的 buffer
                    if buffer:
                        yield buffer
                    break
                
                # 将新数据和上一轮的 "遗留" buffer 合并
                buffer += data
                
                # 找到 buffer 中最后一个特殊标记
                matches = list(re.finditer(special_token_pattern, buffer))
                
                if matches:
                    # 找到最后一个匹配项的结束位置
                    last_match_end = matches[-1].end()
                    
                    # 发送直到这个位置的 "安全" 数据块
                    chunk_to_yield = buffer[:last_match_end]
                    # 保留之后的数据到下一轮
                    buffer = buffer[last_match_end:]
                    
                    yield chunk_to_yield
                else:
                    # 这个 64MB 的块里连一个特殊 token 都没有
                    # (在 TinyStories 里几乎不可能, <|endoftext|> 很多)
                    # 我们可以选择在最后一个换行符处分割，或者继续读取
                    # 为了安全，我们继续读取，直到 buffer 变得过大
                    if len(buffer) > 2 * chunk_size_bytes:
                        # 强行分割 (兜底策略)
                        print("Warning: Chunking at non-special-token boundary.", file=sys.stderr)
                        yield buffer
                        buffer = ""

    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        # 确保即使出错，也处理掉剩余的 buffer
        if buffer:
            yield buffer


###############################################################################
# --- 4. 重构 train_bpe 函数以使用并行化 ---
def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
    
    # 词表初始化 (不变)
    vocab = {i:bytes([i]) for i in range(256)}
    special_token_ids = {}
    for tok in special_tokens:
        tok_id = len(vocab)
        vocab[tok_id] = tok.encode("utf-8")
        special_token_ids[tok] = tok_id

    # --- 并行化预分词和统计词频 (核心修改) ---
    print(f"开始并行化预分词... (输入: {input_path})")
    
    # 确定使用多少个 CPU 核心
    num_workers = multiprocessing.cpu_count()
    print(f"  > 正在启动 {num_workers} 个 CPU 核心...")

    # 1. 准备 worker 需要的固定参数
    # 编译用于分割的正则表达式
    special_pattern = re.compile("(" + "|".join(map(re.escape, special_tokens)) + ")")
    
    # 2. 创建一个进程池
    total_word_counts = Counter()
    with multiprocessing.Pool(processes=num_workers) as pool:
        
        # 3. 创建一个 "数据块读取器" (生成器)
        chunk_iterator = read_file_in_chunks(input_path, special_pattern)
        
        # 4. 准备 worker 参数
        # 我们把 (文本块, 固定参数1, 固定参数2) 组合起来
        worker_args_iterator = (
            (chunk, special_tokens, special_token_ids) for chunk in chunk_iterator
        )

        # 5. 执行并行处理
        # pool.starmap 会自动将 worker_args_iterator 中的参数
        # 分发给 pretok_worker 函数，并阻塞直到所有任务完成
        print(f"  > 正在将数据分块并发送给 {num_workers} 个核心... (这需要一点时间)")
        results = pool.starmap(pretok_worker, worker_args_iterator)
        print(f"  > 所有核心计算完毕，正在聚合结果...")

        # 6. 聚合所有核心返回的 Counter 结果
        for word_count in results:
            total_word_counts.update(word_count)

    # 转换回你原来的 defaultdict 格式
    word_counts = defaultdict(int, total_word_counts)
    print("预分词和词频统计完成！")
    # --- 并行化结束 ---

    # --- BPE 合并循环 (这部分不变，它很快，不需要并行) ---
    print("开始 BPE 合并循环...")
    pair_counts = Counter()
    for word, freq in word_counts.items():
        for j in range(len(word) - 1):
            pair = (word[j], word[j+1])
            pair_counts[pair] += freq

    merges = []
    # 进度条
    from tqdm import tqdm 
    pbar = tqdm(total=vocab_size - len(vocab), desc="BPE Merges")

    for i in range(vocab_size - len(vocab)):
        if not pair_counts:
            break

        best = max(
            pair_counts.items(), 
            key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]])
        )[0]
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        new_mer = (vocab[best[0]], vocab[best[1]])
        merges.append(new_mer)

        word_counts, delta = update_word_counts_and_get_delta(word_counts, best, new_id)
        
        pair_counts.update(delta)
        pbar.update(1)
    
    pbar.close()
    print("BPE 合并循环完成！")

    return vocab, merges