'''
瓶颈很明显：

每次合并都重新扫描整个 word_counts(O(N * L),N 是词数,L 是平均长度）。
pair_counts 每轮都从零开始重新统计，浪费了很多重复计算。

优化合并步骤 上述程式化示例中,BPE 训练的简单实现速度较慢，因为每次合并时，它都会迭代所有字节对以识别最频繁的对。
然而，每次合并后唯一发生变化的对计数是与合并对重叠的对计数,所需修改的地方与代价差距太大。
因此，可以通过索引所有对的计数并逐步更新这些计数来提高 BPE 训练速度,
而不是显式地迭代每对字节来计算对频率,使用此缓存过程可以获得显著的加速.
因此,此次优化主要是在BPE合并循环处进行

优化建议 from gpt:
1. 用增量更新替代全量扫描
目前每轮都会遍历所有 word_counts 重新统计所有 pair。
可以在合并时只更新受影响的 pair,而不是整个词表。
👉 做法：
用一个 pair_counts 全局字典保存所有 pair 的频率。
每次 update_word_counts 时，维护 pair_counts:
减去旧 pair 的贡献；
加上新 pair 的贡献。
这样统计量是增量更新，不需要每轮 O(NL) 扫描。
这是 HuggingFace tokenizers 库里 BPE 的主要优化点。

2. 用优先队列维护最高频 pair
你现在用 max(pair_counts.items()) 每轮找最大值 → O(M),M 是 pair 数。
可以用 heapq(最大堆/优先队列）维护 (freq, pair)，每次 O(log M) 更新。
更新时只 push 新 pair,过期的 pair 可以懒惰删除（遇到频率不一致时丢弃）。

3. 使用 array/numpy 加速 token 表示
现在 word 是 tuple(int),很多小对象,Python 层开销很大。
可以用 array('I') 或 numpy np.ndarray(dtype=np.int32) 来存储，能减少内存和加快切片。
但这需要改 update_word_counts 的写法。

4. 分批处理文本
对大文本,pretok 一次性读入内存会很慢。
可以分块处理，统计词频时逐块更新。

这个优化文件主要实现了第1点和第2点
'''

from pathlib import Path
import regex as re
from collections import defaultdict , Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#更新词典
def update_word_counts_and_get_delta(word_counts:dict[tuple, int], best_pair:tuple, new_id:int): #best_pair里的tuple 是（int， int）
    new_word_counts = defaultdict(int)
    pair_delta = Counter()
    b1, b2 = best_pair

    for word, freq in word_counts.items():
        #检查这个词当中是否包含需要合并的“best_pair”
        has_pair = False
        for i in range(len(word) - 1):
            if word[i] == b1 and word[i+1] == b2:
                has_pair = True
                break
        if has_pair:    #包含了需要合并的对，才进行处理
            #1.（减法）将该词中的所有旧pair删除
            for i in range(len(word) - 1):
                pair_delta[(word[i], word[i+1])] -= freq

            #2.合并操作
            new_word = []
            skip = False
            for i in range(len(word)):
                if skip:
                    skip = False
                    continue
                if i < len(word) - 1 and word[i] == b1 and word[i+1] == b2:
                    # 合并,需要找到在vocab中对应的str
                    new_word.append(new_id)
                    skip = True
                else:
                    new_word.append(word[i])
            new_word = tuple(new_word)
            new_word_counts[new_word] += freq

            #3.（加法）将该词中的所有新pair加入,将它们的频率加回到pair_delta
            for i in range(len(new_word) - 1):
                pair_delta[(new_word[i], new_word[i+1])] += freq
        
        else:
            #没有包含需要合并的对，直接加入new_word_counts
            new_word_counts[word] += freq

    return new_word_counts, pair_delta

#预分词&创建字典统计词频（预分词，频率）
def pretok(text:str, special_tokens:list[str], special_token_ids:dict):
    tokens = [] #利用tokens存储预分词,这里存的

    #处理特殊字符库
    pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    chunks = re.split(pattern, text)
    #chat-gpt预分词正则
    for chunk in chunks:
        if chunk in special_tokens:
            tokens.append(chunk)
        else:
            for m in re.finditer(PAT, chunk):
                tokens.append(m.group(0))
    
    word_counts = defaultdict(int)
    for tok in tokens: #要把special_token独立出来
        if tok in special_token_ids:
            key = (special_token_ids[tok],)
        else:
            key = tuple(tok.encode("utf-8")) #比如word_counts[i] = (104,101,108,108,111),这里tok就是‘hello’
        word_counts[key] += 1

    return word_counts

###############################################################################
def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
#input_path: str: 训练数据文件的路径。
#vocab_size: int: 最终词汇表的最大大小。
#special_tokens: list[str]: 需要添加到词汇表中的特殊字符串列表。    

#词表初始化    
    vocab = {} #(int,bytes) 
    vocab = {i:bytes([i]) for i in range(256)}          # 单字节 0-255
    special_token_ids = {}#特殊词对应的vocab词表id
    for tok in special_tokens:
        tok_id = len(vocab)
        vocab[tok_id] = tok.encode("utf-8")
        special_token_ids[tok] = tok_id
#预分词和统计词频
    #获取文本
    file_path = Path(input_path)
    text = file_path.read_text(encoding='utf-8')

    word_counts = pretok(text, special_tokens, special_token_ids)

    pair_counts = Counter()
    for word, freq in word_counts.items():  #遍历每个词
        for j in range(len(word) - 1):  #遍历每个word tuple中的前后两个元素
            pair = (word[j], word[j+1])
            pair_counts[pair] += freq #pair_counts（tuple，int），tuple是（int，int）

    #BPE合并循环
    merges = []
    for i in range(vocab_size - len(vocab)):
        if not pair_counts:
            break

        best = max(
            pair_counts.items(), 
            key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]])
        )[0]#best为频率最高的一对bytes分别对应的int
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        new_mer = (vocab[best[0]], vocab[best[1]])
        merges.append(new_mer)

        word_counts, delta = update_word_counts_and_get_delta(word_counts, best, new_id)
        
        # 将变化量应用到我们持久化的 pair_counts 上
        pair_counts.update(delta)

    return vocab, merges
#vocab: dict[int, bytes]: 从整数ID到其对应字节的词汇表映射。
#merges: list[tuple[bytes, bytes]]: 按顺序记录的BPE合并操作列表。