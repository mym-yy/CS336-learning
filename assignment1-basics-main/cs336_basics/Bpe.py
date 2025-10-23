from pathlib import Path
import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#更新词典
def update_word_counts(word_counts:dict[tuple, int], best_pair:tuple, new_id:int): #best_pair里的tuple 是（int， int）
    new_word_counts = {}
    b1, b2 = best_pair

    for word, freq in word_counts.items():
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
        new_word_counts[new_word] = new_word_counts.get(new_word, 0) + freq #get(new_word, 0)用法，找到了new_word就返回字典中这个key对应的val，没找到就返回0
    return new_word_counts

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

    #BPE合并循环
    merges = []
    for i in range(vocab_size - len(vocab)):
    #每次循环都将出现频率最高的（一个word中的前后两个元素）合并
        pair_counts = defaultdict(int)  #pair_counts 用来做(（b'b'，b'c'），频率)的字典，（b'b'，b'c'）是（int，int），用整数id来代表字符
        for word, freq in word_counts.items():  #遍历每个词
            for j in range(len(word) - 1):  #遍历每个word tuple中的前后两个元素
                pair = (word[j], word[j+1])
                pair_counts[pair] += freq #pair_counts（tuple，int），tuple是（int，int）

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

        word_counts = update_word_counts(word_counts, best,  new_id)

    return vocab, merges
#vocab: dict[int, bytes]: 从整数ID到其对应字节的词汇表映射。
#merges: list[tuple[bytes, bytes]]: 按顺序记录的BPE合并操作列表。