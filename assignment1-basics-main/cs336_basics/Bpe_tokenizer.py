import re
import regex
import json
import base64
from typing import (
    Iterable, 
    Iterator, 
    Optional, 
    List, 
    Dict, 
    Tuple
)

class Tokenizer:
    """
    一个 BPE 分词器，实现了 encode 和 decode 方法，并支持特殊 token。
    """ 
    def __init__(self, 
                 vocab: Dict[int, bytes], 
                 merges: List[Tuple[bytes, bytes]], 
                 special_tokens: Optional[List[str]] = None):
        """
        从给定的词汇表、合并列表和（可选）特殊 token 构建分词器。

        参数:
            vocab: dict[int, bytes] - 从 token ID 到其 bytes 表示的映射。
            merges: list[tuple[bytes, bytes]] - BPE 合并规则的有序列表。
            special_tokens: list[str] | None = None - 要添加的特殊 token 字符串。
        """
        
        # --- 基础词汇表和合并 ---
        self.vocab: Dict[int, bytes] = vocab
        self.vocab_inv: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        
        # 将合并列表转换为一个 {pair: priority} 字典，以便 O(1) 查找
        # 优先级（priority）就是它在列表中的索引（越小越优先）
        self.merges: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        self.core_pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", 
            regex.UNICODE
        )
        
        # --- 特殊 Token 处理 ---
        self.special_tokens: Dict[str, int] = {}
        self.special_tokens_inv: Dict[int, str] = {}
        self.special_token_pattern = None

        if special_tokens:
            # 按长度降序排序，以确保在正则匹配时优先匹配最长的 token
            # 例如，优先匹配 "<|endoftext|>" 而不是 "<|endof"
            special_tokens.sort(key=len, reverse=True)
            
            # 查找下一个可用的 ID，以便添加新的特殊 token
            # 我们假设词汇表 ID 可能是稀疏的，所以取 max
            current_max_id = max(self.vocab.keys())
            
            for token_str in special_tokens:
                token_bytes = token_str.encode('utf-8')
                # 检查这个特殊 token 是否已经存在于词汇表中
                token_id = self.vocab_inv.get(token_bytes)
                
                if token_id is None:
                    # 如果不存在，将其添加到词汇表
                    current_max_id += 1
                    token_id = current_max_id
                    self.vocab[token_id] = token_bytes
                    self.vocab_inv[token_bytes] = token_id
                    
                # 存储特殊 token 的映射
                self.special_tokens[token_str] = token_id
                self.special_tokens_inv[token_id] = token_str
            
            # 创建一个正则表达式，用于在 encode 时分割特殊 token
            # (token1|token2|token3)
            escaped_tokens = [re.escape(t) for t in special_tokens]
            pattern = f"({'|'.join(escaped_tokens)})"
            self.special_token_pattern = re.compile(pattern)

    @classmethod
    def from_files(cls, 
                   vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens: Optional[List[str]] = None) -> "Tokenizer":
        """
        类方法：从序列化的词汇表和合并文件构造并返回一个 Tokenizer。
        
        **假设文件格式：**
        - vocab_filepath: JSON 文件, 格式: {"id_str": "base64_encoded_bytes", ...}
        - merges_filepath: JSON 文件, 格式: [["base64_b1", "base64_b2"], ...]
        
        参数:
            vocab_filepath: str - 词汇表文件的路径。
            merges_filepath: str - 合并规则文件的路径。
            special_tokens: list[str] | None = None - 可选的特殊 token 列表。
        
        返回:
            Tokenizer - 一个新的 Tokenizer 实例。
        """
        
        # 1. 加载词汇表 (Vocab)
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            # json_vocab 是 {"256": "SGVsbG8="}
            json_vocab_str_keys = json.load(f)
        
        # 将 key 转为 int，将 value 从 base64 解码为 bytes
        vocab = {
            int(k): base64.b64decode(v) 
            for k, v in json_vocab_str_keys.items()
        }
        
        # 2. 加载合并规则 (Merges)
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            # json_merges 是 [["SGU=", "bGxv"], ["d28=", "cmxk"]]
            json_merges_b64 = json.load(f)
        
        # 将每个 pair 中的 base64 字符串解码为 bytes
        merges = [
            (base64.b64decode(b1_b64), base64.b64decode(b2_b64)) 
            for b1_b64, b2_b64 in json_merges_b64
        ]
        
        # 3. 使用加载的数据构造实例
        return cls(vocab, merges, special_tokens)

    def _get_best_pair_and_idx(self, tokens_list: List[bytes]) -> Tuple[Optional[Tuple[bytes, bytes]], int]:
        """
        在 token 列表中查找优先级最高（索引最小）的合并。
        
        返回: (best_pair, best_idx)
             如果找不到合并，则返回 (None, -1)
        """
        best_pair: Optional[Tuple[bytes, bytes]] = None
        best_idx: int = -1
        # 优先级初始化为无穷大（我们寻找最小的优先级数字）
        best_priority: float = float('inf')
        
        # 遍历所有相邻的 token 对
        for i in range(len(tokens_list) - 1):
            pair = (tokens_list[i], tokens_list[i+1])
            # 检查这对 token 是否在我们的合并规则中
            priority = self.merges.get(pair)
            
            if priority is not None and priority < best_priority:
                # 找到了一个更好的（更低优先级的）合并
                best_priority = priority
                best_pair = pair
                best_idx = i
        
        return best_pair, best_idx

    def _encode_chunk(self, text_chunk: str) -> List[int]:
        """
        辅助函数：对一个不含特殊 token 的文本块执行 BPE 编码。
        """
        if not text_chunk:
            return []
            
        # 1. 将文本块转换为 UTF-8 bytes
        try:
            b_text = text_chunk.encode('utf-8')
        except UnicodeEncodeError:
            # 在罕见的代理项对（surrogate pairs）错误时
            return []
            
        # 2. 初始 token 列表是单个字节
        tokens: List[bytes] = [bytes([b]) for b in b_text]
        
        if not tokens:
            return []

        # 3. 循环执行合并
        while True:
            # 查找下一个最佳合并
            best_pair, idx = self._get_best_pair_and_idx(tokens)
            
            if best_pair is None:
                # 没有更多可用的合并
                break
            
            # 执行合并
            merged_token = best_pair[0] + best_pair[1]
            # 更新 token 列表
            tokens = tokens[:idx] + [merged_token] + tokens[idx+2:]
            
        # 4. 将最终的 bytes token 列表转换为 integer ID 列表
        try:
            ids = [self.vocab_inv[token] for token in tokens]
        except KeyError as e:
            # 这是一个严重错误，意味着合并规则产生了词汇表中不存在的 token
            missing_key = e.args[0]
            print(f"严重错误: Token {missing_key} (bytes: {missing_key.hex()}) 由合并规则产生 "
                  f"但它不在词汇表中。您的 vocab 和 merges 文件不匹配。")
            raise e
            
        return ids

    def encode(self, text: str) -> List[int]:
        """
        将输入文本编码为一系列 token ID。
        
        参数:
            text: str - 要编码的字符串。
        
        返回:
            list[int] - token ID 列表。
        """
        final_ids: List[int] = []
        
        # 1. 首先，根据特殊 token 分割文本
        if self.special_token_pattern:
            # 使用 re.split() 并保留分隔符（特殊 token）
            chunks = self.special_token_pattern.split(text)
        else:
            # 没有特殊 token，整个文本是一个块
            chunks = [text]
        
        # 2. 遍历所有块
        for chunk in chunks:
            if not chunk:
                continue
            
            # 检查这个块是否是特殊 token
            token_id = self.special_tokens.get(chunk)
            
            if token_id is not None:
                # 是特殊 token，直接添加其 ID
                final_ids.append(token_id)
            else:
                # 是普通文本块，使用core_pattern再次将其分割
                sub_chunks = self.core_pattern.findall(chunk)
                
                #对每个子块分别执行BPE编码
                for sub_chunk in sub_chunks:
                    final_ids.extend(self._encode_chunk(sub_chunk))
                
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给定一个字符串的可迭代对象（例如文件句柄），
        返回一个懒惰地产生 token ID 的生成器。
        
        参数:
            iterable: Iterable[str] - 字符串的迭代器。
        
        产生 (Yields):
            int - 单个 token ID。
        """
        # 遍历可迭代对象中的每个字符串（例如，文件中的每一行）
        for text_chunk in iterable:
            # 对每个块进行编码，并懒惰地 yield 每个 ID
            yield from self.encode(text_chunk)

    def decode(self, ids: List[int]) -> str:
        """
        将一系列 token ID 解码回文本。
        
        参数:
            ids: list[int] - 要解码的 token ID 列表。
        
        返回:
            str - 解码后的文本。
        """
        
        # 1. 将所有 ID 转换为 bytes
        try:
            token_bytes_list = [self.vocab[id] for id in ids]
        except KeyError as e:
            # 如果提供了词汇表中不存在的 ID
            print(f"解码错误: ID {e} 不在词汇表中。")
            # 您可以选择是抛出异常还是替换
            # 这里我们选择抛出异常，因为这是调用者的数据错误
            raise e
        
        # 2. 将所有 bytes 连接起来
        b_text = b"".join(token_bytes_list)
        
        # 3. 将完整的 bytes 序列解码为 UTF-8 字符串
        # errors='replace' 会将任何无效的 UTF-8 字节序列替换为 ''
        # 这对于处理被截断的多字节字符至关重要
        return b_text.decode('utf-8', errors='replace')