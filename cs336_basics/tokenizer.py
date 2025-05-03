from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from cs336_basics.pretokenization import pretokenization
from collections import defaultdict
from typing import BinaryIO

class Tokenizer(ABC):
    
    @abstractmethod
    def encode(self, string: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        pass

@dataclass(frozen=True)
class BPETokenizerParams():
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]

class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        pass

    def decode(self, tokens: list[str]) -> str:
        pass



def compute_pair_freqs(
    word_freqs: dict[tuple[bytes], int]
) -> dict[tuple[bytes, bytes], int]:
    pair_freqs = defaultdict(int)
    for word_split, freq in word_freqs.items():
        if len(word_split) < 2:  # Skip single-character words
            continue
        # Count pairs within this word
        for i in range(len(word_split) - 1):
            pair = (word_split[i], word_split[i + 1])
            pair_freqs[pair] += freq
    return dict(pair_freqs)  # Convert to regular dict to ensure consistent iteration

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, 
    new_indices = []  # @inspect new_indices
    i = 0 # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def merge_pair(
    first: bytes,
    second: bytes,
    word_freqs: dict[tuple[bytes], int]
) -> dict[tuple[bytes], int]:
    """
    Merge all occurrences of (first, second) pair in the vocabulary.
    """
    new_word_freqs = {}
    
    for word, freq in word_freqs.items():
        if len(word) < 2:
            new_word_freqs[word] = freq
            continue
            
        # Convert word into list for easier manipulation
        chars = list(word)
        i = 0
        
        while i < len(chars) - 1:
            if chars[i] == first and chars[i + 1] == second:
                chars[i:i + 2] = [first + second]
            else:
                i += 1
                
        new_word_freqs[tuple(chars)] = freq
        
    return new_word_freqs


def perform_merge_iteration(
    word_freqs: dict[tuple[bytes], int],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    num_merges: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Perform BPE merge iterations following GPT-2's approach.
    """
    initial_vocab_size = len(vocab)
    
    for i in range(num_merges):
        pair_freqs = compute_pair_freqs(word_freqs)
        # For ties in frequency, prefer the lexicographically greater pair
        # e.g., if ("A", "B") and ("BA", "A") have same frequency, pick ("BA", "A")
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))
            
        first, second = best_pair[0]
        pair_bytes = first + second
        
        # Add the merge rule
        merges.append((first, second))
        # Add to vocabulary with correct index
        new_index = initial_vocab_size + i
        vocab[new_index] = pair_bytes
        
        # Update word frequencies with the merged pair
        word_freqs = merge_pair(first, second, word_freqs)
        
    return vocab, merges

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    word_freqs: dict[tuple[bytes], int] = pretokenization(input_path, special_tokens)
    merges: list[tuple[bytes, bytes]] = []  # (<token1>, <token2>)
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> byte
    
    # add special tokens in vocab
    for idx, st in enumerate(special_tokens):
        vocab[256+idx] = st.encode("utf-8")
        
    initial_vocab_size = len(vocab)
    num_merges = vocab_size - initial_vocab_size
    print(f"Number of merged required: {num_merges}")
    
    updated_vocab, updated_merges = perform_merge_iteration(word_freqs, vocab, merges, num_merges)
    return (updated_vocab, updated_merges)
