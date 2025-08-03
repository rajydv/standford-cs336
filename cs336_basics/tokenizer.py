from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from cs336_basics.pretokenization import pretokenization
from collections import defaultdict
from typing import Iterable, Iterator
from cs336_basics.utils import pre_tokenizer_pattern, file_loader
import regex as re

class Tokenizer(ABC):
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        pass




class BPETokenizer(Tokenizer):
    def __init__(self, vocab, merges, special_tokens=None):
        self.id2char_vacab = vocab
        self.char2id_vocab = {char: id for id, char in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [special_token.encode("utf-8") for special_token in self.special_tokens]
        self._update_vocab()

    
    def _update_vocab(self):
        for special_token_byte in self.special_tokens_bytes:
            if special_token_byte not in self.char2id_vocab:
                self.char2id_vocab[special_token_byte] = len(self.char2id_vocab)
                self.id2char_vacab[len(self.id2char_vacab)] = special_token_byte


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> 'BPETokenizer':
        vocab  = file_loader(vocab_filepath)
        merges  = file_loader(merges_filepath)
        return cls(vocab, merges, special_tokens)
    

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            # Sort special tokens by length in descending order to match longer tokens first
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "(" + "|".join(re.escape(st) for st in sorted_special_tokens) + ")"
            chunks = re.split(special_pattern, text)
        else:
            chunks = [text]

        # Process each chunk
        result = []
        for chunk in chunks:
            # If chunk is a special token
            if chunk in self.special_tokens:
                special_token_bytes = chunk.encode("utf-8")
                result.append(self.char2id_vocab[special_token_bytes])
            # If chunk is regular text, ignore empty chunck eg: ('')
            elif chunk:
                chunk_bytes = chunk.encode("utf-8")
                pre_tokens = re.findall(pre_tokenizer_pattern.encode("utf-8"), chunk_bytes, re.IGNORECASE)
                split_pre_tokens = [tuple([bytes([x]) for x in pre_token]) for pre_token in pre_tokens]
                merge_pre_tokens = self.merge_split_pre_tokens(split_pre_tokens)
                for merge_pre_token in merge_pre_tokens:
                    for token in merge_pre_token:
                        result.append(self.char2id_vocab[token])
        
        return result
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens = self.encode(text)
            for token in tokens:
                yield token


    def decode(self, tokens: list[str]) -> str:
        char_token  = [self.id2char_vacab[token] for token in tokens]
        return b"".join(char_token).decode("utf-8", errors="replace")


    def merge_split_pre_tokens(
        self,
        split_pre_tokens: list[tuple[bytes]]
    ) -> list[tuple[bytes]]:
        merge_tokens = split_pre_tokens
        for merge in self.merges:
            merge_tokens = [merge_pair_util(merge[0], merge[1], token) for token in merge_tokens]
        return merge_tokens



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


def merge_pair_util(
    first: bytes,
    second: bytes,
    word: tuple[bytes]
) -> tuple[bytes]:
    chars = list(word)
    # print(f"Looking merge pair {first}, {second} in word: {chars}")
    if len(chars) < 2:
        return tuple(chars)
    
    i = 0
    while i < len(chars) - 1:
        if chars[i] == first and chars[i + 1] == second:
            chars[i:i + 2] = [first + second]
        else:
            i += 1
    # print(f"Update word: {chars}")
    return tuple(chars)


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
        new_word = merge_pair_util(first, second, word)
        new_word_freqs[new_word] = freq
        
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
