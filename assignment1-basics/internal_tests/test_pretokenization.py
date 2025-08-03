import os
import regex as re
from cs336_basics.pretokenization import preprocess_chunk

def test_preprocess_chunk():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "tiny_file.txt")
    start = 0
    end = os.path.getsize(file_path)
    
    token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_pattern_byte = token_pattern.encode("utf-8")

    special_tokens = ["<|endoftext|>", "<|START|>", "<|END|>"]
    escaped_tokens = [re.escape(token) for token in special_tokens]
    split_special_token = "|".join(escaped_tokens)
    split_special_token_byte = split_special_token.encode("utf-8")

    result = preprocess_chunk(
        file_path,
        start,
        end,
        token_pattern_byte,
        split_special_token_byte,
    )
    expected_result = {
        (b'\n',): 2,
        (b' ',): 1,
        (b' ', b'l', b'o', b'w'): 4,
        (b' ', b'l', b'o', b'w', b'e', b'r'): 1,
        (b' ', b'n', b'e', b'w', b'e', b's', b't'): 5,
        # (b' ', b'w', b'i', b'd', b'e', b's', b't'): 3,
        # (b'l', b'o', b'w'): 1,
        # (b'l', b'o', b'w', b'e', b'r'): 1,
        # (b'n', b'e', b'w', b'e', b's', b't'): 1
    }
    assert result == expected_result

