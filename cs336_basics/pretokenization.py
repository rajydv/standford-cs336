import os
import regex as re
from typing import BinaryIO
from collections import defaultdict, Counter
from multiprocessing import Pool
from cs336_basics.utils import pre_tokenizer_pattern

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    mini_chunk_size: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size


    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be  at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def preprocess_chunk(
    file_path: str,
    start: int,
    end: int,
    pre_tokenizer_pattern_byte: bytes,
    escaped_special_tokens_byte: bytes
) -> dict[tuple[bytes], int]:
    # init
    word_freqs: dict[tuple[bytes], int] = defaultdict(int)

    # read data
    with open(file_path, "rb") as file:
        file.seek(start)
        docs = file.read(end - start)

        # break into sentences/doc and process each doc
        docs_iter = re.split(escaped_special_tokens_byte, docs)
        for doc in docs_iter:
            # cleaned_doc = re.sub(rb'\s+', b' ', doc)
            matchs_iter = re.finditer(pre_tokenizer_pattern_byte, doc, re.IGNORECASE)
            for match in matchs_iter:
                word_freqs[tuple(bytes([b]) for b in match.group())] += 1
                # word_freqs[match.group()] += 1

    return word_freqs

def pretokenization(
    file_path: str,
    special_tokens: list[str],
    num_process: int = 16,
) -> dict[tuple[bytes], int]:
    pre_tokenizer_pattern_bytes = pre_tokenizer_pattern.encode("utf-8")


    escaped_tokens = [re.escape(token) for token in special_tokens]
    split_special_token = "|".join(escaped_tokens)
    split_special_token_byte = split_special_token.encode("utf-8")
    mini_chunk_size = 1024*4 # 4K bytes
    
    with open(file_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_process, mini_chunk_size, split_special_token_byte)

    chunk_params = [(file_path, start, end, pre_tokenizer_pattern_bytes, split_special_token_byte) 
                    for start, end in zip(boundaries[:-1], boundaries[1:])]
        
    with Pool(processes=num_process) as pool:
        results = pool.starmap(preprocess_chunk, chunk_params)
        
    final_word_freqs = Counter()
    for result in results:
        final_word_freqs.update(result)
            
    return dict(final_word_freqs)

if __name__ == '__main__':
    file_path = "data/tiny.txt"
    special_tokens = ["<|endoftext|>", "<|START|>", "<|END|>"]
    result  = pretokenization(file_path, special_tokens, num_process=16)
    print(result)
    
