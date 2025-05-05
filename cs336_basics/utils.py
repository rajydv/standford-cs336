pre_tokenizer_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def file_loader(file_path: str):
    fp = open(file_path, "rb") 
    content  = fp.readlines()
    fp.close()
    return content