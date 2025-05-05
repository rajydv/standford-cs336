from cs336_basics.tokenizer import compute_pair_freqs, merge_pair, perform_merge_iteration, BPETokenizer

def test_compute_pair_freqs():
    word_freqs = {
        (b'\n',): 2,
        (b' ',): 1,
        (b' ', b'l', b'o', b'w'): 4,
        (b' ', b'l', b'o', b'w', b'e', b'r'): 1,
        (b' ', b'n', b'e', b'w', b'e', b's', b't'): 5,
    }

    result = compute_pair_freqs(word_freqs)
    expected_result = {
        (b' ', b'l'): 5,
        (b'l', b'o'): 5,
        (b'o', b'w'): 5,
        (b'w', b'e'): 6,
        (b'e', b'r'): 1,
        (b' ', b'n'): 5,
        (b'n', b'e'): 5,
        (b'e', b'w'): 5,
        (b'e', b's'): 5,
        (b's', b't'): 5
    }

    assert result == expected_result

def test_merge_pair():
    word_freqs = {
        (b'\n',): 2,
        (b' ',): 1,
        (b' ', b'l', b'o', b'w'): 4,
        (b' ', b'l', b'o', b'w', b'e', b'r'): 1,
        (b' ', b'n', b'e', b'w', b'e', b's', b't'): 5,
    }

    first  = b'l'
    second = b'o'

    result = merge_pair(first, second, word_freqs)

    excepted_result = {
        (b'\n',): 2,
        (b' ',): 1,
        (b' ', b'lo', b'w'): 4,
        (b' ', b'lo', b'w', b'e', b'r'): 1,
        (b' ', b'n', b'e', b'w', b'e', b's', b't'): 5,
    }
    assert result == excepted_result


    first = b'lo'
    second = b'w'
    result = merge_pair(first, second, word_freqs)
    excepted_result = {
        (b'\n',): 2,
        (b' ',): 1,
        (b' ', b'low'): 4,
        (b' ', b'low', b'e', b'r'): 1,
        (b' ', b'n', b'e', b'w', b'e', b's', b't'): 5,
    }


def test_perform_merge_iteration():

    # Case 1
    word_freqs = {
        (b'\n',): 2,
        (b' ',): 1,
        (b' ', b'l', b'o', b'w'): 4,
        (b' ', b'l', b'o', b'w', b'e', b'r'): 1,
        (b' ', b'n', b'e', b'w', b'e', b's', b't'): 5,
    }
    vocab = {}
    merges = []
    num_merges = 3

    vocab, merges = perform_merge_iteration(word_freqs, vocab, merges, num_merges)

    expectd_vocab = {
        0: b'we',
        1: b'wes',
        2: b'west'
    }

    expectd_merges = [(b'w', b'e'), (b'we', b's'), (b'wes', b't')]

    assert vocab == expectd_vocab
    assert merges == expectd_merges


    # Case 2
    word_freqs = {
        (b'\n',): 2,
        (b' ',): 1,
        (b' ', b'l', b'o'): 4,
        (b' ', b'l', b'w'): 3,
    }


    vocab = {}
    merges = []
    num_merges = 3

    vocab, merges = perform_merge_iteration(word_freqs, vocab, merges, num_merges)

    expectd_vocab = {
        0: b' l',
        1: b' lo',
        2: b' lw'
    }

    expectd_merges = [(b' ', b'l'), (b' l', b'o'), (b' l', b'w')]

    assert vocab == expectd_vocab
    assert merges == expectd_merges



    # Case 3
    word_freqs = {
        (b'\n',): 2,
        (b' ',): 1,
        (b' ', b'l', b'o'): 4,
        (b' ', b'l', b'w'): 3,
    }


    vocab = {0: b'we'}
    merges = []
    num_merges = 3

    vocab, merges = perform_merge_iteration(word_freqs, vocab, merges, num_merges)

    expectd_vocab = {
        0: b'we',
        1: b' l',
        2: b' lo',
        3: b' lw'
    }

    expectd_merges = [(b' ', b'l'), (b' l', b'o'), (b' l', b'w')]

    assert vocab == expectd_vocab
    assert merges == expectd_merges



def test_bpe_encode():
    vocab = {0: b' ',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at'
            }

    merges = [(b't', b'h'), (b' ', b'c'), (b' ',b'a'), (b'th', b'e'), (b' a', b't')]

    bpe_tokenzier = BPETokenizer(vocab, merges)
    text = 'the cat ate'
    result = bpe_tokenzier.encode(text)
    # [(b't', b'h', b'e'),  => b'th', b'e'
    #  (b' ', b'c', b'a', b't'), => b' c', b'a', b't'
    #  (b' ', b'a', b't', b'e')]
    expected = [9, 7, 1, 5, 10, 3]
    assert result == expected

def test_bpe_decode():
    vocab = {0: b' ',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at'
            }

    merges = [(b't', b'h'), (b' ', b'c'), (b' ',b'a'), (b'th', b'e'), (b' a', b't')]

    bpe_tokenzier = BPETokenizer(vocab, merges)
    tokens = [9, 7, 1, 5, 10, 3]
    result = bpe_tokenzier.decode(tokens)
    expected = 'the cat ate'
    assert result == expected

def test_bpe_encode_iterable():
    vocab = {0: b' ',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at'
            }

    merges = [(b't', b'h'), (b' ', b'c'), (b' ',b'a'), (b'th', b'e'), (b' a', b't')]

    bpe_tokenizer = BPETokenizer(vocab, merges)
    
    # Test with a list of strings
    texts = ['the cat', 'ate']
    token_iterator = bpe_tokenizer.encode_iterable(texts)
    
    # Test tokens are yielded one by one
    assert next(token_iterator) == 9  # 'the'
    assert next(token_iterator) == 7  # ' c'
    assert next(token_iterator) == 1  # 'a'
    assert next(token_iterator) == 5  # 't'
    assert next(token_iterator) == 1  # 'a'
    assert next(token_iterator) == 5  # 't'
    assert next(token_iterator) == 3  # 'e'
    
    # Verify iterator is exhausted
    try:
        next(token_iterator)
        assert False, "Iterator should be exhausted"
    except StopIteration:
        pass
