from cs336_basics.tokenizer import compute_pair_freqs, merge_pair, perform_merge_iteration

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
