from bert_text_summarizer.metrics.rouge import rouge_n, n_grams


def test_ngrams():
    x = [1, 2, 3]

    ngrams = n_grams(x, 1)
    assert ngrams == [(1,), (2,), (3,)]

    ngrams = n_grams(x, 2)
    assert ngrams == [(1, 2), (2, 3)]


def test_rouge_n():
    x = [1, 2, 3]
    y = [1, 2, 3]

    score = rouge_n(x, y, 2)
    assert score == 1

    x = [1, 2, 3]
    y = [1, 2]

    assert rouge_n(x, y, 2) == 2/3
    assert rouge_n(x, y, 1) == 0.8
