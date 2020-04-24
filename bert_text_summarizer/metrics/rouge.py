from typing import List


def n_grams(x: List[any], n: int) -> List[tuple]:
    result = []
    for i in range(len(x) - n + 1):
        result.append(tuple(x[i:i + n]))
    return result


def rouge_n(reference: List[any], candidate: List[any], n: int) -> float:
    ref_ngrams = n_grams(reference, n)
    candidate_ngrams = n_grams(candidate, n)

    if len(ref_ngrams) == 0 and len(candidate_ngrams) == 0:
        return 1.0
    elif len(ref_ngrams) == 0 or len(candidate_ngrams) == 0:
        return 0.0

    unique_c = set(candidate_ngrams)
    unique_r = set(ref_ngrams)

    matches = len(unique_c.intersection(unique_r))
    recall = matches / len(unique_r)

    return recall

    # precision = matches / len(unique_c)

    # if precision + recall == 0:
    #     return 0
    #
    # return 2 * precision * recall / (precision + recall)
    #
