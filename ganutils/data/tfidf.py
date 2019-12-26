"""tfidf
Core TFIDF Modules
"""

import math
from collections import Counter


def countTermFrequency(documents):
    counter = Counter()
    for doc in documents:
        for w in doc:
            counter[w] += 1
    return counter


def countDocumentFrequency(documents):
    return countTermFrequency([set(doc) for doc in documents])


def scoreTFIDF(documents, min_term_freq=1, min_doc_freq=1):
    term_counter = countTermFrequency(documents)
    document_counter = countDocumentFrequency(documents)
    # delete words that occur too infrequent
    if min_term_freq > 1:
        words_to_delete = [ 
            w for w in term_counter 
            if term_counter[w] < min_term_freq
        ]
        for w in words_to_delete:
            del term_counter[w]
            del document_counter[w]
    # delete words that are rare among documents
    if min_doc_freq > 1:
        words_to_delete = [ 
            w for w in document_counter 
            if document_counter[w] < min_doc_freq
        ]
        for w in words_to_delete:
            del term_counter[w]
            del document_counter[w]
    # TF-IDF calculations
    num_term = sum(list(term_counter.values()))
    num_doc = len(documents)
    assert num_term > 0, "empty documents!"
    tfs = {
        w: term_counter[w] / num_term
        for w in term_counter
    }
    idfs = {
        w: math.log(num_doc / document_counter[w])
        for w in document_counter
    }
    scores = {
        w: tfs[w] * idfs[w]
        for w in tfs
    }
    return scores


def selectWordsTFIDF(documents, n=None, min_term_freq=1, min_doc_freq=1):
    tfidf_scores = scoreTFIDF(documents, min_term_freq, min_doc_freq)
    tfidf_scores = Counter(tfidf_scores)
    n = len(tfidf_scores) if n is None else n
    # In case of a tie among top N choices, the behavior is nondeterministic 
    words = [w for w, cnt in tfidf_scores.most_common(n)]
    return words


