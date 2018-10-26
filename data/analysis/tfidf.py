"""analysis.tfidf
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
    counter = Counter()
    for doc in documents:
        for w in set(doc):
            counter[w] += 1
    return counter

def tfidf(documents):
    term_counter = countTermFrequency(documents)
    document_counter = countDocumentFrequency(documents)
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
