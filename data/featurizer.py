"""featurizer
Core Featurization Pipeline
"""

from __future__ import unicode_literals, print_function, division

# define predefined tokens

PREDEFINED_LABELS = [
    "pad", 
    "default"
]

PREDEFINED_ALPHABET = [
    "[cap_unk]",
    "[unk]",
    "[pad]",
    "[space]"
]

PREDEFINED_VOCAB = [
    "[cap_unk]",
    "[title_unk]",
    "[unk]",
    "[pad]",
    "[space]"
]


######################################################################
# Data Encoding
# ===================
#

def getVocabEncoding(words):
    words.extend(PREDEFINED_VOCAB)
    words = list(sorted(set(words)))
    return { w: i for i,w in enumerate(words)}


def encodeWord(word, vocab2idx, case_insensitive=True):
    # sanity check
    for w in PREDEFINED_VOCAB:
        assert w in vocab2idx, \
            "predefined token '{}' not in vocab2idx".format(word)
    # return encoding, if the word is found
    if word in vocab2idx:
        return vocab2idx[word]
    # if case insensitive, encode alternative forms of the word
    elif case_insensitive and word.title() in vocab2idx:
        return vocab2idx[word.title()]
    elif case_insensitive and word.lower() in vocab2idx:
        return vocab2idx[word.lower()]
    elif case_insensitive and word.upper() in vocab2idx:
        return vocab2idx[word.upper()]
    # encode as an unknown word
    if word.isspace():
        return vocab2idx["[space]"]
    elif word.isupper():
        return vocab2idx["[cap_unk]"]
    elif word.istitle():
        return vocab2idx["[title_unk]"]
    else:
        return vocab2idx["[unk]"]


def getCharEncoding(chars):
    if type(chars) == str:
        chars = list(chars)
    chars.extend(PREDEFINED_ALPHABET)
    chars = list(sorted(set(chars)))
    return { c: i for i,c in enumerate(chars)}


def encodeChar(char, char2idx, case_insensitive=True):
    # sanity check
    for c in PREDEFINED_ALPHABET:
        assert c in char2idx, \
            "predefined token '{}' not in char2idx".format(char)
    # return encoding, if the char is found
    if char in char2idx:
        return char2idx[char]
    # if case insensitive, encode alternative forms of the char
    elif case_insensitive and char.lower() in char2idx:
        return char2idx[char.lower()]
    elif case_insensitive and char.upper() in char2idx:
        return char2idx[char.upper()]
    # encode as an unknown char
    if char.isspace():
        return char2idx["[space]"]
    elif char.isupper():
        return char2idx["[cap_unk]"]
    else:
        return char2idx["[unk]"]


def getLabelEncoding(labels):
    labels.extend(PREDEFINED_LABELS)
    labels = list(sorted(set(labels)))
    return { l: i for i,l in enumerate(labels)}


def encodeLabel(label, label2idx):
    assert "default" in label2idx, \
        "predefined token 'default' not in label2idx"
    return label2idx.get(label, label2idx["default"])


def onehotEncode(labels, label_size):
    """
    Convert a list of integer LABELS to a binary class matrix.
    """
    import keras.utils as KU
    return KU.to_categorical(labels, num_classes=label_size).astype(int)


######################################################################
# Data Featurization
# ===================
#

def featurizeWords(words, vocab2idx, case_insensitive=True, doc_maxlen=None):
    words = [ 
        encodeWord(word, vocab2idx, case_insensitive) 
        for word in words
    ]
    # pad/truncate documents, if necessary
    if doc_maxlen is not None:
        words = padSequences(
            [words], 
            doc_maxlen, 
            dtype=int, 
            value=vocab2idx["[pad]"]
        )[0]
    return words


def featurizeChars(words, char2idx, case_insensitive=True, 
                   word_maxlen=None, doc_maxlen=None):
    chars = [[
        encodeChar(c, char2idx, case_insensitive) 
        for c in word
    ] for word in words ]
    # pad/truncate documents, if necessary
    if word_maxlen is not None:
        chars = padSequences(
            chars,
            word_maxlen, 
            dtype=int, 
            value=char2idx["[pad]"]
        )
    if doc_maxlen is not None:
        assert word_maxlen,  "word_maxlen required for doc_maxlen"
        chars = padSequences(
            [chars], 
            doc_maxlen, 
            dtype=int, 
            value=char2idx["[pad]"]
        )[0]
    return chars


def smoothLabel(onehot_labels, label_size, epsilon):
    zero_value = epsilon / (label_size - 1)
    one_value = 1 - epsilon
    # [0, ..., 1, ..., 0]
    smoothed_labels  = onehot_labels * one_value
    # [0, ..., 1 - epsilon, ..., 0]
    smoothed_labels += (1 - onehot_labels) * zero_value
    # [epsilon/(k-1), ..., 1 - epsilon, ... epsilon/(k-1)]
    return smoothed_labels


def featurizeLabels(labels, label2idx, label_size, doc_maxlen=None, 
                    onehot=False, smoothing_epsilon=None):
    labels = [ 
        encodeLabel(label, label2idx) 
        for label in labels
    ]
    # pad/truncate documents, if necessary
    if doc_maxlen is not None:
        labels = padSequences(
            [labels], 
            doc_maxlen, 
            dtype=int, 
            value=label2idx["pad"]
        )[0]
    if onehot:
        labels = onehotEncode(labels, label_size)
        if smoothing_epsilon is not None:
            assert smoothing_epsilon < 1, "smoothing epsilon must be < 1"
            labels = smoothLabel(labels, label_size, smoothing_epsilon)
    return labels


######################################################################
# Data Padding & Truncation
# =========================================
#

def padSequences(sequences, maxlen, dtype='float32', 
                  padding='post', truncating='post', value=0.0):
    import keras.preprocessing as KP        
    return KP.sequence.pad_sequences(
        sequences,
        maxlen, 
        dtype, 
        padding, 
        truncating, 
        value
    )


