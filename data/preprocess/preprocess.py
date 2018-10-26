"""preprocess.preprocess
Core Preprocessing Modules
"""

import re
import string
import unicodedata


###############################################################
# String Normalizations
# ======================
#

def normalize(string, encoding="utf-8"):
    if isinstance(string, type(b'')):
        string = string.decode(encoding)
    # replace "oe" and "ae" letters, or else they are dropped!
    string = string.replace(u"æ", u"ae").replace(u"Æ", u"AE")
    string = string.replace(u"œ", u"oe").replace(u"Œ", u"OE")
    string = unicodedata.normalize('NFKD', string)
    string = string.encode('ascii', 'ignore')
    string = string.decode()
    return string

def padPunctuations(s, punct=".!?.。-!！?？'’,，:…()（）)'\"");
    return re.sub(r"([{}])".format(re.escape(punct)), r" \1 ", s)

def ngram(word, n):
    assert len(word) >= n, "ngram size cannot be larger than the word"
    res = list()
    for i in range(len(word)-n+1):
        res.append(word[i:i+n])
    return res

def stripEmoji(s):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", s)

def shrinkSpaces(s):
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\n+", "[newline]", s)
    s = s.replace("[newline]", "\n")
    return s



