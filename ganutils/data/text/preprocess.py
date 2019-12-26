"""text.preprocess
Core Preprocessing Modules for Texts
"""

import re
import string
import unicodedata


def normalizeUnicode(string, encoding="utf-8"):
    if isinstance(string, type(b'')):
        string = string.decode(encoding)
    # replace "oe" and "ae" letters, or else they are dropped!
    string = string.replace(u"æ", u"ae").replace(u"Æ", u"AE")
    string = string.replace(u"œ", u"oe").replace(u"Œ", u"OE")
    string = unicodedata.normalize('NFKD', string)
    string = string.encode('ascii', 'ignore')
    string = string.decode()
    return string


def normalizeString(string, encoding="utf-8"):
    normalizedChar = [
        normalizeUnicode(c) for c in string
    ]
    normalizedChar = [
        normalizedChar[i]
        if len(normalizedChar[i]) > 0 else c
        for i, c in enumerate(string)
    ]
    return "".join(normalizedChar)


def padPunctuations(s, punct=".!?.。-!！?？'’,，:…()（）)'\""):
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
    # special space-like characters
    s = s.replace("\xa0", " ")
    # shrink consecutive spaces into one
    s = re.sub(r"\s+", " ", s)
    # shrink consecutive new line characters into one
    s = re.sub(r"\n+", "[newline]", s)
    s = s.replace("[newline]", "\n")
    return s.strip()


def convertEmailToToken(s, token="[email]"):
    return re.sub(r"[a-zA-Z\+\-_\d\.]+@[a-zA-Z\+\d\.]+", token, s)


def convertLinkToToken(s, token="[link]"):
    s = re.sub(r"https?://[a-z/A-Z\+\?=\-_\d\.]+", token, s)
    s = re.sub(r"www\.[a-z/A-Z\+\?=\-_\d\.]+", token, s)
    s = re.sub(r"[a-z/A-Z\+\?=\-_\d\.]+\.com", token, s)
    return s




