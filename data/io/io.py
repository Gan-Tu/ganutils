"""io.io
Core IO Modules
"""

import os
import json
import pickle

###############################################################
# Common I/O operations
# ======================
#

def makedirs(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def walk(source_dir):
    paths = list()
    for dirpath, _, filenames in os.walk(source_dir):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
    return paths

def loadJSON(filepath, encoding="utf-8"):
    return json.load(open(filepath, "r", encoding=encoding))

def dumpJSON(obj, filepath, indent=None, ensure_ascii=False):
    makedirs(filepath)    
    json.dump(
        obj, 
        open(filepath, "w"), 
        indent=indent, 
        ensure_ascii=ensure_ascii
    )

def loadPickle(filepath):
    return pickle.load(open(filepath, "rb"))

def dumpPickle(obj, filepath):
    makedirs(filepath)
    pickle.dump(obj, open(filepath, "wb"))
