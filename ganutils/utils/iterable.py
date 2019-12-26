"""iterable
Helpers for dealing with iterables
"""

###############################################################
# List Manipulation
# =================
#


def flattenCustom(lst, flattenTypes):
    def flattenHelper(seq):
        for x in seq:
            if isinstance(x, flattenTypes):
                yield from flattenCustom(x, flattenTypes)
            else:
                yield x
    return list(flattenHelper(lst))


def flatten(lst):
    return flattenCustom(
                lst, 
                flattenTypes=(list, tuple, set, dict)
            )


def shuffleFirstAxis(arrays):
    """
    Shuffle a (possible) multi-dimensional list by first axis
    """
    import random
    random.shuffle(arrays)
    return arrays


def unique(array):
    return list(set(array))


def uniqueOrdered(array):
    seen = set()
    res = list()
    for x in array:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def strAll(array):
    return [str(x) for x in array]


###############################################################
# Dictionaries
# ==============
#

def getDictKeys(d):
    return list(d.keys())


def getDictValues(d):
    return list(d.values())


def getDictItems(d):
    return list(d.items())


def toDict(keys, values):
    if len(keys) != len(values):
        import warnings
        warnings.warn("Keys and values are not of same length.")
    return dict(zip(keys, values))


def printDictPretty(d):
    import json
    print(json.dumps(d, indent=2))


###############################################################
# Selections
# ==============
#

def randomElement(array):
    import random
    return random.choice(array)


def randomElements(array, size, replacement=True):
    if replacement:
        import random
        return random.choices(array, k=size)
    else:
        import numpy as np
        return np.random.choice(array, size, replace=False)


##############################################################
# Advanced
# ==============
#

def loadArrayFromString(string, dtype=None):
    import numpy as np
    from io import StringIO
    if dtype == None:
        dtype = np.float64
    string = string.replace("[","")
    string = string.replace("]","")
    string = string.replace(",", " ")
    return np.loadtxt(StringIO(string)).astype(dtype)

