"""iterable
Helpers for dealing with iterables
"""

###############################################################
# List Manipulation
# =================
#


def flatten_custom(lst, flatten_types):
    def flatten_helper(seq):
        for x in seq:
            if isinstance(x, flatten_types):
                yield from flatten_custom(x, flatten_types)
            else:
                yield x
    return list(flatten_helper(lst))


def flatten(lst):
    return flatten_custom(
        lst,
        flatten_types=(list, tuple, set, dict)
    )


def shuffle_first_axis(arrays):
    """
    Shuffle a (possible) multi-dimensional list by first axis
    """
    import random
    random.shuffle(arrays)
    return arrays


def unique(array):
    return list(set(array))


def unique_ordered(array):
    seen = set()
    res = list()
    for x in array:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def stringify_arr(array):
    return [str(x) for x in array]


###############################################################
# Dictionaries
# ==============
#

def get_dict_keys(d):
    return list(d.keys())


def get_dict_values(d):
    return list(d.values())


def get_dict_items(d):
    return list(d.items())


def to_dict(keys, values):
    if len(keys) != len(values):
        import warnings
        warnings.warn("Keys and values are not of same length.")
    return dict(zip(keys, values))


def print_dict_pretty(d):
    import json
    print(json.dumps(d, indent=2))


###############################################################
# Selections
# ==============
#

def randomly_pick_one(array):
    import random
    return random.choice(array)


def randomly_pick_multiple(array, size, replacement=True):
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

def load_arr_from_string(string, dtype=None):
    import numpy as np
    from io import StringIO
    if dtype == None:
        dtype = np.float64
    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace(",", " ")
    return np.loadtxt(StringIO(string)).astype(dtype)
