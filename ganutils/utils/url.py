"""url
URL helpers
"""

import requests


def fix_url(url):
    """
    .. todo:: fixes more cases
    """
    assert type(url) == str, "URL has to be a string"
    if url.startswith("www."):
        url = "http://{}".format(url)
    return url


def request(url, params=None):
    """
    .. todo:: use params as necessary
    """
    url = fix_url(url)
    return requests.get(url).content
