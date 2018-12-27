"""url
URL helpers
"""

import requests

def fixURL(url):
    assert type(url) == str, "URL has to be a string"
    if url.startswith("www."):
        url = "http://{}".format(url)
    return url


def request(url, params=None):
    url = fixURL(url)
    return requests.get(url).content

