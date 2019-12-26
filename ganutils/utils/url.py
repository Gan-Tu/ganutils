"""url
URL helpers
"""

import requests

def fixURL(url):
    assert type(url) == str, "URL has to be a string"
    # TODO(tugan): fixes more cases
    if url.startswith("www."):
        url = "http://{}".format(url)
    return url


def request(url, params=None):
    # TODO(tugan): use params as necessary
    url = fixURL(url)
    return requests.get(url).content

