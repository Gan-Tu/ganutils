"""client
A Client for interfacing with GitHub
"""

import os
import requests
import json
from .endpoints import get_api_endpoint

class GithubClient(object):

    def __init__(self, access_token):
        """
        Initiate a custom GitHub Client
        """
        self.session = requests.Session()
        self.session.headers["Authorization"] = "token %s" % access_token


    ##########################
    # RESTful API Helper
    ##########################

    def serialize(self, obj):
        """
        Serialize the dictionary obj as a JSON string
        """
        return json.dumps(obj)

    def post(self, url, data={}):
        """
        Make a POST call to URL with DATA and return the response.
        """
        return self.session.post(url, self.serialize(data))

    def patch(self, url, data={}):
        """
        Make a PATCH call to URL with DATA and return the response.
        """
        return self.session.patch(url, self.serialize(data))

    ##########################
    # GitHub Issue API
    ##########################

    def create_issue(self, repo_owner, repo_name, title, body=None):
        """
        Create an issue on GitHub for REPO_OWNER/REPO_NAME with
        the given TITLE and BODY. Return the response.
        """
        url = get_api_endpoint(endpoint="issue", method="create")
        url = url.format(repo_owner=repo_owner, repo_name=repo_name)
        return self.post(url, {'title': title, 'body': body})

    def close_issue(self, repo_owner, repo_name, issue_number):
        """
        Close an issue on GitHub for REPO_OWNER/REPO_NAME with
        the given ISSUE_NUMBER. Return the response.
        """
        url = get_api_endpoint(endpoint="issue", method="close")
        url = url.format(repo_owner=repo_owner,
                         repo_name=repo_name,
                         issue_number=issue_number)
        return self.patch(url, {"state": "closed"})

