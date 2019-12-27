"""
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

    ##########################
    # GitHub Repo API
    ##########################

    def list_my_repos(self, visibility="all", affiliation="owner"):
        """
        List repositories of current authenticated client user.

        Args:
            visibility : string
                Can be one of `all`, `public`, or `private`.\n
                Default: `all`

            affiliation : string
                Comma-separated list of values. Can include:

                * `owner`: repos owned by the authenticated user.
                * `collaborator`: repos that the user has been added to as a collaborator.
                * `organization_member`: repos that the user has access to through being a member of an organization. This includes every repository on every team that the user is on.

                Default: `owner`

        Returns:
            A response of all repos metadata
        """
        url = get_api_endpoint(endpoint="repo", method="list-mine")
        return self.session.get(url, params={'visibility': visibility,
                                             'affiliation': affiliation})


    ############################
    # GitHub Pull Requests API
    ############################

    def list_pulls(self, repo_owner, repo_name, state="open"):
        """
        List pull requests of given repo.
        """
        url = get_api_endpoint(endpoint="pull", method="list")
        url = url.format(repo_owner=repo_owner, repo_name=repo_name)
        return self.session.get(url, params={'state': state})

    def merge_pull(self, repo_owner, repo_name, pull_number):
        """
        Merge a given pull request
        """
        url = get_api_endpoint(endpoint="pull", method="merge")
        url = url.format(repo_owner=repo_owner,
                         repo_name=repo_name,
                         pull_number=pull_number)
        return self.session.put(url)

    ##########################
    # GitHub Issue API
    ##########################

    def create_issue(self, repo_owner, repo_name, title, body=""):
        """
        Create an issue on GitHub for REPO_OWNER/REPO_NAME with
        the given TITLE and BODY. Return the response.
        """
        url = get_api_endpoint(endpoint="issue", method="create")
        url = url.format(repo_owner=repo_owner, repo_name=repo_name)
        return self.session.post(url, data={'title': title, 'body': body})

    def close_issue(self, repo_owner, repo_name, issue_number):
        """
        Close an issue on GitHub for REPO_OWNER/REPO_NAME with
        the given ISSUE_NUMBER. Return the response.
        """
        url = get_api_endpoint(endpoint="issue", method="close")
        url = url.format(repo_owner=repo_owner,
                         repo_name=repo_name,
                         issue_number=issue_number)
        return self.session.patch(url, data={"state": "closed"})

