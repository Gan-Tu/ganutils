"""client
A Client for interfacing with Twilio
"""

import os
from twilio.rest import Client

class TwilioClient(object):

    def __init__(self, account_sid, auth_token, account_phone):
        """
        Initiate a custom Twilio Client
        """
        self.client = Client(account_sid, auth_token)
        self.account_phone = account_phone

    ##########################
    # Twilio API
    ##########################

    def text(self, to_number, text_body):
        return self.client.messages.create(
                from_=self.account_phone,
                to=to_number,
                body=text_body)


