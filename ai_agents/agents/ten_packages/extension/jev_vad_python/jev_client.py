from typesafe_sdk import AsyncTypeClient, Choice, Noul, Score

class JevClient:
    def __init__(self, api_key: str):
        self.client = AsyncTypeClient(api_key)