from pydantic import BaseModel

class EndOfTurnResult(BaseModel):
    text: str