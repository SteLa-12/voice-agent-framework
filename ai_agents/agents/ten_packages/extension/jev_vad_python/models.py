from pydantic import BaseModel
from typing import Literal, Union
from enum import Enum

class EndOfTurnPayload(BaseModel):
    text: str
    reason: Literal["jev", "fallback"]
    confidence: float

class AgentInterruptPayload(BaseModel):
    text: str
    reason: Literal["jev", "fallback"]
    confidence: float

class InterruptPayload(BaseModel):
    text: str
    confidence: float

class FloorState(Enum):
    IDLE = "idle"
    USER = "user"
    AGENT = "agent"
    OVERLAP = "overlap"
    AGENT_PENDING = "agent_pending"

class EmitEndOfTurn(BaseModel):
    payload: EndOfTurnPayload


class EmitUserInterrupt(BaseModel):
    payload: InterruptPayload


class IgnoreUserBackchannel(BaseModel):
    pass


class PlayBackchannel(BaseModel):
    pass


class EmitAgentInterrupt(BaseModel):
    payload: AgentInterruptPayload


class NoOp(BaseModel):
    pass


Action = Union[
    EmitEndOfTurn,
    EmitUserInterrupt,
    IgnoreUserBackchannel,
    PlayBackchannel,
    EmitAgentInterrupt,
    NoOp,
]