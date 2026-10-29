from pydantic import BaseModel
from typing import Literal, Union, Dict, Any
from ten_ai_base.types import LLMToolMetadata


# ==== Base Event ====


class AgentEventBase(BaseModel):
    """Base class for all agent-level events."""

    type: Literal["cmd", "data"]
    name: str


# ==== CMD Events ====


class UserJoinedEvent(AgentEventBase):
    """Event triggered when a user joins the session."""

    type: Literal["cmd", "data"] = "cmd"
    name: str = "on_user_joined"


class UserLeftEvent(AgentEventBase):
    """Event triggered when a user leaves the session."""

    type: Literal["cmd", "data"] = "cmd"
    name: str = "on_user_left"


class ToolRegisterEvent(AgentEventBase):
    """Event triggered when a tool is registered by the user."""

    type: Literal["cmd", "data"] = "cmd"
    name: str = "tool_register"
    tool: LLMToolMetadata
    source: str


# ==== DATA Events ====


class ASRResultEvent(AgentEventBase):
    """Event triggered when ASR result is received (partial or final)."""

    type: Literal["cmd", "data"] = "data"
    name: str = "asr_result"
    text: str
    final: bool
    metadata: Dict[str, Any]
    start_ms: int | None = None


class ASRCommitTimeoutEvent(AgentEventBase):
    """The speech-final fallback deadline for one ASR session."""

    type: Literal["cmd", "data"] = "data"
    name: str = "asr_commit_timeout"
    session_id: str
    stream_id: int
    generation: int


class LLMResponseEvent(AgentEventBase):
    """Event triggered when LLM returns a streaming response."""

    type: Literal["message", "reasoning"] = "message" # type: ignore
    name: str = "llm_response"
    delta: str
    text: str
    is_final: bool


class ModelRouteEvent(AgentEventBase):
    """The selected LLM destination for a user turn."""

    type: Literal["cmd", "data"] = "data"
    name: str = "model_route"
    destination: str
    status: str
    latency_ms: int
    choice: str | None = None
    confidence: float | None = None

class EndOfTurnEvent(AgentEventBase):
    """Indicating the end of the turn."""

    type: Literal["cmd", "data"] = "data"
    name: str = "end_of_turn"
    text: str
    reason: Literal["jev", "fallback"] = "jev"
    confidence: float

class AgentInterruptEvent(AgentEventBase):
    """Indicating the interruption of the user by the agent."""

    type: Literal["cmd", "data"] = "data"
    name: str = "agent_interrupt"
    text: str
    reason: Literal["jev", "fallback"] = "jev"
    confidence: float

# ==== Unified Event Union ====

AgentEvent = Union[
    UserJoinedEvent,
    UserLeftEvent,
    ToolRegisterEvent,
    ASRResultEvent,
    ASRCommitTimeoutEvent,
    LLMResponseEvent,
    ModelRouteEvent,
    EndOfTurnEvent,
    AgentInterruptEvent,
]
