from dataclasses import dataclass
from time import monotonic

from .models import FloorState


@dataclass(frozen=True)
class ConversationSnapshot:
    committed_segments: tuple[str, ...]
    current_interim: str
    user_transcript: str
    version: int
    turn_epoch: int
    user_speaking: bool
    agent_speaking: bool
    agent_pending: bool
    last_user_speech_end_ts: float | None
    user_silence_ms: float
    floor: FloorState
    last_agent_request_id: str | None
    last_agent_utterance: str
    last_backchannel_ts: float | None


class ConversationState:
    def __init__(self) -> None:
        self.committed_segments: list[str] = []
        self.current_interim = ""
        self.version = 0
        self.turn_epoch = 0
        self.user_speaking = False
        self.agent_speaking = False
        self.agent_pending = False
        self.last_user_speech_end_ts: float | None = None
        self.last_agent_request_id: str | None = None
        self.last_agent_utterance = ""
        self.last_backchannel_ts: float | None = None
        self.last_agent_interrupt_ts: float | None = None

    def on_asr(self, text: str, final: bool) -> None:
        """Updates the current asr input buffer, by replacing the buffer with the new input.
        Whenever the ASR result is final, it is committed to the list of segments and the interim buffer is cleared.

        Args:
            text (str): The ASR result text.
            final (bool): Indicates whether the ASR result is final.
        """
        if final:
            self.committed_segments.append(text)
            self.current_interim = ""
        else:
            self.current_interim = text
        self.version += 1

    def user_transcript(self) -> str:
        """Returns the full user transcript, including committed segments and the current interim buffer."""
        segments = (*self.committed_segments, self.current_interim)
        return " ".join(segment for segment in segments if segment)

    def flush_user_turn(self) -> None:
        """Flushes the current user turn by clearing committed segments and the interim buffer, and increments the turn epoch."""
        self.committed_segments.clear()
        self.current_interim = ""
        self.turn_epoch += 1

    def on_vad_start(self) -> None:
        """Marks the start of user speech."""
        self.user_speaking = True
        self.last_backchannel_ts = None

    def on_vad_end(self) -> None:
        """Marks the end of user speech and records the timestamp."""
        self.user_speaking = False
        self.last_user_speech_end_ts = monotonic()

    def on_agent_text(self, request_id: str, text: str) -> None:
        """Updates the agent's last utterance and marks the agent as pending.

        Args:
            request_id (str): The unique identifier for the agent's request.
            text (str): The text of the agent's utterance.
        """
        # Mark a new request id as soon as the request id is different from the last one.
        if request_id != self.last_agent_request_id:
            self.last_agent_request_id = request_id
            self.last_agent_utterance = ""
        self.last_agent_utterance += text
        self.agent_pending = True

    def on_tts_audio_start(self) -> None:
        """Marks the start of agent's TTS audio playback."""
        self.agent_speaking = True
        self.agent_pending = False

    def on_tts_audio_end(self) -> None:
        """Marks the end of agent's TTS audio playback."""
        self.agent_speaking = False

    def floor(self) -> FloorState:
        """Determines the current floor state based on user and agent activity."""
        if self.user_speaking and self.agent_speaking:
            return FloorState.OVERLAP
        if self.user_speaking:
            return FloorState.USER
        if self.agent_speaking:
            return FloorState.AGENT
        if self.agent_pending:
            return FloorState.AGENT_PENDING
        return FloorState.IDLE

    def user_silence_ms(self) -> float:
        """Returns the duration of user silence in milliseconds since the last user speech ended."""
        if self.last_user_speech_end_ts is None:
            return 0.0
        return max(0.0, (monotonic() - self.last_user_speech_end_ts) * 1000)

    def on_agent_backchannel(self) -> None:
        self.last_backchannel_ts = monotonic()

    def on_agent_interrupt(self) -> None:
        self.last_agent_interrupt_ts = monotonic()

    def snapshot(self) -> ConversationSnapshot:
        """Creates a snapshot of the current conversation state."""
        return ConversationSnapshot(
            committed_segments=tuple(self.committed_segments),
            current_interim=self.current_interim,
            user_transcript=self.user_transcript(),
            version=self.version,
            turn_epoch=self.turn_epoch,
            user_speaking=self.user_speaking,
            agent_speaking=self.agent_speaking,
            agent_pending=self.agent_pending,
            last_user_speech_end_ts=self.last_user_speech_end_ts,
            user_silence_ms=self.user_silence_ms(),
            floor=self.floor(),
            last_agent_request_id=self.last_agent_request_id,
            last_agent_utterance=self.last_agent_utterance,
            last_backchannel_ts=self.last_backchannel_ts
        )