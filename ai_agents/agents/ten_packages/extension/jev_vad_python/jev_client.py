import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria, SystemOneResponse

from ten_runtime.async_ten_env import AsyncTenEnv
from .state import ConversationSnapshot


SYSTEM_PROMPT = """
You predict turn-taking behavior in a live voice conversation. `utterance` is
the user's transcript so far. `agent_last_utterance` is the latest agent
request's generated text and may be partial. `agent_is_speaking` indicates
whether agent audio is currently playing; `user_is_speaking` indicates current
user speech; `user_silence_ms` is elapsed time since the user last stopped
speaking. Judge interruption and backchannel behavior relative to the agent's
speech: an interruption means the user is trying to take the floor while the
agent speaks, while a backchannel is a brief acknowledgment that does not take
the floor.
""".strip()


QUESTIONS = {
    "end_of_turn": Noul(
        instructions=(
            "Has the user finished their turn, so that the assistant should "
            "respond now?"
        ),
        criteria=NoulCriteria(
            true=(
                "The user said a complete thought, question, request or "
                "greeting and nothing indicates that more is coming."
            ),
            false=(
                "The user is cut off mid-sentence, trails off on a conjunction, "
                "filler or unfinished clause, or is clearly still building up "
                "to the actual request."
            ),
        ),
    ),
    "interrupt_detection": Noul(
        instructions=(
            "Is the user attempting to take the floor from the agent while "
            "the agent is speaking?"
        ),
        criteria=NoulCriteria(
            true=(
                "The user is clearly trying to take the turn from the agent, "
                "cutting off or speaking over the agent."
            ),
            false="The user is not attempting to take the floor from the agent.",
        ),
    ),
    "backchannel_detection": Noul(
        instructions=(
            "Is the user providing a brief backchannel to the agent's speech "
            "(for example, 'uh-huh' or 'I see') without taking the floor?"
        ),
        criteria=NoulCriteria(
            true=(
                "The user is briefly acknowledging the agent while it speaks, "
                "without trying to take the turn."
            ),
            false="The user is not providing a backchannel to the agent.",
        ),
    ),
    "interrupt_reproduction": Noul(
        instructions=(
            "Should the agent take over the turn and interrupt the user, for "
            "example to help when the user is searching for a word or hesitating?"
        ),
        criteria=NoulCriteria(
            true=(
                "The user is searching for a word or hesitating in a way the "
                "agent could help with by taking the turn."
            ),
            false="The user is not indicating that the agent should interrupt.",
        ),
    ),
    "backchannel_reproduction": Noul(
        instructions=(
            "Should the agent give a brief backchannel now to indicate that it "
            "is listening without taking the user's turn?"
        ),
        criteria=NoulCriteria(
            true="A brief acknowledgment would help show the agent is listening.",
            false="The agent should not produce a backchannel now.",
        ),
    ),
}


@dataclass
class JevPrediction:
    end_of_turn: Optional[float] = None
    """Probability, the turn of the user has ended."""

    interrupt_detection: Optional[float] = None
    """Probability the user is attempting to interrupt the agent."""

    backchannel_detection: Optional[float] = None
    """Probability interrupt by user is actually a backchannel, without intent to take the floor"""

    interrupt_reproduction: Optional[float] = None
    """Probability of the robot needing to take the turn and interrupt the user."""

    backchannel_reproduction: Optional[float] = None
    """Probability of the robot backchanneling to the user."""

    @classmethod
    def from_response(cls, data: SystemOneResponse) -> "JevPrediction":
        """Generate a JevPrediction class object from the SystemOneResponse returned by the Jev API

        Args:
            data (SystemOneResponse): Object returned by the Jev API containing the nouls.

        Returns:
            JevPrediction: An instance of JevPrediction containing the probabilities for each noul.
        """
        nouls = getattr(data, "nouls", None) or {}

        def get_probability(name: str) -> Optional[float]:
            answer = nouls.get(name)
            if answer is None:
                return None
            return getattr(answer, "noul", None)

        return cls(
            end_of_turn=get_probability("end_of_turn"),
            interrupt_detection=get_probability("interrupt_detection"),
            backchannel_detection=get_probability("backchannel_detection"),
            interrupt_reproduction=get_probability("interrupt_reproduction"),
            backchannel_reproduction=get_probability("backchannel_reproduction"),
        )

    def is_end_of_turn(self, thresh: float) -> bool:
        return self.end_of_turn is not None and self.end_of_turn >= thresh

    def is_interrupt_detection(self, thresh: float) -> bool:
        return self.interrupt_detection is not None and self.interrupt_detection >= thresh

    def is_backchannel_detection(self, thresh: float) -> bool:
        return self.backchannel_detection is not None and self.backchannel_detection >= thresh

    def is_interrupt_reproduction(self, thresh: float) -> bool:
        return self.interrupt_reproduction is not None and self.interrupt_reproduction >= thresh

    def is_backchannel_reproduction(self, thresh: float) -> bool:
        return self.backchannel_reproduction is not None and self.backchannel_reproduction >= thresh

class JevClient:
    def __init__(
        self,
        api_key: str,
        ten_env: AsyncTenEnv,
        jev_timeout_ms: int = 1500
    ) -> None:
        self.client = AsyncTypeSafeClient(api_key=api_key)
        self.ten_env = ten_env
        self.jev_timeout_ms = jev_timeout_ms

    async def get_prediction(
        self,
        snapshot: ConversationSnapshot,
        questions: set[str]
    ) -> Optional[JevPrediction]:
        """Make a prediction for the requested `questions` based on the current state snapshot.

        Args:
            snapshot (ConversationSnapshot): Snapshot of the current conversation state.
            questions (set[str]): Set of questions to be predicted by the Jev model.

        Raises:
            ValueError: If any of the requested questions are unknown.

        Returns:
            Optional[JevPrediction]: The :class:`JevPrediction` result from the Jev model, or None if the prediction failed. Returns empty :class:`JevPrediction` if no questions are provided.
        """

        unknown_questions = questions.difference(QUESTIONS)
        if unknown_questions:
            raise ValueError(f"Unknown Jev questions: {sorted(unknown_questions)}")

        if not questions:
            return JevPrediction()

        self.ten_env.log_debug(f"Getting prediction for utterance: {snapshot.user_transcript}")
        selected_questions = {
            name: QUESTIONS[name] for name in QUESTIONS if name in questions
        }
        try:
            response = await asyncio.wait_for(
                self.client.system_one(
                    state={
                        "system": SYSTEM_PROMPT,
                        "utterance": snapshot.user_transcript,
                        "agent_last_utterance": snapshot.last_agent_utterance,
                        "agent_is_speaking": snapshot.agent_speaking,
                        "user_is_speaking": snapshot.user_speaking,
                        "user_silence_ms": snapshot.user_silence_ms,
                    },
                    questions=selected_questions,
                ),
                timeout=self.jev_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            self.ten_env.log_error(
                f"Jev prediction timed out after {self.jev_timeout_ms} ms"
            )
            return None
        except Exception as exc:
            self.ten_env.log_error(f"Jev prediction failed: {exc}")
            return None

        self.ten_env.log_debug(
            f"Received response from Jev API: {response.model_dump_json()}"
        )
        return JevPrediction.from_response(response)