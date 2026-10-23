import os

from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria, SystemOneResponse
from dataclasses import dataclass

from ten_runtime.async_ten_env import AsyncTenEnv

@dataclass
class JevPrediction:
    end_of_turn: float
    """Probability, the turn of the user has ended."""

    interrupt_detection: float
    """Probability the user is attempting to interrupt the agent."""

    backchannel_detection: float
    """Probability interrupt by user is actually a backchannel, without intent to take the floor"""

    interrupt_reproduction: float
    """Probability of the robot needing to take the turn and interrupt the user."""

    backchannel_reproduction: float
    """Probability of the robot backchanneling to the user."""

    @classmethod
    def from_response(cls, data: SystemOneResponse) -> "JevPrediction":
        return cls(
            end_of_turn=data.nouls["end_of_turn"].noul,
            interrupt_detection=data.nouls["interrupt_detection"].noul,
            backchannel_detection=data.nouls["backchannel_detection"].noul,
            interrupt_reproduction=data.nouls["interrupt_reproduction"].noul,
            backchannel_reproduction=data.nouls["backchannel_reproduction"].noul,
        )

    def is_end_of_turn(self, thresh: float) -> bool:
        return self.end_of_turn >= thresh

    def is_interrupt_detection(self, thresh: float) -> bool:
        return self.interrupt_detection >= thresh

    def is_backchannel_detection(self, thresh: float) -> bool:
        return self.backchannel_detection >= thresh

    def is_interrupt_reproduction(self, thresh: float) -> bool:
        return self.interrupt_reproduction >= thresh

    def is_backchannel_reproduction(self, thresh: float) -> bool:
        return self.backchannel_reproduction >= thresh

class JevClient:
    def __init__(self, api_key: str, ten_env: AsyncTenEnv):
        if api_key == "":
            api_key = os.environ.get("TYPESAFE_API_KEY", "")
        self.client = AsyncTypeSafeClient(api_key=api_key)
        self.ten_env = ten_env

    async def get_prediction(self, utterance: str) -> JevPrediction:
        self.ten_env.log_debug(f"Getting prediction for utterance: {utterance}")
        response = await self.client.system_one(
            state={
                "system": """
                You are a helpful robot that predicts turn-taking behavior in conversations. The system transcribes
                the speech of the person for you in a streaming way. The transcriptions will arrive incrementally
                and you need to decide in real-time whether the user has finished their turn, is attempting to interrupt,
                or is providing a backchannel. `utterance` contains the latest transcription of the user's speech
                on which you will make predictions regarding turn-taking behavior.
                """,
                "utterance": utterance,
            },
            questions={
                "end_of_turn": Noul(
                    instructions=(
                        "`utterance` is a live speech-to-text transcript of what a user has said to a "
                        "voice assistant so far. Has the user finished their turn, so that the assistant "
                        "should respond now?"
                    ),
                    criteria=NoulCriteria(
                        true=(
                            "The user said a complete thought, question, request or greeting and nothing "
                            "indicates that more is coming."
                        ),
                        false=(
                            "The user is cut off mid-sentence, trails off on a conjunction, filler or "
                            "unfinished clause, or is clearly still building up to the actual request."
                        )
                    )
                ),
                "interrupt_detection": Noul(
                    instructions=(
                        "`utterance` is a live speech-to-text transcript of what a user has said to a "
                        "voice assistant so far. Is the user attempting to interrupt the assistant?"
                    ),
                    criteria=NoulCriteria(
                        true=(
                            "The user is clearly trying to take the turn from the assistant, "
                            "cutting off or speaking over the assistant."
                        ),
                        false=(
                            "The user is not attempting to interrupt the assistant."
                        )
                    )
                ),
                "backchannel_detection": Noul(
                    instructions=(
                        "`utterance` is a live speech-to-text transcript of what a user has said to a "
                        "voice assistant so far. Is the user providing a backchannel (e.g., 'uh-huh', 'I see')?"
                    ),
                    criteria=NoulCriteria(
                        true=(
                            "The user is providing a backchannel, indicating they are listening but not taking the turn."
                        ),
                        false=(
                            "The user is not providing a backchannel."
                        )
                    )
                ),
                "interrupt_reproduction": Noul(
                    instructions=(
                        "`utterance` is a live speech-to-text transcript of what a user has said to a "
                        "voice assistant so far. Should the agent take over the turn and interrupt. "
                        "(i.e. is the user searching for a word, which the robot can potentially complete or assist with, "
                        "or is the user hesitating in a way that the agent could assist with)"
                    ),
                    criteria=NoulCriteria(
                        true=(
                            "The system can interrupt the user, taking over the turn from the assistant. "
                            "This is appropriate when the user is searching for a word or hesitating in a way that the agent could assist with."
                        ),
                        false=(
                            "The user is not indicating the system should interrupt."
                        )
                    )
                ),
                "backchannel_reproduction": Noul(
                    instructions=(
                        "`utterance` is a live speech-to-text transcript of what a user has said to a "
                        "voice assistant so far. Should the system reproduce a backchannel at this specific point? "
                        "Is it appropriate for the system to indicate that it is listening to the user?"
                    ),
                    criteria=NoulCriteria(
                        true=(
                            "The system should reproduce a backchannel, indicating it is listening but not taking the turn."
                        ),
                        false=(
                            "The system should not reproduce a backchannel."
                        )
                    )
                )
            }
        )

        self.ten_env.log_debug(f"Received response from Jev API: {response.model_dump_json()}")

        return JevPrediction.from_response(response)