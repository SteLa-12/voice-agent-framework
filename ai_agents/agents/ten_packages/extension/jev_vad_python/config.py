from pydantic import BaseModel, Field

class JevVadConfig(BaseModel):
    api_key: str = Field(
        default="",
        description="Typesafe api key"
    )
    end_of_turn_threshold: float = Field(
        default=0.7,
        description="End of turn threshold"
    )
    backchannel_detection_threshold: float = Field(
        default=0.7,
        description="Backchannel detection threshold, whether or not the utterance is a backchannel."
    )
    interruption_detection_threshold: float = Field(
        default=0.7,
        description="Interruption detection threshold, detect whether the user started interrupting the agent."
    )
    interruption_reproduction_threshold: float = Field(
        default=0.7,
        description="Interruption reproduction threshold, should the agent interrupt the user."
    )
    backchannel_reproduction_threshold: float = Field(
        default=0.7,
        description="Backchannel reproduction threshold, whether or not the agent should reproduce backchannels."
    )
    fallback_end_of_turn_ms: int = Field(
        default=1500,
        description="Fallback threshold for when Jev end of turn detection does not fire."
    )
    backchannel_cooldown_ms: int = Field(
        default=4000,
        description="Amount of milliseconds that need to have passed before a new backchannel can be produced."
    )
    backchannel_max_silence_ms: int = Field(
        default=400,
        description="A backchannel is only allowed while the user is speaking or within this time after they stop."
    )
    agent_interrupt_cooldown_ms: int = Field(
        default=10000,
        description="Minimal duration between two consecutive interruptions by the robot, this can ensure a polite interruption policy, by regulating the amount of possible interruptions per time entitiy."
    )
    min_words_for_agent_interrupt: int = Field(
        default=4,
        description="Minimal amount of words that have to be spoken by the user before the robot can actually interrupt the speaker."
    )
    backchannel_asset_dir: str = Field(
        default="assets/backchannels",
        description="Directory of the prerecorded backchannel audio files."
    )
    backchannel_sample_rate: int = Field(
        default=16000,
        description="Sample rate at which the backchannels were recorded."
    )
    audio_dest_frame_ms: int = Field(
        default=10,
        description=""
    )
    jev_timeout_ms: int = Field(
        default=1500,
        description="Max duration for which the module need to wait for the result from the Jev API before going to fallback methods."
    )