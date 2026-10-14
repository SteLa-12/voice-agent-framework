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