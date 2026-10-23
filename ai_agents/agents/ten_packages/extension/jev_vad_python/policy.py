from time import monotonic
from typing import Optional, Type

from .state import ConversationSnapshot, FloorState
from .models import Action, EmitAgentInterrupt, EmitEndOfTurn, EmitUserInterrupt, IgnoreUserBackchannel, NoOp, PlayBackchannel
from .config import JevVadConfig
from .jev_client import JevPrediction

def select_question(
        snapshot: ConversationSnapshot,
        backchannel_cooldown_ms: int
    ) -> Optional[set[str]]:
    """Selects the questions that should be asked to the Jev model API in the current state snapshot

    Args:
        snapshot (ConversationSnapshot): Snapshot of the current state
        backchannel_cooldown_ms (int): Duration between two concurrent backchannels

    Returns:
        Optional[set[str]]: Set of questions that should be asked to the Jev model
    """

    if snapshot.floor in [FloorState.AGENT, FloorState.OVERLAP]:
        return {"interrupt_detection", "backchannel_detection"}
    elif snapshot.floor in [FloorState.USER, FloorState.IDLE]:
        if snapshot.user_transcript != "" and snapshot.last_backchannel_ts is not None and (monotonic() - snapshot.last_backchannel_ts) * 1000 < backchannel_cooldown_ms:
            return {"end_of_turn", "interrupt_reproduction"}
        return {"end_of_turn", "interrupt_reproduction"}
    return None

def decide(
        prediction: JevPrediction,
        snapshot: ConversationSnapshot,
        config: JevVadConfig
    ) -> Type[Action]:
    """Returns the exact type of :class:`Action` that needs to be taken, taking into account the current state and the prediction made by Jev.

    Args:
        prediction (JevPrediction): Predictions made by the Jev decision model
        snapshot (ConversationSnapshot): Current conversation state
        config (JevVadConfig): Configuration of the Jev Voice Activity Manager

    Returns:
        Type[Action]: Action that needs to be executed based on the state and prediction.
    """

    if snapshot.agent_speaking:
        if prediction.is_backchannel_detection(config.backchannel_detection_threshold):
            return IgnoreUserBackchannel
        elif prediction.is_interrupt_detection(config.interruption_detection_threshold):
            return EmitUserInterrupt
    elif snapshot.user_speaking:
        if prediction.is_end_of_turn(config.end_of_turn_threshold):
            return EmitEndOfTurn
        elif (
            prediction.is_interrupt_reproduction(config.interruption_reproduction_threshold) and
            len(snapshot.current_interim) > config.min_words_for_agent_interrupt
            # TODO: also detect the cooldown period
        ):
            return EmitAgentInterrupt
        elif (
            prediction.is_backchannel_reproduction(config.backchannel_detection_threshold) and
            snapshot.user_silence_ms < config.backchannel_max_silence_ms and
            ((snapshot.last_backchannel_ts is not None and
            (snapshot.last_backchannel_ts - monotonic()) * 1000 > config.backchannel_cooldown_ms) or
            snapshot.last_backchannel_ts is None)
        ):
            return PlayBackchannel
    return NoOp
