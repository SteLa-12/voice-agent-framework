#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import asyncio
import os
from time import monotonic

from ten_runtime import (
    AudioFrame,
    VideoFrame,
    AsyncExtension,
    AsyncTenEnv,
    Cmd,
    StatusCode,
    CmdResult,
    Data,
    LogLevel,
)

from ten_ai_base.const import LOG_CATEGORY_KEY_POINT

from .backchannel_player import BackchannelPlayer
from .jev_client import JevClient, QUESTIONS
from .policy import select_question, decide
from .config import JevVadConfig
from .models import EmitAgentInterrupt, EmitEndOfTurn, EmitUserInterrupt, EndOfTurnPayload, Action, IgnoreUserBackchannel, PlayBackchannel
from .state import ConversationSnapshot, ConversationState
from .helper import _send_cmd, _send_data


class JevVadExtension(AsyncExtension):
    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log(LogLevel.DEBUG, "on_init")

        # Loading the config file
        config_json, _ = await ten_env.get_property_to_json()
        try:
            self.config = JevVadConfig.model_validate_json(config_json)
        except Exception as e:
            ten_env.log_error(
                f"invalid property: {e}", category=LOG_CATEGORY_KEY_POINT
            )
            self.config = None

        # Assert that config file is correctly loaded
        assert self.config is not None

        # Fallback options for when api_key is not correctly loaded
        typesafe_api_key: str | None = self.config.api_key
        if typesafe_api_key == "":
            env_api_key = os.getenv("TYPESAFE_API_KEY")
            if env_api_key is None:
                ten_env.log_error(
                    "TYPESAFE API key is not set", category=LOG_CATEGORY_KEY_POINT
                )
                return
            typesafe_api_key = env_api_key

        # Create the Jev Client for Jev decisions
        self.jev_client = JevClient(
            typesafe_api_key, ten_env, self.config.jev_timeout_ms
        )  # type: ignore

        self.ten_env = ten_env

        # Initiate the empty state
        self.conversation_state = ConversationState()

        # Initiate backchannel playback module
        self.backchannel_player = BackchannelPlayer(self.config)

        # Event for evaluation request
        self._evaluation_event = asyncio.Event()

        # Starting the fallback timer
        self._stop_fallback_timer_event = asyncio.Event()
        self._fallback_timer_worker = asyncio.create_task(self._fallback_timer())

    async def on_start(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log(LogLevel.DEBUG, "on_start")

        # Load the required backchannel audio files
        self.backchannel_player.load()

        # Startup evaluation task for Jev predictions
        self._evaluation_worker_task = asyncio.create_task(self._evaluation_worker())

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log(LogLevel.DEBUG, "on_stop")

        # Stop the backchannel player
        self.backchannel_player.stop()

        # Cancel the evaluation worker running task
        if self._evaluation_worker_task is not None:
            self._evaluation_worker_task.cancel()

        self._cancel_fallback_timer()

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log(LogLevel.DEBUG, "on_deinit")

    async def on_cmd(self, ten_env: AsyncTenEnv, cmd: Cmd) -> None:
        cmd_name = cmd.get_name()
        ten_env.log(LogLevel.DEBUG, "on_cmd name {}".format(cmd_name))

        if cmd_name == "start_of_sentence":
            self.conversation_state.on_vad_start()

            await self._stop_fallback_timer()

        elif cmd_name == "end_of_sentence":
            self.conversation_state.on_vad_end()

            await self._reset_fallback_timer()

            await self._request_evaluation()

        # Return success message
        cmd_result = CmdResult.create(StatusCode.OK, cmd)
        await ten_env.return_result(cmd_result)

    async def on_data(self, ten_env: AsyncTenEnv, data: Data) -> None:
        data_name = data.get_name()
        ten_env.log(LogLevel.DEBUG, "on_data name {}".format(data_name))

        # Depending on the data name, route differently
        if data_name == "asr_result":

            # Retrieve transcript from the data package
            asr_result, err = data.get_property_string("text")
            if err is not None:
                ten_env.log_error(f"Failed to read 'text' from asr_result: {err}")
                return
            final, err = data.get_property_bool("final")
            if err is not None:
                ten_env.log_error(f"Failed to read 'final' from asr_result: {err}")
                return
            self.ten_env.log(LogLevel.DEBUG, f"Received ASR result: {asr_result}")

            # Add new transcript to the state
            self.conversation_state.on_asr(asr_result, final)

            await self._request_evaluation()

        elif data_name == "asr_finalize_end":

            self.ten_env.log_debug(f"Retrieved message with name: {data_name}")

        elif data_name == "agent_text":

            # Retrieve text from the data package
            text, err = data.get_property_string("text")
            if err is not None:
                ten_env.log_error(f"Failed to read 'text' from agent_text: {err}")
                return
            request_id, err = data.get_property_string("request_id")
            if err is not None:
                ten_env.log_error(f"Failed to read 'request_id' from agent_text: {err}")
                return
            self.ten_env.log(LogLevel.DEBUG, f"Received agent text: {text}")

            self.conversation_state.on_agent_text(request_id, text)

        elif data_name == "tts_audio_start":

            # Set state accordingly
            self.conversation_state.on_tts_audio_start()

            # Stop playing any backchannel (to prevent backchannels during robot speech)
            self.backchannel_player.stop()

        elif data_name == "tts_audio_end":

            # Set state accordingly
            self.conversation_state.on_tts_audio_end()

    async def _handle_decision_call(self, snapshot: ConversationSnapshot) -> None:
        """Callback for decision making and correct system handling (based on prediction)"""

        if not snapshot.user_transcript:
            self.ten_env.log(LogLevel.DEBUG, "Transcript buffer is empty, skipping decision call")
            return

        probs = await self.jev_client.get_prediction(snapshot, set(QUESTIONS))
        if probs is None:
            return
        if (
            snapshot.version != self.conversation_state.version
            or snapshot.turn_epoch != self.conversation_state.turn_epoch
        ):
            return
        self.ten_env.log(LogLevel.INFO, f"Jev prediction: {probs}")

        assert self.config is not None

        if (
            probs.end_of_turn is not None
            and probs.is_end_of_turn(self.config.end_of_turn_threshold)
        ):
            self.ten_env.log(LogLevel.INFO, "End of turn detected")

            props = EndOfTurnPayload(
                text=snapshot.user_transcript,
                reason="jev",
                confidence=probs.end_of_turn,
            )

            self.conversation_state.flush_user_turn()

            data_obj = Data.create("end_of_turn")

            data_obj.set_property_from_json(
                None,
                props.model_dump_json()
            )

            # Sending message to any node that is listening to this topic
            await self.ten_env.send_data(data_obj)

        if probs.is_interrupt_detection(self.config.interruption_detection_threshold):
            self.ten_env.log(LogLevel.INFO, "Interrupt detection detected")

            cmd_obj = Cmd.create("interrupt")

            await self.ten_env.send_cmd(cmd_obj)

        if probs.is_backchannel_detection(self.config.backchannel_detection_threshold):
            self.ten_env.log(LogLevel.INFO, "Backchannel detection detected")
        if probs.is_interrupt_reproduction(self.config.interruption_reproduction_threshold):
            self.ten_env.log(LogLevel.INFO, "Interrupt reproduction detected")
        if probs.is_backchannel_reproduction(self.config.backchannel_reproduction_threshold):
            self.ten_env.log(LogLevel.INFO, "Backchannel reproduction detected")
            self.backchannel_player.play(self.ten_env)

    async def on_audio_frame(
        self, ten_env: AsyncTenEnv, audio_frame: AudioFrame
    ) -> None:
        audio_frame_name = audio_frame.get_name()
        ten_env.log(LogLevel.DEBUG, "on_audio_frame name {}".format(audio_frame_name))

        # IMPLEMENT: process audio frame

    async def on_video_frame(
        self, ten_env: AsyncTenEnv, video_frame: VideoFrame
    ) -> None:
        video_frame_name = video_frame.get_name()
        ten_env.log(LogLevel.DEBUG, "on_video_frame name {}".format(video_frame_name))

        # IMPLEMENT: process video frame

    async def _evaluation_worker(self) -> None:
        """Evaluation worker loop, iterates over the evaluation process."""
        try:
            while True:
                # Wait as long as the evaluation event flag is not being set
                while not self._evaluation_event.is_set():
                    await asyncio.sleep(0.2)

                # Reset the evaluation flag
                self._evaluation_event.clear()

                # Take a snapshot of the current state
                snapshot = self.conversation_state.snapshot()

                # Select the questions that should be asked to Jev based on the current state
                assert self.config is not None
                questions = select_question(snapshot, self.config)

                # If no questions do conform to the current state, the loop is continued
                if questions is None:
                    continue

                turn_epoch_before_call = self.conversation_state.turn_epoch

                # Request a prediction from Jev
                prediction = await self.jev_client.get_prediction(snapshot, questions)

                # If the turn has switched during evaluation (e.g. during time-out or vad end of turn), the prediction is skipped, because it is no longer relevant
                if self.conversation_state.turn_epoch != self.conversation_state.turn_epoch:
                    continue

                # TODO: Is this usefull? It also drops it if the floor changed between `AGENT` and the user states.

                if prediction is None:
                    continue

                # Decide upon an action based on this prediction
                action = decide(prediction, snapshot, self.config)

                await self._execute(action)

        except asyncio.exceptions.CancelledError:
            self.ten_env.log_debug("Evaluation worker is being cancelled.")
            return

        except ValueError as e:
            self.ten_env.log_error(f"An error occurred during evaluation: {e}")


    async def _request_evaluation(self) -> None:
        """Sends event to evaluation worker loop to start an evaluation."""
        self._evaluation_event.set()

    async def _reset_fallback_timer(self) -> None:
        """Starts a fallback timer loop which triggers after `fallback_end_of_turn_ms` amount of milliseconds to ensure the turn switches even if Jev misses the turn."""
        # Resetting the stop fallback timer to ensure it starts running again
        self._stop_fallback_timer_event.clear()

    async def _stop_fallback_timer(self) -> None:
        """Stops the fallback timer, disabling end of turn firing."""
        # Set the fallback timer stop event to disable it
        self._stop_fallback_timer_event.set()

    async def _fallback_timer(self) -> None:
        """The fallback timer triggers as soon as the end of turn fallback is triggered."""

        self._fallback_timer_starting_point: float | None = None

        try:
            while True:
                if self._stop_fallback_timer_event.is_set():
                    # When the fallback timer event is triggered, start timer

                    if self._fallback_timer_starting_point is None:
                        self._fallback_timer_starting_point = monotonic()

                    if (self.config is not None and
                        (monotonic() - self._fallback_timer_starting_point) * 1000 > self.config.fallback_end_of_turn_ms):
                        # Start end of turn

                        # Retrieve snapshot
                        snapshot = self.conversation_state.snapshot()
                        payload = EndOfTurnPayload(
                            text=snapshot.user_transcript,
                            reason="fallback",
                            confidence=1
                        )

                        # Send end_of_turn data
                        await _send_data(
                            self.ten_env,
                            data_name="end_of_turn",
                            payload=payload
                        )

                        # Flush the end of user turn
                        self.conversation_state.flush_user_turn()

                        # Robot put back to listening state
                        self.conversation_state.agent_speaking = False
                        self.conversation_state.agent_pending = True

                        self.backchannel_player.stop()

                        self._stop_fallback_timer_event.clear()

                await asyncio.sleep(0.2)

        except asyncio.exceptions.CancelledError:
            self.ten_env.log_debug("Stopping fallback timer.")
            return


    def _cancel_fallback_timer(self) -> None:
        """Cancels the fallback timer and stops the worker."""
        if self._fallback_timer_worker is not None:
            self._fallback_timer_worker.cancel()

    async def _execute(self, action: Action, *args) -> None:
        """Executes the right calls depending on the action that is being given as input.

        Args:
            action (Type[Action]): The action that should be performed.
        """

        if type(action) is EmitEndOfTurn:

            # Send end_of_turn data
            await _send_data(
                self.ten_env,
                data_name="end_of_turn",
                payload=action.payload
            )

            # Flush the end of user turn
            self.conversation_state.flush_user_turn()

            # Robot put back to listening state
            self.conversation_state.agent_speaking = False
            self.conversation_state.agent_pending = True

            self._stop_fallback_timer_event.clear()

            self.backchannel_player.stop()

        elif type(action) is EmitUserInterrupt:

            # Send interrupt cmd command
            await _send_cmd(
                self.ten_env,
                cmd_name="interrupt",
                payload=action.payload
            )

            self.conversation_state.agent_speaking = False

        elif type(action) is IgnoreUserBackchannel:

            # Removes backchannel from current input buffer
            self.conversation_state.flush_user_turn()

        elif type(action) is PlayBackchannel:

            # Play the backchannel audio
            self.backchannel_player.play(self.ten_env)

            self.conversation_state.on_agent_backchannel()

        elif type(action) is EmitAgentInterrupt:

            # Send agent interrupt
            await _send_data(
                self.ten_env,
                data_name="agent_interrupt",
                payload=action.payload
            )

            self.conversation_state.flush_user_turn()

            self.conversation_state.agent_pending = True

            self.conversation_state.on_agent_interrupt()
