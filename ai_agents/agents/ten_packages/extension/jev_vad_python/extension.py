#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import os

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

from .jev_client import JevClient
from .config import JevVadConfig
from .models import EndOfTurnPayload


class JevVadExtension(AsyncExtension):
    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log(LogLevel.DEBUG, "on_init")

        config_json, _ = await ten_env.get_property_to_json()
        try:
            self.config = JevVadConfig.model_validate_json(config_json)
        except Exception as e:
            ten_env.log_error(
                f"invalid property: {e}", category=LOG_CATEGORY_KEY_POINT
            )
            self.config = None

        assert self.config is not None

        typesafe_api_key: str | None = self.config.api_key
        if typesafe_api_key == "":
            if os.getenv("TYPESAFE_API_KEY") is not None:
                typesafe_api_key = os.getenv("TYPESAFE_API_KEY")
            else:
                ten_env.log_error(
                    "TYPESAFE API key is not set", category=LOG_CATEGORY_KEY_POINT
                )
                return

        self.jev_client = JevClient(typesafe_api_key, ten_env) # type: ignore

        self.ten_env = ten_env

        self.transcript_buffer = []

    async def on_start(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log(LogLevel.DEBUG, "on_start")

        # IMPLEMENT: read properties, initialize resources

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log(LogLevel.DEBUG, "on_stop")

        # IMPLEMENT: clean up resources

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log(LogLevel.DEBUG, "on_deinit")

    async def on_cmd(self, ten_env: AsyncTenEnv, cmd: Cmd) -> None:
        cmd_name = cmd.get_name()
        ten_env.log(LogLevel.DEBUG, "on_cmd name {}".format(cmd_name))

        # IMPLEMENT: process cmd

        cmd_result = CmdResult.create(StatusCode.OK, cmd)
        await ten_env.return_result(cmd_result)

    async def on_data(self, ten_env: AsyncTenEnv, data: Data) -> None:
        data_name = data.get_name()
        ten_env.log(LogLevel.DEBUG, "on_data name {}".format(data_name))

        if data_name == "asr_result":
            asr_result, err = data.get_property_string("text")
            if err is not None:
                ten_env.log_error(f"Failed to read 'text' from asr_result: {err}")
                return
            self.ten_env.log(LogLevel.DEBUG, f"Received ASR result: {asr_result}")
            self.transcript_buffer.append(asr_result)
            await self._handle_decision_call()

    async def _handle_decision_call(self) -> None:
        """Callback for decision making and correct system handling (based on prediction)"""

        if not self.transcript_buffer:
            self.ten_env.log(LogLevel.DEBUG, "Transcript buffer is empty, skipping decision call")
            return

        transcription = " ".join(self.transcript_buffer)
        probs = await self.jev_client.get_prediction(transcription)
        self.ten_env.log(LogLevel.INFO, f"Jev prediction: {probs}")

        assert self.config is not None

        if probs.is_end_of_turn(self.config.end_of_turn_threshold):
            self.ten_env.log(LogLevel.INFO, "End of turn detected")

            props = EndOfTurnPayload(
                text = " ".join(self.transcript_buffer),
                reason="jev",
                confidence=probs.end_of_turn,
            )

            self.transcript_buffer.clear()

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
