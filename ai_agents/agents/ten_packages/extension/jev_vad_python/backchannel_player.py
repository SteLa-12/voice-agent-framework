import asyncio
import random
import wave
from dataclasses import dataclass
from pathlib import Path

from ten_runtime import AudioFrame, AudioFrameDataFmt, AsyncTenEnv
from .config import JevVadConfig


@dataclass(frozen=True)
class AudioClip:
	category: str
	path: Path
	data: bytes


class BackchannelPlayer:
	def __init__(self, config: JevVadConfig) -> None:
		self.config = config
		if config.audio_dest_frame_ms <= 0:
			raise ValueError("audio_dest_frame_ms must be positive")

		self.asset_dir = Path(config.backchannel_asset_dir)
		self.audio_dest_frame_ms = config.audio_dest_frame_ms
		self.clips: dict[str, list[AudioClip]] = {}
		self._last_clip_path: Path | None = None
		self._task: asyncio.Task[None] | None = None
		self.bytes_per_sample: int = 2
		self.bytes_per_second: float = self.config.backchannel_sample_rate / self.bytes_per_sample

	@property
	def is_playing(self) -> bool:
		"""Returns True when a backchannel clip is playing

        Returns:
            bool: Whether or not the clip is playing.
        """
		return (
			self._task is not None
			and not self._task.done()
			and not self._task.cancelling()
		)

	def load(self) -> None:
		"""Loads all the backchannels files inside the `/assets` folder."""
		self.clips.clear()
		self._last_clip_path = None
		if not self.asset_dir.is_dir():
			return

		for path in sorted(self.asset_dir.iterdir()):
			if not path.is_file() or path.suffix.lower() not in {".pcm", ".wav"}:
				continue

			data = self._read_clip(path)
			if not data:
				continue

			category = path.stem.rsplit("_", 1)[0] if "_" in path.stem else path.stem
			clip = AudioClip(category=category, path=path, data=data)
			self.clips.setdefault(category, []).append(clip)

	def _read_clip(self, path: Path) -> bytes:
		"""Reads the clip bytes from the file specified by `path`

        Args:
            path (Path): The path of the backchannel file

        Raises:
            ValueError: When the samplerate is not similar to that defined in the config file.
            ValueError: If the file has an uncomplete sample at the end of the file.

        Returns:
            bytes: The actual wave or pcm bytes in the file.
        """
		if path.suffix.lower() == ".pcm":
			data = path.read_bytes()
		else:
			with wave.open(str(path), "rb") as audio_file:
				if (
					audio_file.getframerate() != self.config.backchannel_sample_rate
					or audio_file.getnchannels() != 1
					or audio_file.getsampwidth() != self.bytes_per_sample
					or audio_file.getcomptype() != "NONE"
				):
					raise ValueError(
						f"{path} must be 16 kHz, mono, uncompressed 16-bit PCM"
					)
				data = audio_file.readframes(audio_file.getnframes())

		if len(data) % self.bytes_per_sample:
			raise ValueError(f"{path} has an incomplete 16-bit PCM sample")
		return data

	def play(self, ten_env: AsyncTenEnv) -> bool:
		"""Play a random backchannel clip. If no clips were loaded initially, the function won't play any backchannel and return `False`

        Args:
            ten_env (AsyncTenEnv): Ten Framework Environment handle

        Raises:
            Exception: Can't play two backchannels at the same time.

        Returns:
            bool: Whether or not the function has succeeded.
        """
		available = [clip for group in self.clips.values() for clip in group]
		if not available:
			return False
		if self.is_playing:
			raise Exception("Another backchannel audio file is already playing.")

		choices = [clip for clip in available if clip.path != self._last_clip_path]
		clip = random.choice(choices or available)
		self._last_clip_path = clip.path
		self._task = asyncio.create_task(self._play_clip(ten_env, clip.data))
		self._task.add_done_callback(self._clear_task)
		return True

	def _clear_task(self, task: asyncio.Task[None]) -> None:
		"""Clear the task at when finishing the task. Callback which should be handled automatically at the end of running the task.

        Args:
            task (asyncio.Task[None]): The running task.
        """
		if self._task is task:
			self._task = None

	async def _play_clip(self, ten_env: AsyncTenEnv, audio_data: bytes) -> None:
		"""Playing loop, which continuously sends the next audio chunk to the receiving extension.

        Args:
            ten_env (AsyncTenEnv): Ten Framework Environment handle
            audio_data (bytes): The audio bytes of the backchannel audio file.

        Raises:
            ValueError: Raises when `audio_dest_frame_ms` produces chunks which do not contain any bytes due to a too short chunk length.
        """
		chunk_size = int(self.bytes_per_second * self.audio_dest_frame_ms // 1000)
		chunk_size -= chunk_size % self.bytes_per_sample
		if chunk_size <= 0:
			raise ValueError("audio_dest_frame_ms produces an empty audio chunk")

        # Load chunks into audio frames and send them to the correct extension
		for offset in range(0, len(audio_data), chunk_size):
			chunk = audio_data[offset : offset + chunk_size]
			audio_frame = AudioFrame.create("pcm_frame")
			audio_frame.set_bytes_per_sample(self.bytes_per_sample)
			audio_frame.set_sample_rate(self.config.backchannel_sample_rate)
			audio_frame.set_number_of_channels(1)
			audio_frame.set_data_fmt(AudioFrameDataFmt.INTERLEAVE)
			audio_frame.set_samples_per_channel(len(chunk) // self.bytes_per_sample)
			audio_frame.alloc_buf(len(chunk))
			buf = audio_frame.lock_buf()
			buf[:] = chunk
			audio_frame.unlock_buf(buf)
			await ten_env.send_audio_frame(audio_frame)

			if offset + len(chunk) < len(audio_data):
				await asyncio.sleep(len(chunk) / self.bytes_per_second)

	def stop(self) -> None:
		"""Stop the currently running audio task"""
		if self._task is not None and not self._task.done():
			self._task.cancel()
