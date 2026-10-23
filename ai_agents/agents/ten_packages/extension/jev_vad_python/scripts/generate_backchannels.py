"""Offline helper that renders backchannel clips with ElevenLabs.

Writes raw 16 kHz mono 16-bit PCM files to assets/backchannels/ using the
naming scheme `<category>_<n>.pcm` expected by BackchannelPlayer.

Usage:
    ELEVENLABS_TTS_KEY=... python scripts/generate_backchannels.py [--variants 2]
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# DEFAULT CONFIG FOR THE AUDIO GENERATION
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
# Keep in sync with the tts node in voice-agent/tenapp/property.json.
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "pcm_16000"
DEFAULT_PHRASES = ["mhm", "uh-huh", "yeah", "right", "I see", "okay"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "backchannels"


def slugify(text: str) -> str:
    """Generates a correct file name for the input string

    Args:
        text (str): Name of the audio file

    Raises:
        ValueError: Raised when no slugged name could be created

    Returns:
        str: Returns the slugged name for the file.
    """
    slug = re.sub(r"[^a-z0-9-]+", "_", text.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive a file name from {text!r}")
    return slug

def synthesize(text: str, api_key: str, voice_id: str, model_id: str) -> bytes:
    """Creates the corresponding audio bytes for the input text

    Args:
        text (str): Backchannel text
        api_key (str): Api key for Elevenlabs
        voice_id (str): used voice id for the generation
        model_id (str): model id used for generation

    Raises:
        RuntimeError: Raises error when no audio is being returned

    Returns:
        bytes: Audio bytes after generation
    """
    url = API_URL.format(voice_id=urllib.parse.quote(voice_id, safe=""))
    url += "?" + urllib.parse.urlencode({"output_format": OUTPUT_FORMAT})
    body = json.dumps({"text": text, "model_id": model_id}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()

    if not data:
        raise RuntimeError(f"empty audio returned for {text!r}")
    # Drop a trailing odd byte so the clip contains whole 16-bit samples.
    return data[: len(data) - len(data) % 2]


def main() -> int:
    if __doc__ is None:
        return 1
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phrases", nargs="*", default=DEFAULT_PHRASES)
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--variants", type=int, default=1,
                        help="number of takes to generate per phrase")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ELEVENLABS_TTS_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Set ELEVENLABS_TTS_KEY (or ELEVENLABS_API_KEY).", file=sys.stderr)
        return 1
    if args.variants < 1:
        parser.error("--variants must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for phrase in args.phrases:
        category = slugify(phrase)
        for index in range(1, args.variants + 1):
            path = args.output_dir / f"{category}_{index}.pcm"
            if path.exists() and not args.overwrite:
                print(f"skip  {path.name} (exists)")
                continue
            try:
                audio = synthesize(phrase, api_key, args.voice_id, args.model_id)
            except (urllib.error.URLError, RuntimeError) as exc:
                print(f"fail  {path.name}: {exc}", file=sys.stderr)
                failures += 1
                continue
            path.write_bytes(audio)
            print(f"wrote {path.name} ({len(audio) / 32000:.2f}s)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
