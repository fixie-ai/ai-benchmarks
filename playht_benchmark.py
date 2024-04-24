import argparse
import asyncio
import dataclasses
import logging
import os
import subprocess
import time

import aiohttp
from pyht import client
from pyht.protos import api_pb2

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)

PLAYHT_API_KEY = os.environ.get("PLAYHT_API_KEY")
PLAYHT_USER_ID = os.environ.get("PLAYHT_USER_ID")

# Defaults Text and voice settings
DEFAULT_TEXT = "Ah, these kids today! They don't know the struggle, I'll tell ya that much. Back in my day, the internet wasn't this instant-gratification paradise it is now."
DEFAULT_VOICE_ID = "s3://voice-cloning-zero-shot/7c339a9d-370f-4643-adf5-4134e3ec9886/mlae02/manifest.json"

# Audio settings
DEFAULT_QUALITY = "draft"  # Quality of the audio. Default is "draft". Other options are low, medium, high, and premium.
DEFAULT_OUTPUT_FORMAT = "mp3"
DEFAULT_SPEED = 1  # Values 1 to 5
DEFAULT_SAMPLE_RATE = (
    24000  # Sample rate of the audio. A number between 8000 and 48000.
)

# Random generator settings
DEFAULT_SEED = (
    None  # Seed for the random generator. If None, a random seed will be used.
)
DEFAULT_TEMPERATURE = None  # Temperature for the random generator. Controls variance. If None, the model's default temperature will be used.

# Voice engine settings
DEFAULT_VOICE_ENGINE = "PlayHT2.0"  # Voice engine to be used. Default is "PlayHT1.0".
DEFAULT_EMOTION = "male_angry"  # Emotion of the speech. Only supported when voice_engine is set to "PlayHT2.0".
DEFAULT_VOICE_GUIDANCE = 2  # Voice guidance level. A number between 1 and 6. Lower numbers reduce the voice's uniqueness, higher numbers maximize its individuality. Only supported when voice_engine is set to "PlayHT2.0".
DEFAULT_STYLE_GUIDANCE = 20  # Style guidance level. A number between 1 and 30. Lower numbers reduce the strength of the chosen emotion, higher numbers create a more emotional performance. Only supported when voice_engine is set to "PlayHT2.0".

WARMUP_TEXT = "a"
CHUNK_SIZE = 4096

# argument parser
parser = argparse.ArgumentParser(description="Stream audio from server.")
parser.add_argument(
    "text", default=DEFAULT_TEXT, nargs="?", help="Text to be converted to speech"
)
parser.add_argument(
    "--play", "-p", default=False, action="store_true", help="Play the audio using mpv"
)
parser.add_argument("--transport", "-t", default="rest", help="Transport to be use")
parser.add_argument("--voice", "-v", default=DEFAULT_VOICE_ID, help="Voice to be used")
parser.add_argument(
    "--quality", "-q", default=DEFAULT_QUALITY, help="Quality of the audio"
)
parser.add_argument(
    "--speed", "-s", default=DEFAULT_SPEED, type=int, help="Speed of the speech"
)
parser.add_argument(
    "--format", "-f", default=DEFAULT_OUTPUT_FORMAT, help="Output format of the audio"
)
parser.add_argument(
    "--warmup",
    "-w",
    default=False,
    action="store_true",
    help="Perform a warmup call before generation",
)
parser.add_argument(
    "--voice-engine", default=DEFAULT_VOICE_ENGINE, help="Voice engine to be used"
)
parser.add_argument("--emotion", default=DEFAULT_EMOTION, help="Emotion of the speech")
parser.add_argument(
    "--voice-guidance",
    default=DEFAULT_VOICE_GUIDANCE,
    type=int,
    help="Voice guidance level",
)
parser.add_argument(
    "--style-guidance",
    default=DEFAULT_STYLE_GUIDANCE,
    type=int,
    help="Style guidance level",
)

args = parser.parse_args()


@dataclasses.dataclass
class LatencyData:
    def __init__(self):
        self.start_time = 0
        self.headers_received = 0
        self.first_chunk = 0
        self.chunk_times = []
        self.total_time = 0

    def start(self):
        self.start_time = time.perf_counter()

    def set_headers_received(self):
        self.headers_received = time.perf_counter() - self.start_time

    def set_first_chunk(self):
        self.first_chunk = time.perf_counter() - self.start_time

    def add_chunk_time(self):
        self.chunk_times.append(time.perf_counter() - self.start_time)

    def set_total_time(self):
        self.total_time = time.perf_counter() - self.start_time

    start_time: float
    headers_received: float
    chunk_times: list[float]
    total_time: float


async def stream_rest(response, latency_data: LatencyData):
    if args.play:
        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        mpv_process = None

    bytes_received = 0
    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
        latency_data.add_chunk_time()
        if mpv_process:
            mpv_process.stdin.write(chunk)
        bytes_received += len(chunk)
        print(
            f"Received chunk of size {len(chunk):<5} bytes | Total bytes received: {bytes_received:<6}",
            end="\r",
        )

    latency_data.set_total_time()
    if mpv_process:
        mpv_process.stdin.close()
        mpv_process.wait()


def create_rest_body(text: str):
    return {
        "text": text,
        "voice": args.voice,
        "quality": args.quality,
        "output_format": args.format,
        "speed": args.speed,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "seed": DEFAULT_SEED,
        "temperature": DEFAULT_TEMPERATURE,
        "voice_engine": args.voice_engine,
        "emotion": args.emotion,
        "voice_guidance": args.voice_guidance,
        "style_guidance": args.style_guidance,
    }


async def async_generate_rest(latency_data: LatencyData):
    url = "https://play.ht/api/v2/tts/stream"
    headers = {
        "AUTHORIZATION": f"Bearer {PLAYHT_API_KEY}",
        "X-USER-ID": PLAYHT_USER_ID,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        if args.warmup:
            logging.info("Sending warmup request...")
            async with session.post(
                url, headers=headers, json=create_rest_body(WARMUP_TEXT)
            ) as response:
                pass
        logging.info("Sending REST request...")
        if latency_data:
            latency_data.start()
        async with session.post(
            url, headers=headers, json=create_rest_body(args.text)
        ) as response:
            latency_data.set_headers_received()
            logging.info(f"Latency: {latency_data.headers_received*1000:.2f} ms")
            logging.info(f"Status code: {response.status}")
            logging.info("-" * 40)
            logging.info(f'Text: "{args.text}"')
            logging.info("-" * 40)
            if response.ok and "audio/mpeg" in response.headers.get("Content-Type"):
                logging.info("Streaming audio...")
                logging.info("-" * 40)
                await stream_rest(response, latency_data)
            else:
                logging.error("No audio data in the response.")


def generate_rest(latency_data: LatencyData):
    return asyncio.get_event_loop().run_until_complete(
        async_generate_rest(latency_data)
    )


def stream_grpc(gen, latency_data: LatencyData):
    if args.play:
        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        mpv_process = None

    bytes_received = 0
    for chunk in gen:
        latency_data.add_chunk_time()
        if mpv_process:
            mpv_process.stdin.write(chunk)
        bytes_received += len(chunk)
        print(
            f"Received chunk of size {len(chunk):<5} bytes | Total bytes received: {bytes_received:<6}",
            end="\r",
        )

    latency_data.set_total_time()
    if mpv_process:
        mpv_process.stdin.close()
        mpv_process.wait()


def generate_grpc(latency_data: LatencyData):
    advanced = client.Client.AdvancedOptions(grpc_addr="prod.turbo.play.ht:443")
    grpc_client = client.Client(PLAYHT_USER_ID, PLAYHT_API_KEY, advanced=advanced)
    if args.format == "mp3":
        format = api_pb2.FORMAT_MP3
    elif args.format == "wav":
        format = api_pb2.FORMAT_WAV
    else:
        logging.error("Invalid format")
        exit(1)
    options = client.TTSOptions(format=format, voice=args.voice, quality=args.quality)
    if args.warmup:
        logging.info("Sending warmup request...")
        list(grpc_client.tts(WARMUP_TEXT, options))
    logging.info("Sending GRPC request...")
    latency_data.start()
    result = grpc_client.tts(args.text, options)
    header = next(result)
    latency_data.set_headers_received()
    logging.info(f"Latency: {latency_data.headers_received*1000:.2f} ms")
    logging.info("-" * 40)
    logging.info(f'Text: "{args.text}"')
    logging.info("-" * 40)
    logging.info("Streaming audio...")
    logging.info("-" * 40)
    stream_grpc(result, latency_data)
    grpc_client.close()


def main():
    latency_data = LatencyData()
    if args.transport == "rest":
        generate_rest(latency_data)
    elif args.transport == "grpc":
        generate_grpc(latency_data)
    else:
        logging.error("Invalid transport")
        exit(1)

    # Latency Summary
    logging.info("\n" + "=" * 40)
    logging.info("LATENCY SUMMARY")
    logging.info("-" * 40)
    logging.info(
        f"Time to receive headers: {latency_data.headers_received*1000:.2f} ms"
    )
    if latency_data.chunk_times:
        logging.info(f"Time to first chunk: {latency_data.chunk_times[0]*1000:.2f} ms")
    logging.info(f"Total time: {latency_data.total_time*1000:.2f} ms")
    logging.info("=" * 40 + "\n")


if __name__ == "__main__":
    main()
