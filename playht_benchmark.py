import argparse
import os
import subprocess
from pathlib import Path
import aiohttp
import asyncio
import time
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)

API_KEY = os.environ.get("API_KEY")
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
USER_ID = os.environ.get("USER_ID")

# Defaults Text and voice settings
DEFAULT_TEXT = """Ah, these kids today! They don't know the struggle, I'll tell ya that much. Back in my day, the internet wasn't this instant-gratification paradise it is now. Oh no, it was a test of patience and determination. You'd sit down, ready to see what the "World Wide Web" had to offer, and then you'd have to endure that awful, ear-piercing dial-up tone. And you'd hope and pray that someone wasn't using the phone, because if they were, you were outta luck. No internet for you!"""
DEFAULT_VOICE = "s3://voice-cloning-zero-shot/7c339a9d-370f-4643-adf5-4134e3ec9886/mlae02/manifest.json"

# Audio settings
DEFAULT_QUALITY = "draft"
DEFAULT_OUTPUT_FORMAT = "mp3"
DEFAULT_SPEED = 1
DEFAULT_SAMPLE_RATE = 24000

# Random generator settings
DEFAULT_SEED = None
DEFAULT_TEMPERATURE = None

# Voice engine settings
DEFAULT_VOICE_ENGINE = "PlayHT2.0"
DEFAULT_EMOTION = "male_angry"
DEFAULT_VOICE_GUIDANCE = 2
DEFAULT_STYLE_GUIDANCE = 20

# argument parser
parser = argparse.ArgumentParser(description="Stream audio from server.")
parser.add_argument(
    "--text", default=DEFAULT_TEXT, help="Text to be converted to speech"
)
parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice to be used")
parser.add_argument("--quality", default=DEFAULT_QUALITY, help="Quality of the audio")
parser.add_argument(
    "--speed", default=DEFAULT_SPEED, type=int, help="Speed of the speech"
)
parser.add_argument(
    "--voice_engine", default=DEFAULT_VOICE_ENGINE, help="Voice engine to be used"
)
parser.add_argument("--emotion", default=DEFAULT_EMOTION, help="Emotion of the speech")
parser.add_argument(
    "--voice_guidance",
    default=DEFAULT_VOICE_GUIDANCE,
    type=int,
    help="Voice guidance level",
)
parser.add_argument(
    "--style_guidance",
    default=DEFAULT_STYLE_GUIDANCE,
    type=int,
    help="Style guidance level",
)

args = parser.parse_args()

# use parsed arguments
data = {
    "text": args.text,
    "voice": args.voice,
    "quality": args.quality,
    "output_format": DEFAULT_OUTPUT_FORMAT,
    "speed": args.speed,
    "sample_rate": DEFAULT_SAMPLE_RATE,
    "seed": DEFAULT_SEED,
    "temperature": DEFAULT_TEMPERATURE,
    "voice_engine": args.voice_engine,
    "emotion": args.emotion,
    "voice_guidance": args.voice_guidance,
    "style_guidance": args.style_guidance,
}

url = "https://play.ht/api/v2/tts/stream"
headers = {
    "AUTHORIZATION": f"Bearer {BEARER_TOKEN}",
    "X-USER-ID": USER_ID,
    "accept": "audio/mpeg",
    "content-type": "application/json",
}

start_time = time.perf_counter()

# initialize dictionary to store latency data
latency_data = {
    "headers_received": None,
    "first_chunk": None,
    "chunk_times": [],
    "total_time": None,
}


async def stream(response, start_time, data):
    mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
    mpv_process = subprocess.Popen(
        mpv_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    first_chunk = True
    bytes_received = 0

    while True:
        chunk = await response.content.read(4096)
        if not chunk:
            break

        if first_chunk:
            latency_data["first_chunk"] = time.perf_counter() - start_time
            first_chunk = False

        latency_data["chunk_times"].append(time.perf_counter() - start_time)

        mpv_process.stdin.write(chunk)
        bytes_received += len(chunk)
        print(
            f"Received chunk of size {len(chunk):<5} bytes | Total bytes received: {bytes_received:<6}",
            end="\r",
        )

    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()

    latency_data["total_time"] = time.perf_counter() - start_time


async def main():
    async with aiohttp.ClientSession() as session:
        logging.info("Sending request to the server...")
        async with session.post(url, headers=headers, json=data) as response:
            latency_data["headers_received"] = time.perf_counter() - start_time
            logging.info(f"Latency: {latency_data['headers_received']*1000:.2f} ms")
            logging.info(f"Status code: {response.status}")
            logging.info("-" * 40)
            logging.info(f"Text: \"{data['text']}\"")
            logging.info("-" * 40)
            if "audio/mpeg" in response.headers.get("Content-Type"):
                logging.info("Streaming audio...")
                logging.info("-" * 40)
                await stream(response, start_time, data)
            else:
                logging.error("No audio data in the response.")


# main function
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

# Latency Summary
logging.info("\n" + "=" * 40)
logging.info("LATENCY SUMMARY")
logging.info("-" * 40)
logging.info(f"Time to receive headers: {latency_data['headers_received']*1000:.2f} ms")
if latency_data["first_chunk"] is not None:
    logging.info(
        f"Time to first chunk after headers received: {latency_data['first_chunk']*1000:.2f} ms"
    )
logging.info(f"Total time: {latency_data['total_time']*1000:.2f} ms")
logging.info("=" * 40 + "\n")
