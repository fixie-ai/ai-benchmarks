import asyncio
import websockets
import json
import base64
import time
import logging
from typing import Iterator
import os
import argparse

# Read some settings from command line
parser = argparse.ArgumentParser()
parser.add_argument("--voice", "-v")
parser.add_argument("--model", "-m", default="eleven_monolingual_v1")
parser.add_argument("--text-chunker", action="store_true", default=False)
args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configuration section
voice_id = args.voice
# choices: eleven_monolingual_v1, eleven_english_v2, eleven_multilingual_v1, eleven_multilingual_v2
model = args.model
stability = 0.5
similarity_boost = False
chunk_length_schedule = [50]
xi_api_key = os.environ["ELEVEN_API_KEY"]
max_length = 1  # Maximum length for audio string truncation
delay_time = 0.0001  # Use this to simulate the token output speed of your LLM
try_trigger_generation = True
optimize_streaming_latency = "4"  # The default setting in the WS API is 4. Change it to 3 or lower to improve the pronunciation of numbers and dates to enable the text normalizer.
use_text_chunker = args.text_chunker
output_format = "mp3_44100"  # Output format of the generated audio. Must be one of: mp3_44100, pcm_16000, pcm_22050, pcm_24000, pcm_44100


def text_chunker(text: str) -> Iterator[str]:
    """
    Used during input streaming to chunk text blocks and set last char to space.
    Use this function to simulate the default behavior of the official 11labs Library.
    """
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""
    for i, char in enumerate(text):
        buffer += char
        if i < len(text) - 1:  # Check if this is not the last character
            next_char = text[i + 1]
            if buffer.endswith(splitters) and next_char == " ":
                logging.info(f"Chunked text: {buffer}")
                yield buffer if buffer.endswith(" ") else buffer + " "
                buffer = ""
    if buffer != "":
        logging.info(f"Chunked text: {buffer}")
        yield buffer + " "


def simulate_text_stream():
    """
    When use_text_chunker is True, use a single text chunk here to process via the text_chunker function from elevenlabs library.
    When use_text_chunker is False, you can simulate chunks of text from an LLM by adding more lines like this, in the above text_chunks list:
    text_chunks = [
        "Hello ",
        "World, ",
        "this ",
        "is ",
        "a ",
        "voice ",
        "sample! ",
    ]
    """
    text_chunks = [
        "Hello world! This is a sample of a streaming voice. ",
    ]
    for text_chunk in text_chunks:
        time.sleep(delay_time)
        yield text_chunk


def truncate_audio_string(audio_string):
    """
    Truncate audio string if it exceeds the max_length
    """
    if len(audio_string) > max_length:
        return audio_string[:max_length] + "..."
    return audio_string


async def text_to_speech():
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_type={model}&optimize_streaming_latency={optimize_streaming_latency}&output_format={output_format}"

    start_time = time.time()
    chunk_times = []
    first_chunk_received = False
    first_chunk_time = None

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        connection_open_time = time.time()
        time_to_open_connection = connection_open_time - start_time

        bos_message = {
            "text": " ",
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
            "generation_config": {"chunk_length_schedule": chunk_length_schedule},
            "xi_api_key": xi_api_key,
            "try_trigger_generation": try_trigger_generation,
        }
        await websocket.send(json.dumps(bos_message))

        for text_chunk in simulate_text_stream():
            if use_text_chunker:
                for chunk in text_chunker(text_chunk):
                    input_message = {
                        "text": chunk,
                        "try_trigger_generation": try_trigger_generation,
                    }
                    input_message_time = time.time()
                    logging.info(
                        f"[{input_message_time:.4f}] Sending input message: {chunk}"
                    )
                    await websocket.send(json.dumps(input_message))
            else:
                input_message = {
                    "text": text_chunk,
                    "try_trigger_generation": try_trigger_generation,
                }
                input_message_time = time.time()
                logging.info(
                    f"[{input_message_time:.4f}] Sending input message: {text_chunk}"
                )
                await websocket.send(json.dumps(input_message))

            try:
                start_waiting_time = time.time()
                logging.info(f"[{start_waiting_time:.4f}] Start waiting for response")
                response = await asyncio.wait_for(websocket.recv(), timeout=0.0001)
                end_waiting_time = time.time()
                logging.info(f"[{end_waiting_time:.4f}] End waiting for response")
                response_received_time = time.time()
                logging.info(f"[{response_received_time:.4f}] Response received")
                data = json.loads(response)

                data_copy = data.copy()

                if "audio" in data_copy:
                    data_copy["audio"] = truncate_audio_string(data_copy["audio"])

                logging.info(f"Server response: {data_copy}")

                if "audio" in data:
                    chunk = base64.b64decode(data["audio"])
                    logging.info("Received audio chunk")
                    chunk_received_time = time.time()
                    if not first_chunk_received:
                        first_chunk_received = True
                        first_chunk_time = chunk_received_time - connection_open_time
                        logging.info(
                            f"Time to receive first chunk after connection opened: {first_chunk_time:.4f} seconds"
                        )
                    chunk_times.append(chunk_received_time - connection_open_time)
                else:
                    logging.info("No audio data in the response")
            except asyncio.TimeoutError:
                pass

        eos_message = {"text": ""}
        eos_message_time = time.time()
        logging.info(f"[{eos_message_time:.4f}] Sending eos_message")
        await websocket.send(json.dumps(eos_message))

        while True:
            try:
                response = await websocket.recv()
                response_received_time = time.time()
                logging.info(f"[{response_received_time:.4f}] Response received")
                data = json.loads(response)
                audio = data.get("audio")
                if audio is not None:
                    truncated_audio = truncate_audio_string(data["audio"])
                    logging.info(f"Server response: {{'audio': '{truncated_audio}'}}")
                else:
                    logging.info("Server response:", data)
                await asyncio.sleep(0)

                if audio is not None:
                    chunk = base64.b64decode(data["audio"])
                    logging.info("Received audio chunk")
                    await asyncio.sleep(0)
                    chunk_received_time = time.time()
                    chunk_times.append(chunk_received_time - connection_open_time)
                else:
                    logging.info("No audio data in the response")
                    await asyncio.sleep(0)
                    break
            except websockets.exceptions.ConnectionClosed:
                logging.info("Connection closed")
                await asyncio.sleep(0)
                break

        connection_close_time = time.time()
        total_time_websocket_was_open = connection_close_time - connection_open_time

    logging.info("\n-----Latency Summary-----")
    logging.info(f"Time to open connection: {time_to_open_connection:.4f} seconds")
    if (
        first_chunk_time is not None
    ):  # Check if first_chunk_time is not None before trying to print it
        logging.info(
            f"Time to first chunk after connection opened: {first_chunk_time:.4f} seconds"
        )
    for i, chunk_time in enumerate(chunk_times, start=1):
        logging.info(
            f"Time to receive chunk {i} after connection opened: {chunk_time:.4f} seconds"
        )
    logging.info(
        f"Total time WebSocket connection was open: {total_time_websocket_was_open:.4f} seconds"
    )


asyncio.get_event_loop().run_until_complete(text_to_speech())
