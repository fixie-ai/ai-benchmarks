import requests
import json
import time
import os
import argparse
import asyncio
import websockets
import base64
import logging
from typing import Iterator

logging.basicConfig(level=logging.INFO)

# Defaults for both scripts
DEFAULT_SAMPLES = 10
DEFAULT_TEXT = "Hello World!"
DEFAULT_MODEL_ID = "eleven_monolingual_v1"
DEFAULT_LATENCY_OPTIMIZER = 4
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
DEFAULT_OUTPUT_FORMAT = "mp3_44100"
DEFAULT_STABILITY = 0.5
DEFAULT_SIMILARITY_BOOST = False
DEFAULT_XI_API_KEY = os.environ["ELEVEN_API_KEY"],

# Configuration for HTTP API
DEFAULT_CHUNK_SIZE = 7868

# Configuration for WebSocket API
chunk_length_schedule = [50]
max_length = 10  # Maximum length for audio string truncation
delay_time = 0.0001  # Use this to simulate the token output speed of your LLM
try_trigger_generation = True


# Argument parsing
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''\
The script allows for comprehensive benchmarking of the 11Labs API for text-to-speech generation to achieve the lowest possible latency, given any combination of parameters.
''')

API_group = parser.add_argument_group('API Type')
API_group.add_argument("--API", choices=["http", "websocket"], required=True,
                       help="API type: 'http' or 'websocket'")

input_group = parser.add_argument_group('Input Parameters')
input_group.add_argument("--text", default=DEFAULT_TEXT,
                         help="Input text for speech synthesis")
input_group.add_argument("--model", default=DEFAULT_MODEL_ID,
                         help="Model ID for speech synthesis. Options: 'eleven_monolingual_v1', 'eleven_english_v2', 'eleven_multilingual_v1', 'eleven_multilingual_v2'")

output_group = parser.add_argument_group('Output Parameters')
output_group.add_argument("--num_samples", type=int, default=DEFAULT_SAMPLES,
                          help="Number of speech samples to generate")
output_group.add_argument("--output_format", default=DEFAULT_OUTPUT_FORMAT,
                          help="Speech output format. Options: 'mp3_44100', 'pcm_16000', 'pcm_22050', 'pcm_24000', 'pcm_44100'")

http_group = parser.add_argument_group('HTTP API Parameters')
http_group.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Size of the first playable chunk in bytes, default is 7868")

websocket_group = parser.add_argument_group('WebSocket API Parameters')
websocket_group.add_argument("--latency_optimizer", type=int, default=DEFAULT_LATENCY_OPTIMIZER,
                             help="Latency optimization level. Default is 4. Lower to 3 or less to improve pronunciation of numbers and dates.")
websocket_group.add_argument("--text_chunker", action="store_true", default=False,
                             help="Enable text chunker for input streaming. This chunks text blocks and sets last char to space, simulating the default behavior of the 11labs Library.")

general_group = parser.add_argument_group('General Parameters')
general_group.add_argument("--voice_id", default=DEFAULT_VOICE_ID,
                           help="ID of the voice for speech synthesis")

args = parser.parse_args()



# Text chunker function
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
                yield buffer if buffer.endswith(" ") else buffer + " "
                buffer = ""
    if buffer != "":
        yield buffer + " "

# Simulate text stream function
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
    text_chunks = ["Hello world! This is a sample of a streaming voice. "]
    for text_chunk in text_chunks:
        time.sleep(delay_time)
        yield text_chunk

# Truncate audio string function
def truncate_audio_string(audio_string):
    """
    Truncate audio string if it exceeds the max_length
    """
    if len(audio_string) > max_length:
        return audio_string[:max_length] + "..."
    return audio_string

# HTTP API request function
def http_api_request():
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{args.voice_id}/stream?optimize_streaming_latency={args.latency_optimizer}"
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": DEFAULT_XI_API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "text": args.text,
        "model_id": args.model,
        "voice_settings": {"stability": DEFAULT_STABILITY, "similarity_boost": DEFAULT_SIMILARITY_BOOST},
    }
    response_latencies = []
    chunk_latencies = []
    for i in range(args.num_samples):
        print(f"\nAPI Call {i+1}:")
        start_time = time.perf_counter()
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        if not response.ok:
            print("Error: " + response.json()["detail"]["message"])
            exit(1)
        response_received_time = time.perf_counter()
        response_latency = (response_received_time - start_time) * 1000
        response_latencies.append(response_latency)
        print(f"  Initial Response (Header) Time: {response_latency:.2f} ms")
        audio_data = b""
        for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
            if chunk:
                audio_data += chunk
                if len(audio_data) >= args.chunk_size:  
                    chunk_received_time = time.perf_counter()
                    chunk_latency = (chunk_received_time - start_time) * 1000
                    chunk_latencies.append(chunk_latency)
                   
                    print(f"  First Playable Chunk (Body) Time: {chunk_latency:.2f} ms")
                    break

    average_response_latency = sum(response_latencies) / len(response_latencies)
    median_response_latency = sorted(response_latencies)[len(response_latencies) // 2]
    average_chunk_latency = sum(chunk_latencies) / len(chunk_latencies)
    median_chunk_latency = sorted(chunk_latencies)[len(chunk_latencies) // 2]
    return average_response_latency, median_response_latency, average_chunk_latency, median_chunk_latency

async def websocket_api_request():
    logging.basicConfig(level=logging.INFO)  # Configure logging inside the function
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{args.voice_id}/stream-input?model_type={args.model}&optimize_streaming_latency={args.latency_optimizer}&output_format={args.output_format}"
    start_time = time.time()  # Record the time before the request is made
    chunk_times = []
    first_chunk_received = False
    first_chunk_time = None
    async with websockets.connect(uri) as websocket:
        connection_open_time = time.time()
        time_to_open_connection = connection_open_time - start_time
        bos_message = {
            "text": " ",
            "voice_settings": {
                "stability": DEFAULT_STABILITY,
                "similarity_boost": DEFAULT_SIMILARITY_BOOST,
            },
            "generation_config": {"chunk_length_schedule": chunk_length_schedule},
            "xi_api_key": DEFAULT_XI_API_KEY,
            "try_trigger_generation": try_trigger_generation,
        }
        await websocket.send(json.dumps(bos_message))
        for text_chunk in simulate_text_stream():
            if args.text_chunker:
                for chunk in text_chunker(text_chunk):
                    input_message = {
                        "text": chunk,
                        "try_trigger_generation": try_trigger_generation,
                    }
                    await websocket.send(json.dumps(input_message))
            else:
                input_message = {
                    "text": text_chunk,
                    "try_trigger_generation": try_trigger_generation,
                }
                await websocket.send(json.dumps(input_message))
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=delay_time)
                response_received_time = time.time()
                data = json.loads(response)
                if "audio" in data:
                    chunk = base64.b64decode(data["audio"])
                    truncated_audio_string = truncate_audio_string(data["audio"])
                    logging.info(f"Truncated audio string: {truncated_audio_string}")
                    chunk_received_time = time.time()
                    if not first_chunk_received:
                        first_chunk_received = True
                        first_chunk_time = chunk_received_time - start_time  # Calculate the time from the request to the first chunk
                    chunk_times.append(chunk_received_time - connection_open_time)
            except asyncio.TimeoutError:
                pass
        eos_message = {"text": ""}
        await websocket.send(json.dumps(eos_message))
        while True:
            try:
                response = await websocket.recv()
                response_received_time = time.time()
                data = json.loads(response)
                audio = data.get("audio")
                if audio is not None:
                    chunk = base64.b64decode(data["audio"])
                    truncated_audio_string = truncate_audio_string(data["audio"])
                    logging.info(f"Truncated audio string: {truncated_audio_string}")
                    chunk_received_time = time.time()
                    chunk_times.append(chunk_received_time - connection_open_time)
                else:
                    break
            except websockets.exceptions.ConnectionClosed:
                break
        connection_close_time = time.time()
        total_time_websocket_was_open = connection_close_time - connection_open_time
    return time_to_open_connection, first_chunk_time, chunk_times, total_time_websocket_was_open

# Main function
if args.API == "http":
    average_response_latency, median_response_latency, average_chunk_latency, median_chunk_latency = http_api_request()
    print(f"\nAverage Initial Response (Header) Time: {average_response_latency:.2f} ms")
    print(f"Median Initial Response (Header) Time: {median_response_latency:.2f} ms")
    print(f"Average First Playable Chunk (Body) Time: {average_chunk_latency:.2f} ms")
    print(f"Median First Playable Chunk (Body) Time: {median_chunk_latency:.2f} ms")
elif args.API == "websocket":
    time_to_open_connection, first_chunk_time, chunk_times, total_time_websocket_was_open = asyncio.run(websocket_api_request())
    print(f"\nTime to open connection: {time_to_open_connection:.4f} seconds")
    if first_chunk_time is not None:
        print(f"Time from request to first chunk: {first_chunk_time:.4f} seconds")  # Updated print statement
    for i, chunk_time in enumerate(chunk_times, start=1):
        print(f"Time to receive chunk {i} after request: {chunk_time:.4f} seconds")  # Updated print statement
    print(f"Total time WebSocket connection was open: {total_time_websocket_was_open:.4f} seconds")
