import argparse
import asyncio
import base64
import json
import logging
import os
import time
import typing
import uuid

import grpc
import websockets
from pyht import AsyncClient
from pyht.client import TTSOptions
from pyht.protos import api_pb2
from pyht.protos import api_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


class LatencyData:

    def __init__(self):
        # We start the latency timer right BEFORE we send the request.
        self.start_time = None
        # We end the latency timer AFTER we receive the final response.
        # this is the point at which we know we are done. But we may have
        # received the final audio chunk before that.
        self.end_time = None
        # The total latency is the sum of all the chunk latencies.
        self.total_latency = 0
        # The chunk times are the times at which each chunk was received.
        # These should be in order, and the last one should be the end time.
        self.chunk_times = []

    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        self.end_time = time.perf_counter()
        if len(self.chunk_times) > 0:
            # we grab the last recorded chunk time and subtract it from the start time.
            self.total_latency = self.chunk_times[-1] - self.start_time
        else:
            self.total_latency = self.end_time - self.get_start_time()
            logging.warning("No chunk times found, using end time instead")

    def add_chunk_latency(self):
        chunk_time = time.perf_counter()
        logging.debug(f"Chunk latency added: {chunk_time}")
        self.chunk_times.append(chunk_time)

    def get_start_time(self) -> float:
        if not self.start_time:
            raise RuntimeError("Start time not calculated. start() not called")
        return self.start_time

    def time_to_first_chunk(self):
        if len(self.get_chunk_times()) == 0:
            return 0
        return self.get_chunk_times()[0] - self.get_start_time()

    def get_average_chunk_latency(self):
        if not len(self.get_chunk_times()):
            return 0
        # we first have to iterate over the chunk times
        # and subtract each chunk time from the next.
        previous_chunk_time = self.get_start_time()
        chunk_latencies = []
        for chunk_time in self.get_chunk_times():
            latency = chunk_time - previous_chunk_time
            chunk_latencies.append(latency)
            previous_chunk_time = chunk_time
        # then we average the chunk latencies.
        return sum(chunk_latencies) / len(chunk_latencies)

    def get_total_latency(self):
        if not self.total_latency:
            raise RuntimeError("Total latency not calculated. end() not called")
        return self.total_latency

    def get_chunk_times(self):
        return self.chunk_times


# This is a modified version of the elevenlabs text chunker.
def elevenlabs_text_chunker(chunks: typing.Iterator[str]) -> typing.Iterator[str]:
    """Used during input streaming to chunk text blocks and set last char to space"""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""
    for text in chunks:
        if buffer.endswith(splitters):
            yield buffer if buffer.endswith(" ") else f"{buffer} "
            buffer = text
        elif text.startswith(splitters):
            output = f"{buffer}{text[0]}"
            yield output if output.endswith(" ") else f"{output} "
            buffer = text[1:]
        else:
            buffer += text
    if buffer != "":
        yield f"{buffer} "


async def elevenlabs_ws_tts(latency_data: LatencyData, config: dict):
    """Eleven Labs WebSocket raw implementation"""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{config['voice_id']}/stream-input?model_id={config['model']}&optimize_streaming_latency={config.get('optimize_streaming_latency', 0)}&output_format={config.get('output_format', 'mp3_44100_128')}"
    async with websockets.connect(uri) as websocket:
        bos_message = {
            # Start with a space " " as per documentation:
            # https://elevenlabs.io/docs/api-reference/websockets#streaming-input-text
            "text": " ",
            "voice_settings": {
                "stability": config["stability"],
                "similarity_boost": config["similarity_boost"],
            },
            "generation_config": {"chunk_length_schedule": [50]},
            "xi_api_key": config["xi_api_key"],
            "try_trigger_generation": True,
        }
        await websocket.send(json.dumps(bos_message))

        audio_chunks: list[bytes] = []
        text_chunks = elevenlabs_text_chunker(config["text"])
        # we start the latency timer right BEFORE we send the first text chunk.
        latency_data.start()
        try:
            # send text chunks, in order, one at time. This is also how the
            # elevenlabs api implements it on their client aparently.
            for text_chunk in text_chunks:
                data = dict(text=text_chunk, flush=True, try_trigger_generation=True)
                await websocket.send(json.dumps(data))
                logging.debug(f"Sent chunk: {text_chunk}")

            # EOS should always end with a empty space string ""
            # as per documentation:
            # https://elevenlabs.io/docs/api-reference/websockets#close-connection
            await websocket.send(json.dumps(dict(text="", try_trigger_generation=True)))
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=30)
                latency_data.add_chunk_latency()
                data = json.loads(response)
                logging.debug(f"Received response: {data}")
                if "audio" in data:
                    audio = data["audio"]
                    if isinstance(audio, str):
                        audio_bytes = base64.b64decode(audio)
                        audio_chunks.append(audio_bytes)
                    elif audio is not None:
                        logging.warning(
                            f"Received non-string audio data: {type(audio)}"
                        )
                if data.get("isFinal"):
                    logging.debug("We are done")
                    break
        except TimeoutError as e:
            logging.error(f"TimeoutError: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            logging.debug(f"Connection closed: {e}")
        finally:
            latency_data.end()

    return b"".join(audio_chunks)


async def cartesia_tts(latency_data: LatencyData, config: dict):
    """Cartesia TTS WebSocket raw implementation"""
    uri = (
        f"wss://api.cartesia.ai/tts/websocket"
        f"?api_key={config['api_key']}"
        f"&cartesia_version={config['cartesia_version']}"
    )
    async with websockets.connect(uri) as websocket:
        request = {
            "context_id": f"tts-benchmark-suite-{uuid.uuid4()}",
            "model_id": config["model_id"],
            "transcript": config["text"],
            "duration": config.get("duration", 180),
            "voice": {
                "mode": "id",
                "id": config["voice_id"],
                "__experimental_controls": config.get("experimental_controls", {}),
            },
            "output_format": config["output_format"],
            "language": config.get("language", "en"),
            "add_timestamps": config.get("add_timestamps", False),
            "continue": False,  # Sending to cartesia to indicate we are done.
        }
        # we start the latency timer right BEFORE we send the request.
        latency_data.start()
        await websocket.send(json.dumps(request))
        audio_chunks: list[bytes] = []
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=30)
                latency_data.add_chunk_latency()
                data = json.loads(response)
                if "audio" in data:
                    audio = data["audio"]
                    audio_bytes = base64.b64decode(audio)
                    audio_chunks.append(audio_bytes)
                if data.get("done", False):
                    break
        except asyncio.TimeoutError:
            logging.error("Timeout waiting for Cartesia response")
        except websockets.exceptions.ConnectionClosed as e:
            logging.debug(f"Cartesia connection closed: {e}")
        finally:
            latency_data.end()

    return b"".join(audio_chunks)


async def playht_grpc_tts(latency_data: LatencyData, config: dict):
    """PlayHT TTS gRPC streaming implementation using raw gRPC client"""
    # It seems that the playht grpc api is not working, is this the right
    # address?
    channel = grpc.aio.insecure_channel("prod.turbo.play.ht:443")
    stub = api_pb2_grpc.TtsStub(channel)

    if config["output_format"] == "mp3":
        audio_format = api_pb2.FORMAT_MP3
    elif config["output_format"] == "wav":
        audio_format = api_pb2.FORMAT_WAV
    else:
        raise ValueError(f"Invalid format: {config['output_format']}")

    # https://github.com/playht/pyht/blob/428e7af21962205883f5e14ba2f482dcf6c11b6d/protos/api.proto#L10
    request = api_pb2.TtsRequest(
        # https://github.com/playht/pyht/blob/428e7af21962205883f5e14ba2f482dcf6c11b6d/protos/api.proto#L24
        params=api_pb2.TtsParams(
            text=config["text"],
            voice=config["voice_id"],
            format=audio_format,
            sample_rate=config["sample_rate"],
            quality=config.get("quality", api_pb2.QUALITY_DRAFT),
            speed=config.get("speed", 1.0),
        )
    )

    metadata = [
        ("authorization", f"Bearer {config['user_id']}:{config['api_key']}"),
        ("x-api-key", config["api_key"]),
    ]
    audio_chunks: list[bytes] = []
    latency_data.start()
    try:
        # https://github.com/playht/pyht/blob/428e7af21962205883f5e14ba2f482dcf6c11b6d/protos/api.proto#L6C9-L6C12
        async for response in stub.Tts(request, metadata=metadata):
            latency_data.add_chunk_latency()
            logging.info(f"Received response chunk: {response}")
            if response.HasField("audio"):
                audio_chunks.append(response.audio)
    except grpc.RpcError as e:
        error_code = e.code()
        logging.error(f"PlayHT gRPC streaming error: {error_code} {e}")
        if error_code not in {
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.UNAVAILABLE,
        }:
            raise e
    except Exception as e:
        logging.error(f"Error during PlayHT gRPC streaming: {e}")
        raise e
    finally:
        latency_data.end()
        await channel.close()
    return b"".join(audio_chunks)


async def playht_python_sdk_tts(latency_data: LatencyData, config: dict):
    """PlayHT TTS gRPC streaming implementation using the playht python sdk"""
    client = AsyncClient(
        user_id=config["user_id"],
        api_key=config["api_key"],
        # Uncomment the following line if you need to specify a custom gRPC address
        # advanced=AsyncClient.AdvancedOptions(grpc_addr="prod.turbo.play.ht:443")
    )

    if config["output_format"] == "mp3":
        audio_format = api_pb2.FORMAT_MP3
    elif config["output_format"] == "wav":
        audio_format = api_pb2.FORMAT_WAV
    else:
        raise ValueError(f"Invalid format: {config['output_format']}")

    options = TTSOptions(
        voice=config["voice_id"],
        format=audio_format,
        quality=config.get("quality", api_pb2.QUALITY_DRAFT),
        speed=config.get("speed", 1.0),
        sample_rate=config.get("sample_rate", 24000),
    )

    audio_chunks = []
    latency_data.start()
    try:
        async for chunk in client.tts(config["text"], options):
            latency_data.add_chunk_latency()
            if isinstance(chunk, bytes):
                audio_chunks.append(chunk)
            else:
                logging.debug(f"Received non-audio chunk: {chunk}")
    except Exception as e:
        logging.error(f"Error during PlayHT async streaming: {e}")
        raise e
    finally:
        latency_data.end()
        await client.close()

    return b"".join(audio_chunks)


def parse_args():
    parser = argparse.ArgumentParser(description="TTS Benchmark Suite")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--eleven-api-key", help="Eleven Labs API Key")
    parser.add_argument("--cartesia-api-key", help="Cartesia API Key")
    parser.add_argument("--playht-api-key", help="PlayHT API Key")
    parser.add_argument("--playht-user-id", help="PlayHT User ID")
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.eleven_api_key:
        os.environ["ELEVEN_API_KEY"] = args.eleven_api_key
    if args.cartesia_api_key:
        os.environ["CARTESIA_API_KEY"] = args.cartesia_api_key
    if args.playht_api_key:
        os.environ["PLAYHT_API_KEY"] = args.playht_api_key
    if args.playht_user_id:
        os.environ["PLAYHT_USER_ID"] = args.playht_user_id

    services = {
        "Eleven Labs - websocket": {
            "function": elevenlabs_ws_tts,
            "config": {
                "text": args.text,
                "voice_id": "EXAVITQu4vr4xnSDxMaL",
                "model": "eleven_turbo_v2",
                "stability": 0.5,
                "similarity_boost": False,
                "xi_api_key": os.environ["ELEVEN_API_KEY"],
                "output_format": "pcm_44100",
                "optimize_streaming_latency": 4,  # 1 is the lowest, 4 is the highest
            },
        },
        "Cartesia - websocket": {
            "function": cartesia_tts,
            "config": {
                "api_key": os.environ["CARTESIA_API_KEY"],
                "cartesia_version": "2024-08-11",  # Update this as needed
                "voice_id": "a0e99841-438c-4a64-b679-ae501e7d6091",
                "text": args.text,
                "model_id": "sonic-english",
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 8000,
                },
                "language": "en",
                "add_timestamps": False,
            },
        },
        "PlayHT - GRPC Python SDK": {
            "function": playht_python_sdk_tts,
            "config": {
                "text": args.text,
                "api_key": os.environ["PLAYHT_API_KEY"],
                "user_id": os.environ["PLAYHT_USER_ID"],
                "voice_id": "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
                "voice_engine": "PlayHT2.0-turbo",
                "quality": "faster",
                "output_format": "wav",
                "sample_rate": 44100,
                "speed": 1,
            },
        },
    }

    for service, details in services.items():
        logging.info(f"Benchmarking {service}...")

        latency_data = LatencyData()
        # TODO: We might also want to save the audio to a file here, for a
        # subjective quality comparison.
        wav_audio = await details["function"](latency_data, details["config"])
        total_time = latency_data.get_total_latency()
        logging.info(
            f"{service}: (TTFU) time to first utterance: {latency_data.time_to_first_chunk() * 1000:.2f}ms"
        )
        logging.info(
            f"{service}: Average chunk latency: {latency_data.get_average_chunk_latency() * 1000:.2f}ms"
        )
        logging.info(
            f"{service}: Total chunks received: {len(latency_data.get_chunk_times())}"
        )
        logging.info(f"{service}: Total processing time: {total_time * 1000:.2f}ms")
        logging.info("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
