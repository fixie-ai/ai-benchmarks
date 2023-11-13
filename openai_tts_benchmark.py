import requests
import json
import time
import os
import argparse

DEFAULT_SAMPLES = 10
DEFAULT_TEXT = "I'm calling for Jim."
DEFAULT_MODEL_ID = "tts-1"
DEFAULT_CHUNK_SIZE = 7868  # This defines the size of the first playable chunk in bytes, which is 7868, roughly equivalent to half a second of audio
DEFAULT_VOICE_ID = "shimmer"

parser = argparse.ArgumentParser()
parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
parser.add_argument(
    "--model",
    "-m",
    default=DEFAULT_MODEL_ID,
)
parser.add_argument(
    "--num_samples",
    "-n",
    type=int,
    default=DEFAULT_SAMPLES,
)
parser.add_argument(
    "--chunk_size",
    "-c",
    type=int,
    default=DEFAULT_CHUNK_SIZE,
)
parser.add_argument(
    "--voice_id",
    "-v",
    default=DEFAULT_VOICE_ID,
)
args = parser.parse_args()


def openai_tts_request(session: requests.Session, model: str, text: str):
    url = "https://api.openai.com/v1/audio/speech"

    headers = {
        "Accept": "audio/mpeg",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }

    data = {
        "input": text,
        "model": model,
        "voice": args.voice_id,
    }
    return session.post(url, headers=headers, data=json.dumps(data), stream=True)


def main():
    response_latencies = []
    chunk_latencies = []
    for i in range(args.num_samples):
        print(f"\nAPI Call {i+1}:")
        start_time = time.perf_counter()
        session = requests.Session()
        response = openai_tts_request(session, args.model, args.text)
        if not response.ok:
            print(response.json())
            exit(1)

        response_received_time = time.perf_counter()
        response_latency = (response_received_time - start_time) * 1000
        response_latencies.append(response_latency)
        print(f"  Initial Response (Header) Time: {response_latency:.2f} ms")

        audio_data = b""
        can_play = False
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                audio_data += chunk
                if not can_play and len(audio_data) >= args.chunk_size:
                    chunk_received_time = time.perf_counter()
                    chunk_latency = (chunk_received_time - start_time) * 1000
                    chunk_latencies.append(chunk_latency)
                    can_play = True
                    print(f"  First Playable Chunk (Body) Time: {chunk_latency:.2f} ms")

        with open(f"audio_sample_{i+1}.mp3", "wb") as f:
            f.write(audio_data)

    average_response_latency = sum(response_latencies) / len(response_latencies)
    median_response_latency = sorted(response_latencies)[len(response_latencies) // 2]
    print(
        f"\nAverage Initial Response (Header) Time: {average_response_latency:.2f} ms"
    )
    print(f"Median Initial Response (Header) Time: {median_response_latency:.2f} ms")

    average_chunk_latency = sum(chunk_latencies) / len(chunk_latencies)
    median_chunk_latency = sorted(chunk_latencies)[len(chunk_latencies) // 2]
    print(f"\nAverage First Playable Chunk (Body) Time: {average_chunk_latency:.2f} ms")
    print(f"Median First Playable Chunk (Body) Time: {median_chunk_latency:.2f} ms")


if __name__ == "__main__":
    main()
