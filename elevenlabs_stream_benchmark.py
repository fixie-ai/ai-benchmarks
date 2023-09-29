import requests
import json
import time
import os
import argparse

DEFAULT_TEXT = "I'm calling for Jim."
DEFAULT_MODEL_ID = "eleven_monolingual_v1"
DEFAULT_SAMPLES = 10
DEFAULT_VOICE = "flq6f7yk4E4fJM5XTYuZ"

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
    "--optimize-streaming-latency",
    "-o",
    type=int,
    default=4,
)
parser.add_argument(
    "--voice",
    "-v",
    default=DEFAULT_VOICE,
)
args = parser.parse_args()

# URL of the text-to-speech API
url = f"https://api.elevenlabs.io/v1/text-to-speech/{args.voice}/stream?optimize_streaming_latency={args.optimize_streaming_latency}"

# Headers for the API request
headers = {
    "accept": "audio/mpeg",
    "xi-api-key": os.environ["ELEVEN_API_KEY"],
    "Content-Type": "application/json",
}

# Data for the API request
data = {
    "text": args.text,
    "model_id": args.model,
    "voice_settings": {"stability": 0.5, "similarity_boost": 1},
}

latencies = []

for i in range(args.num_samples):
    start_time = (
        time.perf_counter()
    )  # Record the current time before sending the API request
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    if not response.ok:
        print("Error: " + response.text)
        exit(1)

    # Calculate latency
    latency = (time.perf_counter() - start_time) * 1000
    print(
        f"{args.model} API Call {i+1}: Response Time: {latency:.2f} ms"
    )  # Print the latency
    latencies.append(latency)  # Append the latency to the list

# Calculate and print the average latency
average_latency = sum(latencies) / len(latencies)
median_latency = sorted(latencies)[len(latencies) // 2]
print(f"Average Latency: {average_latency:.2f} ms")
print(f"Median Latency: {median_latency:.2f} ms")
