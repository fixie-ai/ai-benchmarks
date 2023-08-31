import requests
import json
import time  

# URL of the text-to-speech API
url = 'https://api.elevenlabs.io/v1/text-to-speech/flq6f7yk4E4fJM5XTYuZ/stream?optimize_streaming_latency=0'

# Headers for the API request
headers = {
    'accept': 'audio/mpeg',
    'xi-api-key': 'api_key',
    'Content-Type': 'application/json',
}

# Data for the API request
data = {
  "text": """I'm calling for Jim.""",
  "model_id": "eleven_monolingual_v1",
  #"model_id": "eleven_english_v1",
  #"model_id": "eleven_multilingual_v2",
  #"model_id": "eleven_english_v2",
  "voice_settings": {
    "stability": 0.5,
    "similarity_boost": 1
  }
}

latencies = []  # Initialize an empty list to store latencies

for i in range(10):  # Make 10 API calls
    start_time = time.perf_counter()  # Record the current time before sending the API request
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    
    # Calculate latency
    latency = (time.perf_counter() - start_time) * 1000
    print(f"eleven_english_v1 API Call {i+1}: Response Time: {latency:.2f} ms")  # Print the latency
    latencies.append(latency)  # Append the latency to the list

# Calculate and print the average latency
average_latency = sum(latencies) / len(latencies)
print(f"\nLatencies: {latencies}")
print(f"\nAverage Latency: {average_latency:.2f} ms")
