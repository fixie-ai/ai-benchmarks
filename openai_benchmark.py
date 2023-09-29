import asyncio
import aiohttp
import json
import time
import argparse
import os
import dataclasses

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", "-m", type=str, default="gpt-3.5-turbo", help="Model to benchmark"
)
parser.add_argument(
    "--max-tokens", "-t", type=int, default=50, help="Max tokens for the response"
)
parser.add_argument(
    "--num-requests", "-n", type=int, default=5, help="Number of requests to make"
)
args = parser.parse_args()


@dataclasses.dataclass
class ApiResult:
    index: int
    latency: float
    response: aiohttp.ClientResponse
    chunk_gen: any


async def make_chunk_gen(response):
    async for line in await response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            if content != "[DONE]":
                chunk = json.loads(content)
                text_chunk = chunk["choices"][0]["delta"].get("content", "")
                yield text_chunk


async def make_api_call(session, url, headers, data, index):
    start_time = time.time()
    async with session.post(url, headers=headers, data=data) as resp:
        latency = time.time() - start_time
        return ApiResult(index, latency, resp, make_chunk_gen(resp))


async def get_fastest_response():
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    data = json.dumps(
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello."},
            ],
            "stream": True,
            "max_tokens": args.max_tokens,
        }
    )

    print(f"Racing {args.num_requests} API calls to {args.model}...")
    async with aiohttp.ClientSession() as session:
        tasks = [
            make_api_call(session, url, headers, data, i)
            for i in range(args.num_requests)
        ]
        results = []
        done = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        chosen_result = done[0].pop().result()
        results.append(chosen_result)

        print(f"\nChosen API Call: {chosen_result.index}\n")

        async for chunk in chosen_result.chunk_gen:
            print(chunk, end="", flush=True)
        print("\n")  # Print a newline at the end
        return results


results = asyncio.run(get_fastest_response())
results.sort(key=lambda x: x.index)
task1 = results[0]
for result in results:
    print(
        f"Initial Response Latency for API Call {result.index}: {result.latency:.2f} seconds"
    )

results.sort(key=lambda x: x.latency)
chosen_task = results[0]
latency_saved = task1.latency - chosen_task.latency
print(f"\nLatency saved: {latency_saved:.2f} seconds")
print(f"Optimized response time: {chosen_task.latency:.2f} seconds")


print(f"Median response time: {results[len(results)//2].latency:.2f} seconds")
