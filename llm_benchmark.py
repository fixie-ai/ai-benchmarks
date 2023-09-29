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
    async for line in response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            if content != "[DONE]":
                chunk = json.loads(content)
                text_chunk = chunk["choices"][0]["delta"].get("content", "")
                yield text_chunk


async def make_api_call(session, index):
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
    start_time = time.time()
    response = await session.post(url, headers=headers, data=data)
    latency = time.time() - start_time
    return ApiResult(index, latency, response, make_chunk_gen(response))


async def async_main():
    print(f"Racing {args.num_requests} API calls to {args.model}...")
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(make_api_call(session, i))
            for i in range(args.num_requests)
        ]
        # Wait for just the first task to complete
        done = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        chosen = done[0].pop().result()
        print(f"Chosen API Call: {chosen.index} ({chosen.latency:.2f}s)")

        # Stream out the tokens
        async for chunk in chosen.chunk_gen:
            print(chunk, end="", flush=True)
        print("\n")

        # Wait for the rest of the tasks to complete and clean up
        done = await asyncio.wait(tasks)
        results = [task.result() for task in done[0]]
        for result in results:
            await result.response.release()

    # Print out each result, sorted by index
    results.sort(key=lambda x: x.index)
    task1 = results[0]
    for r in results:
        if r.response.ok:
            print(
                f"API Call {r.index} Initial Response Latency: {r.latency:.2f} seconds"
            )
        else:
            print(
                f"API Call {r.index} Result: {r.response.status} in {r.latency:.2f} seconds"
            )

    # Print a timing summary
    latency_saved = task1.latency - chosen.latency
    print(f"\nLatency saved: {latency_saved:.2f} seconds")
    print(f"Optimized response time: {chosen.latency:.2f} seconds")
    med_index1 = (len(results) - 1) // 2
    med_index2 = len(results) // 2
    median_latency = (results[med_index1].latency + results[med_index2].latency) / 2
    print(f"Median response time: {median_latency:.2f} seconds")


asyncio.run(async_main())
