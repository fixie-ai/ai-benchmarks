import aiohttp
import argparse
import asyncio
import dataclasses
import json
import os
import time
import urllib
from typing import Generator

DEFAULT_PROMPT = "Say hello."
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_MAX_TOKENS = 50
DEFAULT_NUM_REQUESTS = 5

parser = argparse.ArgumentParser()
parser.add_argument(
    "prompt",
    type=str,
    nargs="?",
    default=DEFAULT_PROMPT,
    help="Prompt to send to the API",
)
parser.add_argument(
    "--model", "-m", type=str, default=DEFAULT_MODEL, help="Model to benchmark"
)
parser.add_argument(
    "--max-tokens",
    "-t",
    type=int,
    default=DEFAULT_MAX_TOKENS,
    help="Max tokens for the response",
)
parser.add_argument(
    "--base-url",
    "-b",
    type=str,
    default=None,
    help="Base URL for the LLM API endpoint",
)
parser.add_argument(
    "--num-requests",
    "-n",
    type=int,
    default=DEFAULT_NUM_REQUESTS,
    help="Number of requests to make",
)
args = parser.parse_args()


@dataclasses.dataclass
class ApiResult:
    index: int
    latency: float
    response: aiohttp.ClientResponse
    chunk_gen: Generator[str, None, None]


async def make_openai_chunk_gen(response) -> Generator[str, None, None]:
    async for line in response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            if content != "[DONE]":
                chunk = json.loads(content)
                if chunk["choices"]:
                    yield chunk["choices"][0]["delta"].get("content", "")


async def make_openai_api_call(session: aiohttp.ClientSession, index: int) -> ApiResult:
    url = args.base_url or "https://api.openai.com/v1"
    use_azure = urllib.parse.urlparse(url).hostname.endswith(".azure.com")
    headers = {
        "Content-Type": "application/json",
    }
    if use_azure:
        headers["Api-Key"] = os.environ["AZURE_OPENAI_API_KEY"]
        url += f"/openai/deployments/{args.model.replace('.', '')}"
    else:
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    url += "/chat/completions"
    if use_azure:
        url += "?api-version=2023-07-01-preview"
    data = json.dumps(
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt},
            ],
            "stream": True,
            "max_tokens": args.max_tokens,
        }
    )
    start_time = time.time()
    response = await session.post(url, headers=headers, data=data)
    latency = time.time() - start_time
    return ApiResult(index, latency, response, make_openai_chunk_gen(response))


async def make_anthropic_chunk_gen(response) -> Generator[str, None, None]:
    async for line in response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            chunk = json.loads(content)
            yield chunk.get("completion", "")


async def make_anthropic_api_call(
    session: aiohttp.ClientSession, index: int
) -> ApiResult:
    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "content-type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    }
    data = json.dumps(
        {
            "model": args.model,
            "prompt": f"\n\nHuman: {args.prompt}\n\nAssistant:",
            "max_tokens_to_sample": 256,
            "stream": True,
        }
    )
    start_time = time.time()
    response = await session.post(url, headers=headers, data=data)
    latency = time.time() - start_time
    return ApiResult(index, latency, response, make_anthropic_chunk_gen(response))


async def make_api_call(
    session: aiohttp.ClientSession, index: int, model: str
) -> ApiResult:
    if model.startswith("gpt-") or model.startswith("ft:gpt-"):
        return await make_openai_api_call(session, index)
    elif model.startswith("claude-"):
        return await make_anthropic_api_call(session, index)
    else:
        raise ValueError(f"Unknown model: {model}")


async def async_main():
    print(f"Racing {args.num_requests} API calls to {args.model}...")
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(make_api_call(session, i, args.model))
            for i in range(args.num_requests)
        ]
        results = []

        # Wait for the first task to complete successfully
        chosen = None
        while tasks and not chosen:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            task = done.pop()
            results.append(task.result())
            if task.result().response.ok:
                chosen = task.result()
            else:
                tasks.remove(task)
        print(f"Chosen API Call: {chosen.index} ({chosen.latency:.2f}s)")

        # Stream out the tokens
        async for chunk in chosen.chunk_gen:
            print(chunk, end="", flush=True)
        print("\n")

        # Wait for the rest of the tasks to complete and clean up
        done = await asyncio.wait(tasks)
        results += [task.result() for task in done[0]]
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
    results.sort(key=lambda x: x.latency)
    med_index1 = (len(results) - 1) // 2
    med_index2 = len(results) // 2
    median_latency = (results[med_index1].latency + results[med_index2].latency) / 2
    print(f"Median response time: {median_latency:.2f} seconds")


asyncio.run(async_main())
