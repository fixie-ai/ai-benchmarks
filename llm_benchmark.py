import argparse
import asyncio
import dataclasses
import json
import os
import time
import urllib
from typing import Generator

import aiohttp

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
    def __init__(self, index, start_time, response, chunk_gen):
        self.index = index
        self.start_time = start_time
        self.latency = time.time() - start_time
        self.response = response
        self.chunk_gen = chunk_gen

    index: int
    start_time: int
    latency: float  # HTTP response time
    response: aiohttp.ClientResponse
    chunk_gen: Generator[str, None, None]


async def post(
    session: aiohttp.ClientSession,
    index: int,
    url: str,
    headers: dict,
    data: dict,
    make_chunk_gen: callable(aiohttp.ClientResponse) = None,
):
    start_time = time.time()
    response = await session.post(url, headers=headers, data=json.dumps(data))
    chunk_gen = make_chunk_gen(response) if make_chunk_gen else None
    return ApiResult(index, start_time, response, chunk_gen)


async def make_openai_chunk_gen(response) -> Generator[str, None, None]:
    async for line in response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            if content == "[DONE]":
                break
            chunk = json.loads(content)
            if chunk["choices"]:
                yield chunk["choices"][0]["delta"].get("content", "")


def make_openai_url_and_headers(path: str):
    url = args.base_url or "https://api.openai.com/v1"
    use_azure = urllib.parse.urlparse(url).hostname.endswith(".azure.com")
    headers = {
        "Content-Type": "application/json",
    }
    if use_azure:
        headers["Api-Key"] = os.environ["AZURE_OPENAI_API_KEY"]
        url += f"/openai/deployments/{args.model.replace('.', '')}{path}?api-version=2023-07-01-preview"
    else:
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
        url += path
    return url, headers


async def openai_chat(session: aiohttp.ClientSession, index: int) -> ApiResult:
    url, headers = make_openai_url_and_headers("/chat/completions")
    data = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        "stream": True,
        "max_tokens": args.max_tokens,
    }
    return await post(session, index, url, headers, data, make_openai_chunk_gen)


async def openai_embed(session: aiohttp.ClientSession, index: int) -> ApiResult:
    url, headers = make_openai_url_and_headers("/embeddings")
    data = {
        "model": args.model,
        "input": args.prompt,
    }
    return await post(session, index, url, headers, data)


async def make_anthropic_chunk_gen(response) -> Generator[str, None, None]:
    async for line in response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            chunk = json.loads(content)
            yield chunk.get("completion", "")


async def anthropic_chat(session: aiohttp.ClientSession, index: int) -> ApiResult:
    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "content-type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    }
    data = {
        "model": args.model,
        "prompt": f"\n\nHuman: {args.prompt}\n\nAssistant:",
        "max_tokens_to_sample": args.max_tokens,
        "stream": True,
    }
    return await post(session, index, url, headers, data, make_anthropic_chunk_gen)


async def cohere_embed(session: aiohttp.ClientSession, index: int) -> ApiResult:
    url = "https://api.cohere.ai/v1/embed"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ['COHERE_API_KEY']}",
    }
    data = {"model": args.model, "texts": [args.prompt], "input_type": "search_query"}
    return await post(session, index, url, headers, data)


async def make_fixie_chunk_gen(response) -> Generator[str, None, None]:
    text = ""
    async for line in response.content:
        line = line.decode("utf-8").strip()
        obj = json.loads(line)
        curr_turn = obj["turns"][-1]
        if (
            curr_turn["role"] == "assistant"
            and curr_turn["messages"]
            and "content" in curr_turn["messages"][-1]
        ):
            if curr_turn["state"] == "done":
                break
            new_text = curr_turn["messages"][-1]["content"]
            # Sometimes we get a spurious " " message
            if new_text == " ":
                continue
            if new_text.startswith(text):
                delta = new_text[len(text) :]
                text = new_text
                yield delta
            else:
                print(f"Warning: got unexpected text: '{new_text}' vs '{text}'")


async def fixie_chat(session: aiohttp.ClientSession, index: int) -> ApiResult:
    url = f"https://api.fixie.ai/api/v1/agents/{args.model}/conversations"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ['FIXIE_API_KEY']}",
    }
    data = {"message": args.prompt, "runtimeParameters": {}}
    return await post(session, index, url, headers, data, make_fixie_chunk_gen)


async def make_api_call(
    session: aiohttp.ClientSession, index: int, model: str
) -> ApiResult:
    if model.startswith("gpt-") or model.startswith("ft:gpt-"):
        return await openai_chat(session, index)
    elif model.startswith("claude-"):
        return await anthropic_chat(session, index)
    elif model.startswith("text-embedding-ada-"):
        return await openai_embed(session, index)
    elif model.startswith("embed-"):
        return await cohere_embed(session, index)
    elif "/" in model:
        return await fixie_chat(session, index)
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
            result = task.result()
            results.append(result)
            if result.response.ok:
                chosen = task.result()
            else:
                status = result.response.status
                text = await result.response.text()
                text = text[:80] + "..." if len(text) > 80 else text
                print(f"API Call {result.index} failed, status={status} text={text}")
            tasks.remove(task)

        # Bail out if no tasks succeed
        if not chosen:
            print("No successful API calls")
            exit(1)
        print(f"Chosen API Call: {chosen.index} ({chosen.latency:.2f}s)")

        # Stream out the tokens, if we're doing completion
        first_token_time = None
        num_tokens = 0
        if chosen.chunk_gen:
            async for chunk in chosen.chunk_gen:
                num_tokens += 1
                if not first_token_time:
                    first_token_time = time.time()
                print(chunk, end="", flush=True)
            end_time = time.time()
            print("\n")

        # Wait for the rest of the tasks to complete and clean up
        if tasks:
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
    if num_tokens > 0:
        print(
            f"Time to first token: {first_token_time - chosen.start_time:.2f} seconds"
        )
        print(
            f"Tokens: {num_tokens} ({(num_tokens - 1) / (end_time - first_token_time):.0f} tokens/sec)"
        )
        print(f"Total time: {end_time - chosen.start_time:.2f} seconds")


asyncio.run(async_main())
