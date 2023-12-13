#!/usr/bin/env python
import argparse
import asyncio
import dataclasses
import json
import os
import time
import urllib
from typing import Any, Dict, Generator, List, Optional

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
    "--temperature",
    "-t",
    type=float,
    default=0.0,
    help="Temperature for the response",
)
parser.add_argument(
    "--max-tokens",
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
    "--api-key",
    "-k",
    type=str,
    default=None,
    help="API key for the LLM API endpoint",
)
parser.add_argument(
    "--no-warmup",
    action="store_false",
    dest="warmup",
    help="Don't do a warmup call to the API",
)
parser.add_argument(
    "--num-requests",
    "-n",
    type=int,
    default=DEFAULT_NUM_REQUESTS,
    help="Number of requests to make",
)
parser.add_argument(
    "--print",
    "-p",
    action="store_true",
    dest="print",
    help="Print the response",
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    dest="verbose",
    help="Print verbose output",
)
group.add_argument(
    "--minimal",
    action="store_true",
    dest="minimal",
    help="Print minimal output",
)
args = parser.parse_args()


@dataclasses.dataclass
class ApiContext:
    session: aiohttp.ClientSession
    index: int
    model: str
    prompt: str


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
    context: ApiContext,
    url: str,
    headers: dict,
    data: dict,
    make_chunk_gen: callable(aiohttp.ClientResponse) = None,
):
    start_time = time.time()
    response = await context.session.post(url, headers=headers, data=json.dumps(data))
    chunk_gen = make_chunk_gen(response) if make_chunk_gen else None
    return ApiResult(context.index, start_time, response, chunk_gen)


def get_api_key(env_var: str) -> str:
    if args.api_key:
        return args.api_key
    if env_var in os.environ:
        return os.environ[env_var]
    raise ValueError(f"Missing API key: {env_var}")


def make_headers(auth_token: Optional[str] = None, x_api_key: Optional[str] = None):
    headers = {
        "content-type": "application/json",
    }
    if auth_token:
        headers["authorization"] = f"Bearer {auth_token}"
    if x_api_key:
        headers["x-api-key"] = x_api_key
    return headers


def make_openai_url_and_headers(model: str, path: str):
    url = args.base_url or "https://api.openai.com/v1"
    use_azure = urllib.parse.urlparse(url).hostname.endswith(".azure.com")
    headers = {
        "Content-Type": "application/json",
    }
    if use_azure:
        api_key = get_api_key("AZURE_OPENAI_API_KEY")
        headers["Api-Key"] = api_key
        url += f"/openai/deployments/{model.replace('.', '')}{path}?api-version=2023-07-01-preview"
    else:
        api_key = get_api_key("OPENAI_API_KEY")
        headers["Authorization"] = f"Bearer {api_key}"
        url += path
    return url, headers


def make_messages(prompt: str):
    return [{"role": "user", "content": prompt}]


def make_openai_chat_body(
    prompt: Optional[str] = None, messages: Optional[List[Dict]] = None
):
    body = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if prompt:
        body["prompt"] = prompt
    elif messages:
        body["messages"] = messages
    return body


async def make_sse_chunk_gen(response) -> Generator[Dict, None, None]:
    async for line in response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            if content == "[DONE]":
                break
            yield json.loads(content)


async def openai_chunk_gen(response) -> Generator[str, None, None]:
    async for chunk in make_sse_chunk_gen(response):
        if chunk["choices"]:
            delta_content = chunk["choices"][0]["delta"].get("content")
            if delta_content:
                yield delta_content


async def openai_chat(context: ApiContext) -> ApiResult:
    url, headers = make_openai_url_and_headers(context.model, "/chat/completions")
    data = make_openai_chat_body(messages=make_messages(context.prompt))
    return await post(context, url, headers, data, openai_chunk_gen)


async def openai_embed(context: ApiContext) -> ApiResult:
    url, headers = make_openai_url_and_headers(context.model, "/embeddings")
    data = {
        "model": context.model,
        "input": context.prompt,
    }
    return await post(context, url, headers, data)


async def anthropic_chat(context: ApiContext) -> ApiResult:
    async def chunk_gen(response) -> Generator[str, None, None]:
        async for chunk in make_sse_chunk_gen(response):
            yield chunk.get("completion", "")

    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "content-type": "application/json",
        "x-api-key": get_api_key("ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
    }
    data = {
        "model": context.model,
        "prompt": f"\n\nHuman: {context.prompt}\n\nAssistant:",
        "max_tokens_to_sample": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    return await post(context, url, headers, data, chunk_gen)


async def cloudflare_chat(context: ApiContext) -> ApiResult:
    async def chunk_gen(response) -> Generator[str, None, None]:
        async for chunk in make_sse_chunk_gen(response):
            yield chunk["response"]

    account_id = os.environ["CF_ACCOUNT_ID"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{args.model}"
    headers = make_headers(auth_token=get_api_key("CF_API_KEY"))
    data = make_openai_chat_body(messages=make_messages(context.prompt))
    return await post(context, url, headers, data, chunk_gen)


async def make_json_chunk_gen(response) -> Generator[Any, None, None]:
    """Hacky parser for the JSON streaming format used by Google Vertex AI."""
    buf = ""
    async for line in response.content:
        # Eat the first array bracket, we'll do the same for the last one below.
        line = line.decode("utf-8").strip()
        if not buf and line.startswith("["):
            line = line[1:]
        # Split on comma-only lines, otherwise concatenate.
        if line == ",":
            yield json.loads(buf)
            buf = ""
        else:
            buf += line
    yield json.loads(buf[:-1])


async def google_chat(context: ApiContext) -> ApiResult:
    async def chunk_gen(response) -> Generator[str, None, None]:
        async for chunk in make_json_chunk_gen(response):
            yield chunk["outputs"][0]["structVal"]["candidates"]["listVal"][0][
                "structVal"
            ]["content"]["stringVal"][0]

    region = "us-west1"
    project_id = os.environ["GCP_PROJECT"]
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{args.model}:serverStreamingPredict"
    headers = make_headers(auth_token=get_api_key("GOOGLE_VERTEXAI_API_KEY"))
    data = {
        "inputs": [
            {
                "struct_val": {
                    "messages": {
                        "list_val": [
                            {
                                "struct_val": {
                                    "content": {
                                        "string_val": context.prompt,
                                    },
                                    "author": {"string_val": "user"},
                                }
                            }
                        ]
                    }
                }
            }
        ],
        "parameters": {
            "struct_val": {
                "temperature": {"float_val": args.temperature},
                "maxOutputTokens": {"int_val": args.max_tokens},
            }
        },
    }
    return await post(context, url, headers, data, chunk_gen)


async def gemini_chat(context: ApiContext) -> ApiResult:
    async def chunk_gen(response) -> Generator[str, None, None]:
        async for chunk in make_json_chunk_gen(response):
            content = chunk["candidates"][0]["content"]
            if "parts" in content:
                part = content["parts"][0]
                if "text" in part:
                    yield part["text"]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent?key={get_api_key('GOOGLE_VERTEXAI_API_KEY')}"
    headers = make_headers()
    data = {
        "contents": [{"role": "user", "parts": [{"text": context.prompt}]}],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_tokens,
        },
    }
    return await post(context, url, headers, data, chunk_gen)


async def neets_chat(context: ApiContext) -> ApiResult:
    url = "https://api.neets.ai/v1/chat/completions"
    headers = make_headers(x_api_key=get_api_key("NEETS_API_KEY"))
    data = make_openai_chat_body(messages=make_messages(context.prompt))
    return await post(context, url, headers, data, openai_chunk_gen)


async def together_chat(context: ApiContext) -> ApiResult:
    async def chunk_gen(response) -> Generator[str, None, None]:
        async for chunk in make_sse_chunk_gen(response):
            yield chunk["choices"][0].get("text", "")

    url = "https://api.together.xyz/inference"
    headers = make_headers(auth_token=get_api_key("TOGETHER_API_KEY"))
    data = make_openai_chat_body(prompt=context.prompt)
    return await post(context, url, headers, data, chunk_gen)


async def cohere_embed(context: ApiContext) -> ApiResult:
    url = "https://api.cohere.ai/v1/embed"
    headers = make_headers(auth_token=get_api_key("COHERE_API_KEY"))
    data = {
        "model": context.model,
        "texts": [context.prompt],
        "input_type": "search_query",
    }
    return await post(context, url, headers, data)


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


async def fixie_chat(context: ApiContext) -> ApiResult:
    url = f"https://api.fixie.ai/api/v1/agents/{context.model}/conversations"
    headers = make_headers(auth_token=get_api_key("FIXIE_API_KEY"))
    data = {"message": context.prompt, "runtimeParameters": {}}
    return await post(context, url, headers, data, make_fixie_chunk_gen)


async def make_api_call(
    session: aiohttp.ClientSession, index: int, model: str, prompt: str
) -> ApiResult:
    context = ApiContext(session, index, model, prompt)
    if model.startswith("claude-"):
        return await anthropic_chat(context)
    elif model.startswith("@cf/"):
        return await cloudflare_chat(context)
    elif model.startswith("chat-bison") or model.startswith("chat-unicorn"):
        return await google_chat(context)
    elif model.startswith("gemini-"):
        return await gemini_chat(context)
    elif model.startswith("Neets"):
        return await neets_chat(context)
    elif model.startswith("togethercomputer/"):
        return await together_chat(context)
    elif model.startswith("text-embedding-ada-"):
        return await openai_embed(context)
    elif model.startswith("embed-"):
        return await cohere_embed(context)
    # This catch-all needs to be at the end, since it triggers if args.base_url is set.
    elif args.base_url or model.startswith("gpt-") or model.startswith("ft:gpt-"):
        return await openai_chat(context)
    elif "/" in model:
        return await fixie_chat(context)
    else:
        raise ValueError(f"Unknown model: {model}")


async def async_main():
    async with aiohttp.ClientSession() as session:
        if args.warmup:
            # Do a warmup call to make sure the connection is ready
            if args.verbose:
                print("Making a warmup API call...")
            await make_api_call(session, -1, args.model, "")

        fq_model = (
            args.model if not args.base_url else f"{args.base_url[8:]}/{args.model}"
        )
        if not args.minimal:
            print(f"Racing {args.num_requests} API calls to {fq_model}...")
        tasks = [
            asyncio.create_task(make_api_call(session, i, args.model, args.prompt))
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
                print(
                    f"API Call {result.index} failed, status={status} latency={result.latency} text={text}"
                )
            tasks.remove(task)

        # Bail out if no tasks succeed
        if not chosen:
            print("No successful API calls")
            exit(1)

        if not args.minimal:
            print(f"Chosen API Call: {chosen.index} ({chosen.latency:.2f}s)")

        # Stream out the tokens, if we're doing completion
        first_token_time = None
        num_tokens = 0
        if chosen.chunk_gen:
            async for chunk in chosen.chunk_gen:
                num_tokens += 1
                if not first_token_time:
                    first_token_time = time.time()
                if args.print:
                    print(chunk, end="", flush=True)
            end_time = time.time()
            if args.print:
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
    if args.verbose:
        for r in results:
            if r.response.ok:
                print(
                    f"API Call {r.index} Initial Response Latency: {r.latency:.2f} seconds"
                )
            else:
                print(
                    f"API Call {r.index} Result: {r.response.status} in {r.latency:.2f} seconds"
                )
        print("")

    # Print a timing summary
    latency_saved = task1.latency - chosen.latency
    results.sort(key=lambda x: x.latency)
    med_index1 = (len(results) - 1) // 2
    med_index2 = len(results) // 2
    median_latency = (results[med_index1].latency + results[med_index2].latency) / 2
    if num_tokens > 0:
        ttft = first_token_time - chosen.start_time
        tps = (num_tokens - 1) / (end_time - first_token_time)
        total_time = end_time - chosen.start_time
    if not args.minimal:
        print(f"Latency saved: {latency_saved:.2f} seconds")
        print(f"Optimized response time: {chosen.latency:.2f} seconds")
        print(f"Median response time: {median_latency:.2f} seconds")
        if num_tokens > 0:
            print(f"Time to first token: {ttft:.2f} seconds")
            print(f"Tokens: {num_tokens} ({tps:.0f} tokens/sec)")
            print(f"Total time: {total_time:.2f} seconds")
    else:
        print(
            f"{fq_model:48} | {chosen.latency:4.2f} | {ttft:4.2f} | {tps:4.0f} | {total_time:5.2f} | {num_tokens:4}"
        )


asyncio.run(async_main())
