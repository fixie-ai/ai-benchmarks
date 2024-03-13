#!/usr/bin/env python
import argparse
import asyncio
import base64
import dataclasses
import json
import os
import time
import urllib
from typing import Generator, Optional

import aiohttp

DEFAULT_PROMPT = "A pixel art version of the Mona Lisa."
API_VERSION = "2023-12-01-preview"

parser = argparse.ArgumentParser()
parser.add_argument(
    "prompt",
    type=str,
    nargs="?",
    default=DEFAULT_PROMPT,
    help="Prompt to send to the API",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="dall-e-3",
    help="Model to use",
)
parser.add_argument(
    "--num-images",
    "-n",
    type=int,
    default=1,
    help="Number of images to generate",
)
parser.add_argument(
    "--image-size",
    "-s",
    type=int,
    default=1024,
    help="Size of image to generate",
)
parser.add_argument(
    "--base-url",
    "-b",
    type=str,
)
parser.add_argument(
    "--api-key",
    "-k",
    type=str,
)
parser.add_argument(
    "--play",
    "-p",
    action="store_true",
    help="Display the image after generation",
)
parser.add_argument(
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
    def __init__(self, index, start_time, response):
        self.index = index
        self.start_time = start_time
        self.latency = time.time() - start_time
        self.response = response

    index: int
    start_time: int
    latency: float  # HTTP response time
    response: aiohttp.ClientResponse
    chunk_gen: Generator[str, None, None]


async def post(context: ApiContext, url: str, headers: dict, data: dict):
    start_time = time.time()
    response = await context.session.post(url, headers=headers, data=json.dumps(data))
    return ApiResult(context.index, start_time, response)


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
    hostname = urllib.parse.urlparse(url).hostname
    use_azure = hostname and hostname.endswith(".azure.com")
    headers = {
        "Content-Type": "application/json",
    }
    if use_azure:
        api_key = get_api_key("AZURE_OPENAI_API_KEY")
        headers["Api-Key"] = api_key
        url += f"/openai/deployments/{model.replace('.', '')}{path}?api-version={API_VERSION}"
    else:
        api_key = get_api_key("OPENAI_API_KEY")
        headers["Authorization"] = f"Bearer {api_key}"
        url += path
    return url, headers


async def dalle_image(context: ApiContext) -> ApiResult:
    url, headers = make_openai_url_and_headers(context.model, "/images/generations")
    data = {
        "model": context.model,
        "prompt": context.prompt,
        "n": args.num_images,
        "size": f"{args.image_size}x{args.image_size}",
        "response_format": "b64_json",
    }
    return await post(context, url, headers, data)


async def async_main():
    async with aiohttp.ClientSession() as session:
        fq_model = (
            args.model if not args.base_url else f"{args.base_url[8:]}/{args.model}"
        )
        if not args.minimal:
            print(f"Invoking {fq_model}...")
        result = await dalle_image(ApiContext(session, 0, args.model, args.prompt))
        if not result.response.ok:
            print(f"Error: {result.response.status} {result.response.reason}")
            return

        data = await result.response.json()
        end_time = time.time()

    latency = result.latency
    total_time = end_time - result.start_time
    if not args.minimal:
        print(f"Response time: {latency:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
    else:
        print(f"{fq_model:48} | {latency:5.2f} | {total_time:5.2f}")
    if args.play:
        with open("image.png", "wb") as f:
            b64 = data["data"][0]["b64_json"]
            f.write(base64.b64decode(b64))
            os.system("open image.png")


asyncio.run(async_main())
