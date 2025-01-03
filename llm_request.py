import argparse
import asyncio
import base64
import dataclasses
import io
import json
import mimetypes
import os
import re
import time
import urllib
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import aiohttp
import dataclasses_json
import numpy as np
import soundfile as sf
import soxr

TokenGenerator = AsyncGenerator[str, None]
ApiResult = Tuple[aiohttp.ClientResponse, TokenGenerator]

AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
MAX_TPS = 9999
MAX_TTFT = 9.99
MAX_TOTAL_TIME = 99.99


@dataclasses.dataclass
class InputFile:
    @classmethod
    def from_file(cls, path: str):
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            raise ValueError(f"Unknown file type: {path}")
        with open(path, "rb") as f:
            data = f.read()
        return cls(mime_type, data)

    @classmethod
    def from_bytes(cls, mime_type: str, data: bytes):
        return cls(mime_type, data)

    mime_type: str
    data: bytes

    @property
    def is_image(self):
        return self.mime_type.startswith("image/")

    @property
    def is_audio(self):
        return self.mime_type.startswith("audio/")

    @property
    def is_video(self):
        return self.mime_type.startswith("video/")

    @property
    def base64_data(self):
        return base64.b64encode(self.data).decode("utf-8")


@dataclasses.dataclass
class ApiMetrics(dataclasses_json.DataClassJsonMixin):
    model: str
    ttr: Optional[float] = None
    ttft: Optional[float] = None
    tps: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_time: Optional[float] = None
    provider_queue_time: Optional[float] = None
    provider_input_time: Optional[float] = None
    provider_output_time: Optional[float] = None
    provider_total_time: Optional[float] = None
    output: Optional[str] = None
    error: Optional[str] = None


@dataclasses.dataclass
class ApiContext:
    session: aiohttp.ClientSession
    index: int
    name: str
    func: Callable
    model: str
    prompt: str
    files: List[InputFile]
    tools: List[Dict]
    strict: bool
    temperature: float
    max_tokens: int
    detail: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    peft: Optional[str] = None
    ws: Optional[aiohttp.ClientWebSocketResponse] = None

    def __init__(self, session, index, name, func, args, prompt, files, tools):
        self.session = session
        self.index = index
        self.name = name
        self.func = func
        self.model = args.model
        self.prompt = prompt
        self.files = files
        self.tools = tools
        self.strict = args.strict
        self.detail = args.detail
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.api_key = args.api_key
        self.base_url = args.base_url
        self.peft = args.peft
        self.metrics = ApiMetrics(model=self.name)

    @property
    def is_warmup(self):
        return self.index == -1

    async def run(self, on_token: Optional[Callable[["ApiContext", str], None]] = None):
        response = None
        try:
            start_time = time.time()
            first_token_time = None
            response, chunk_gen = await self.func(self)
            self.metrics.ttr = time.time() - start_time
            if response.ok:
                if chunk_gen:
                    self.metrics.output_tokens = 0
                    self.metrics.output = ""
                    async for chunk in chunk_gen:
                        self.metrics.output += chunk
                        self.metrics.output_tokens += 1
                        if not first_token_time:
                            first_token_time = time.time()
                            self.metrics.ttft = first_token_time - start_time
                        if on_token and chunk:
                            on_token(self, chunk)
                    if first_token_time:
                        # Signal the end of the generation.
                        if on_token:
                            on_token(self, "")
                    else:
                        self.metrics.error = "No tokens received"
            else:
                text = await response.text()
                self.metrics.error = f"{response.status} {response.reason} {text}"
        except TimeoutError:
            self.metrics.error = "Timeout"
        except aiohttp.ClientError as e:
            self.metrics.error = str(e)
        end_time = time.time()
        if not self.metrics.error:
            token_time = end_time - first_token_time
            self.metrics.total_time = end_time - start_time
            self.metrics.tps = min(
                (self.metrics.output_tokens - 1) / token_time, MAX_TPS
            )
            if self.metrics.tps == MAX_TPS:
                self.metrics.tps = 0.0
        else:
            self.metrics.ttft = MAX_TTFT
            self.metrics.tps = 0.0
            self.metrics.total_time = MAX_TOTAL_TIME
        if response:
            await response.release()


async def post(
    ctx: ApiContext,
    url: str,
    headers: dict,
    data: dict,
    make_chunk_gen: Optional[Callable[[aiohttp.ClientResponse], TokenGenerator]] = None,
):
    response = await ctx.session.post(url, headers=headers, data=json.dumps(data))
    chunk_gen = make_chunk_gen(ctx, response) if make_chunk_gen else None
    return response, chunk_gen


def get_api_key(ctx: ApiContext, env_var: str) -> str:
    if ctx.api_key:
        return ctx.api_key
    if env_var in os.environ:
        return os.environ[env_var]
    raise ValueError(f"Missing API key: {env_var}")


def make_headers(
    auth_token: Optional[str] = None,
    api_key: Optional[str] = None,
    x_api_key: Optional[str] = None,
):
    headers = {
        "content-type": "application/json",
    }
    if auth_token:
        headers["authorization"] = f"Bearer {auth_token}"
    if api_key:
        headers["api-key"] = api_key
    if x_api_key:
        headers["x-api-key"] = x_api_key
    return headers


def make_openai_url_and_headers(ctx: ApiContext, path: str):
    url = ctx.base_url or "https://api.openai.com/v1"
    hostname = urllib.parse.urlparse(url).hostname
    use_azure_openai = hostname and hostname.endswith("openai.azure.com")
    use_ovh = hostname and hostname.endswith("cloud.ovh.net")
    if use_azure_openai:
        api_key = get_api_key(ctx, "AZURE_OPENAI_API_KEY")
        headers = make_headers(api_key=api_key)
        url += f"/openai/deployments/{ctx.model.replace('.', '')}{path}?api-version={AZURE_OPENAI_API_VERSION}"
    elif use_ovh:
        api_key = get_api_key(ctx, "OVH_AI_ENDPOINTS_API_KEY")
        headers = {
            "content-type": "application/json",
            "authorization": api_key
        }
        url += path
    else:
        api_key = ctx.api_key if ctx.base_url else get_api_key(ctx, "OPENAI_API_KEY")
        headers = make_headers(auth_token=api_key)
        url += path
    return url, headers


def make_openai_messages(ctx: ApiContext):
    if not ctx.files:
        return [{"role": "user", "content": ctx.prompt}]

    content: List[Dict[str, Any]] = [{"type": "text", "text": ctx.prompt}]
    for file in ctx.files:
        url = f"data:{file.mime_type};base64,{file.base64_data}"
        media_url = {"url": url}
        url_type = "audio_url" if file.is_audio else "image_url"
        if ctx.detail:
            media_url["detail"] = ctx.detail
        content.append({"type": url_type, url_type: media_url})
    return [{"role": "user", "content": content}]


def make_openai_ws_message(ctx: ApiContext):
    content = [{"type": "input_text", "text": ctx.prompt}]
    for file in ctx.files:
        if file.is_audio:
            audio, sr = sf.read(io.BytesIO(file.data))
            audio_24k = soxr.resample(audio, sr, 24000)
            audio_pcm = (audio_24k * 32767).astype(np.int16).tobytes()
            b64_data = base64.b64encode(audio_pcm).decode("utf-8")
            content.append({"type": "input_audio", "audio": b64_data})
        else:
            raise NotImplementedError("Images not yet supported in WebSocket mode")
    return {"type": "message", "role": "user", "content": content}


def make_openai_chat_body(ctx: ApiContext, **kwargs):
    # Models differ in how they want to receive the prompt, so
    # we let the caller specify the key and format.
    body = {
        "model": ctx.model or None,
        "max_tokens": ctx.max_tokens,
        "temperature": ctx.temperature,
        "stream": True,
    }
    for key, value in kwargs.items():
        body[key] = value
    return body


async def make_sse_chunk_gen(response) -> AsyncGenerator[Dict[str, Any], None]:
    done = False
    async for line in response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data:"):
            content = line[5:].strip()
            if content == "[DONE]":
                done = True
            elif not done:
                yield json.loads(content)


async def openai_chunk_gen(ctx: ApiContext, response) -> TokenGenerator:
    async for chunk in make_sse_chunk_gen(response):
        if chunk.get("choices", []):
            delta = chunk["choices"][0]["delta"]
            delta_content = delta.get("content")
            delta_tool = delta.get("tool_calls")
            if delta_content:
                yield delta_content
            elif delta_tool:
                function = delta_tool[0]["function"]
                name = function.get("name", "").strip()
                if name:
                    yield name
                args = function.get("arguments", "").strip()
                if args:
                    yield args
        usage = chunk.get("usage") or chunk.get("x_groq", {}).get("usage")
        if usage:
            ctx.metrics.input_tokens = usage.get("prompt_tokens")
            ctx.metrics.output_tokens = usage.get("completion_tokens")
            ctx.metrics.provider_queue_time = usage.get("queue_time")
            ctx.metrics.provider_input_time = usage.get("prompt_time")
            ctx.metrics.provider_output_time = usage.get("completion_time")
            ctx.metrics.provider_total_time = usage.get("total_time")


async def openai_chat(ctx: ApiContext, path: str = "/chat/completions") -> ApiResult:
    url, headers = make_openai_url_and_headers(ctx, path)
    kwargs = {"messages": make_openai_messages(ctx)}
    if ctx.tools:
        tools = ctx.tools[:]
        if ctx.strict:
            for t in tools:
                t["function"]["strict"] = True
                t["function"]["parameters"]["additionalProperties"] = False
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "required"
    if ctx.peft:
        kwargs["peft"] = ctx.peft
    # Some providers require opt-in for stream stats, but some providers don't like this opt-in.
    # Regardless of opt-in, Azure and ovh.net don't return stream stats at the moment.
    # See https://github.com/Azure/azure-rest-api-specs/issues/25062
    if not any(p in ctx.name for p in ["azure", "databricks", "fireworks", "mistral"]):
        kwargs["stream_options"] = {"include_usage": True}
    # Hack to identify our baseten deployment, which isn't contained in the URL.
    if ctx.name.startswith("baseten"):
        kwargs["baseten"] = {"model_id": "rwn2v41w"}
    data = make_openai_chat_body(ctx, **kwargs)
    return await post(ctx, url, headers, data, openai_chunk_gen)


async def openai_embed(ctx: ApiContext) -> ApiResult:
    url, headers = make_openai_url_and_headers(ctx, "/embeddings")
    data = {"model": ctx.model, "input": ctx.prompt}
    return await post(ctx, url, headers, data)


class WebSocketResponse:
    """Mirrors the aiohttp.ClientHttpResponse interface, but for a WebSocket."""

    def __init__(self, ctx: ApiContext):
        self.ctx = ctx

    @property
    def ok(self):
        return True

    async def release(self):
        if not self.ctx.is_warmup:
            await self.ctx.ws.close()


async def openai_ws(ctx: ApiContext) -> ApiResult:
    async def warmup_gen() -> TokenGenerator:
        yield " "

    async def chunk_gen(ctx: ApiContext) -> TokenGenerator:
        async for msg in ctx.ws:
            chunk = json.loads(msg.data)
            match chunk["type"]:
                case "error":
                    print(chunk)
                    break
                case "response.text.delta":
                    yield chunk["delta"]
                case "response.audio_transcript.delta":
                    yield chunk["delta"]
                case "response.done":
                    response = chunk["response"]
                    ctx.metrics.input_tokens = response["usage"]["input_tokens"]
                    ctx.metrics.output_tokens = response["usage"]["output_tokens"]
                    break

    if not ctx.ws:
        base_url = ctx.base_url or "wss://api.openai.com/v1/realtime"
        url = f"{base_url}?model={ctx.model}"
        api_key = get_api_key(ctx, "OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}
        ctx.ws = await ctx.session.ws_connect(url, headers=headers)
        if ctx.is_warmup:
            return WebSocketResponse(ctx), warmup_gen()

    create_item = {
        "type": "conversation.item.create",
        "item": make_openai_ws_message(ctx),
    }
    await ctx.ws.send_json(create_item)

    modalities = ["text"]
    if any(file.is_audio for file in ctx.files):
        modalities.append("audio")
    create_response = {
        "type": "response.create",
        "response": {"modalities": modalities},
    }
    await ctx.ws.send_json(create_response)
    return WebSocketResponse(ctx), chunk_gen(ctx)


def make_anthropic_messages(prompt: str, files: Optional[List[InputFile]] = None):
    """Formats the prompt as a text chunk and any images as image chunks.
    Note that Anthropic's image protocol is somewhat different from OpenAI's."""
    if not files:
        return [{"role": "user", "content": prompt}]

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for file in files:
        if not file.mime_type.startswith("image/"):
            raise ValueError(f"Unsupported file type: {file.mime_type}")
        source = {
            "type": "base64",
            "media_type": file.mime_type,
            "data": file.base64_data,
        }
        content.append({"type": "image", "source": source})
    return [{"role": "user", "content": content}]


async def anthropic_chat(ctx: ApiContext) -> ApiResult:
    """Make an Anthropic chat completion request. The request protocol is similar to OpenAI's,
    but the response protocol is completely different."""

    async def chunk_gen(ctx: ApiContext, response) -> TokenGenerator:
        async for chunk in make_sse_chunk_gen(response):
            delta = chunk.get("delta")
            if delta and delta.get("type") == "text_delta":
                yield delta["text"]

            type = chunk.get("type")
            if type == "message_start":
                usage = chunk["message"].get("usage")
                if usage:
                    ctx.metrics.input_tokens = usage.get("input_tokens")
            elif type == "message_delta":
                usage = chunk.get("usage")
                if usage:
                    ctx.metrics.output_tokens = usage.get("output_tokens")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": get_api_key(ctx, "ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "messages-2023-12-15",
    }
    # Anthropic's schema is slightly different than OpenAI's.
    tools = [t["function"].copy() for t in ctx.tools]
    for tool in tools:
        tool["input_schema"] = tool["parameters"]
        del tool["parameters"]
    data = make_openai_chat_body(
        ctx, messages=make_anthropic_messages(ctx.prompt, ctx.files), tools=tools
    )
    return await post(ctx, url, headers, data, chunk_gen)


async def cohere_chat(ctx: ApiContext) -> ApiResult:
    """Make a Cohere chat completion request."""

    async def chunk_gen(ctx: ApiContext, response) -> TokenGenerator:
        async for line in response.content:
            chunk = json.loads(line)
            if chunk.get("event_type") == "text-generation" and "text" in chunk:
                yield chunk["text"]
            elif chunk.get("event_type") == "stream-end":
                meta = chunk["response"]["meta"]
                ctx.metrics.input_tokens = meta["tokens"]["input_tokens"]
                ctx.metrics.output_tokens = meta["tokens"]["output_tokens"]

    url = "https://api.cohere.ai/v1/chat"
    headers = make_headers(auth_token=get_api_key(ctx, "COHERE_API_KEY"))
    data = make_openai_chat_body(ctx, message=ctx.prompt)
    return await post(ctx, url, headers, data, chunk_gen)


async def cloudflare_chat(ctx: ApiContext) -> ApiResult:
    """Make a Cloudflare chat completion request. The protocol is similar to OpenAI's,
    but the URL doesn't follow the same scheme and the response structure is different.
    """

    async def chunk_gen(ctx: ApiContext, response) -> TokenGenerator:
        async for chunk in make_sse_chunk_gen(response):
            yield chunk["response"]

    account_id = os.environ["CF_ACCOUNT_ID"]
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{ctx.model}"
    )
    headers = make_headers(auth_token=get_api_key(ctx, "CF_API_KEY"))
    data = make_openai_chat_body(ctx, messages=make_openai_messages(ctx))
    return await post(ctx, url, headers, data, chunk_gen)


async def make_json_chunk_gen(response) -> AsyncGenerator[Dict[str, Any], None]:
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


def get_google_access_token():
    from google.auth.transport import requests
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(
        "service_account.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    if not creds.token:
        creds.refresh(requests.Request())
    return creds.token


def make_google_url_and_headers(ctx: ApiContext, method: str):
    region = "us-west1"
    project_id = os.environ["GCP_PROJECT"]
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{ctx.model}:{method}"
    api_key = ctx.api_key
    if not api_key:
        api_key = get_google_access_token()
    headers = make_headers(auth_token=api_key)
    return url, headers


def make_gemini_messages(prompt: str, files: List[InputFile]):
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for file in files:
        parts.append(
            {"inline_data": {"mime_type": file.mime_type, "data": file.base64_data}}
        )

    return [{"role": "user", "parts": parts}]


async def gemini_chat(ctx: ApiContext) -> ApiResult:
    async def chunk_gen(ctx: ApiContext, response) -> TokenGenerator:
        async for chunk in make_json_chunk_gen(response):
            candidates = chunk.get("candidates")
            if candidates:
                content = candidates[0].get("content")
                if content and "parts" in content:
                    part = content["parts"][0]
                    if "text" in part:
                        yield part["text"]
                    elif "functionCall" in part:
                        call = part["functionCall"]
                        if "name" in call:
                            yield call["name"]
                        if "args" in call:
                            yield str(call["args"])
            usage = chunk.get("usageMetadata")
            if usage:
                ctx.metrics.input_tokens = usage.get("promptTokenCount")
                ctx.metrics.output_tokens = usage.get("candidatesTokenCount")

    # The Google AI Gemini API (URL below) doesn't return the number of generated tokens.
    # Instead we use the Google Cloud Vertex AI Gemini API, which does return the number of tokens, but requires an Oauth credential.
    # Also, setting safetySettings to BLOCK_NONE is not supported in the Vertex AI Gemini API, at least for now.
    if True:
        url, headers = make_google_url_and_headers(ctx, "streamGenerateContent")
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{ctx.model}:streamGenerateContent?key={get_api_key(ctx, 'GOOGLE_GEMINI_API_KEY')}"
        headers = make_headers()
    harm_categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    data = {
        "contents": make_gemini_messages(ctx.prompt, ctx.files),
        "generationConfig": {
            "temperature": ctx.temperature,
            "maxOutputTokens": ctx.max_tokens,
        },
        "safetySettings": [
            {"category": category, "threshold": "BLOCK_NONE"}
            for category in harm_categories
            if not ctx.files or ctx.files[0].is_image
        ],
    }
    if ctx.tools:
        data["tools"] = (
            [{"function_declarations": [tool["function"] for tool in ctx.tools]}],
        )
    return await post(ctx, url, headers, data, chunk_gen)


async def cohere_embed(ctx: ApiContext) -> ApiResult:
    url = "https://api.cohere.ai/v1/embed"
    headers = make_headers(auth_token=get_api_key(ctx, "COHERE_API_KEY"))
    data = {
        "model": ctx.model,
        "texts": [ctx.prompt],
        "input_type": "search_query",
    }
    return await post(ctx, url, headers, data)


async def fake_chat(ctx: ApiContext) -> ApiResult:
    class FakeResponse(aiohttp.ClientResponse):
        def __init__(self, status, reason):
            self.status = status
            self.reason = reason

        # async def release(self):
        # pass

    async def make_fake_chunk_gen(output: str):
        for word in output.split():
            yield word + " "
            await asyncio.sleep(0.05)

    output = "This is a fake response."
    if ctx.index % 2 == 0:
        response = FakeResponse(200, "OK")
    else:
        response = FakeResponse(500, "Internal Server Error")
    sleep = 0.5 * (ctx.index + 1)
    max_sleep = ctx.session.timeout.total
    if max_sleep:
        await asyncio.sleep(min(sleep, max_sleep))
    if sleep > max_sleep:
        raise TimeoutError
    return (response, make_fake_chunk_gen(output))


def make_display_name(provider_or_url: str, model: str) -> str:
    # Clean up the base URL to get a nicer provider name.
    if provider_or_url.startswith("https://"):
        provider = (
            provider_or_url[8:]
            .split("/")[0]
            .replace("openai-sub-with-gpt4", "eastus2")
            .replace("fixie-", "")
            .replace("-serverless", "")
            .replace("inference.ai.azure.com", "azure")
            .replace("openai.azure.com", "azure")
        )
        # Get the last two segments of the domain, and swap foo.azure to azure.foo.
        provider = ".".join(provider.split(".")[-2:])
        provider = re.sub(r"(\w+)\.azure$", r"azure.\1", provider)
    else:
        provider = provider_or_url
    model_segments = model.split("/")
    if provider:
        # We already have a provider, so just need to add the model name.
        # If we've got a model name, add the end of the split to the provider.
        # Otherwise, we have model.domain.com, so we need to swap to domain.com/model.
        if model:
            name = provider + "/" + model_segments[-1]
        else:
            domain_segments = provider.split(".")
            name = ".".join(domain_segments[1:]) + "/" + domain_segments[0]
    elif len(model_segments) > 1:
        # We've got a provider/model string, from which we need to get the provider and model.
        provider = model_segments[0]
        name = provider + "/" + model_segments[-1]
    return name


def make_context(
    session: aiohttp.ClientSession,
    index: int,
    args: argparse.Namespace,
    prompt: Optional[str] = None,
    files: Optional[List[InputFile]] = None,
    tools: Optional[List[Dict]] = None,
) -> ApiContext:
    model = args.model
    prefix = re.split("-|/", model)[0]
    provider = args.base_url
    match prefix:
        case "claude":
            provider = "anthropic"
            func = anthropic_chat
        case "command":
            provider = "cohere"
            func = cohere_chat
        case "@cf":
            provider = "cloudflare"
            func = cloudflare_chat
        case "gemini":
            provider = "google"
            func = gemini_chat
        case "text-embedding-ada":
            provider = "openai"
            func = openai_embed
        case "embed":
            provider = "cohere"
            func = cohere_embed
        case "fake":
            provider = "test"
            func = fake_chat
        case _ if "realtime" in model:
            func = openai_ws
            if not args.base_url:
                provider = "openai"
        case _ if args.base_url or model.startswith("gpt-") or model.startswith(
            "ft:gpt-"
        ):
            func = openai_chat
            if not args.base_url:
                provider = "openai"
        case _:
            raise ValueError(f"Unknown model: {model}")
    name = args.display_name or make_display_name(provider, model)
    return ApiContext(
        session, index, name, func, args, prompt or "", files or [], tools or []
    )
