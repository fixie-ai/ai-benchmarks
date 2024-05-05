import argparse
import asyncio
import dataclasses
import datetime
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import gcloud.aio.storage as gcs

import llm_benchmark

DEFAULT_DISPLAY_LENGTH = 64
DEFAULT_GCS_BUCKET = "thefastest-data"
GPT_4_TURBO = "gpt-4-turbo"
GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
GPT_35_TURBO = "gpt-3.5-turbo"
GPT_35_TURBO_0125 = "gpt-3.5-turbo-0125"
GPT_35_TURBO_1106 = "gpt-3.5-turbo-1106"
LLAMA_3_70B_CHAT = "llama-3-70b-chat"
LLAMA_3_8B_CHAT = "llama-3-8b-chat"
LLAMA_2_70B_CHAT = "llama-2-70b-chat"
LLAMA_2_13B_CHAT = "llama-2-13b-chat"
LLAMA_2_7B_CHAT = "llama-2-7b-chat"
MIXTRAL_8X22B_INSTRUCT = "mixtral-8x22b-instruct"
MIXTRAL_8X7B_INSTRUCT = "mixtral-8x7b-instruct"
PHI_2 = "phi-2"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--format",
    "-F",
    choices=["text", "json"],
    default="text",
    help="Output results in the specified format",
)
parser.add_argument(
    "--mode",
    "-m",
    choices=["text", "image", "audio", "video"],
    default="text",
    help="Mode to run benchmarks for",
)
parser.add_argument(
    "--filter",
    "-r",
    help="Filter models by name",
)
parser.add_argument(
    "--spread",
    "-s",
    type=float,
    default=0.0,
    help="Spread the requests out over the specified time in seconds",
)
parser.add_argument(
    "--display-length",
    "-l",
    type=int,
    default=DEFAULT_DISPLAY_LENGTH,
    help="Amount of the generation response to display",
)
parser.add_argument(
    "--store",
    action="store_true",
    help="Store the results in the configured GCP bucket",
)


def _dict_to_argv(d: Dict[str, Any]) -> List[str]:
    return [
        f"--{k.replace('_', '-')}" + (f"={v}" if v or v == 0 else "")
        for k, v in d.items()
    ]


class _Llm:
    """
    We maintain a dict of params for the llm, as well as any
    command-line flags that we didn't already handle. We'll
    turn this into a single command line for llm_benchmark.run
    to consume, which allows us to reuse the parsing logic
    from that script, rather than having to duplicate it here.
    """

    def __init__(self, model: str, display_name: Optional[str] = None, **kwargs):
        self.args = {
            "model": model,
            "format": "none",
            **kwargs,
        }
        if display_name:
            self.args["display_name"] = display_name

    async def run(self, pass_argv: List[str], spread: float) -> asyncio.Task:
        if spread:
            await asyncio.sleep(spread)
        full_argv = _dict_to_argv(self.args) + pass_argv
        return await llm_benchmark.run(full_argv)


class _AnyscaleLlm(_Llm):
    """See https://docs.endpoints.anyscale.com/text-generation/query-a-model"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "anyscale.com/" + (display_model or model),
            api_key=os.getenv("ANYSCALE_API_KEY"),
            base_url="https://api.endpoints.anyscale.com/v1",
        )


class _CloudflareLlm(_Llm):
    """See https://developers.cloudflare.com/workers-ai/models/"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "cloudflare.com/" + (display_model or model),
        )


class _DatabricksLlm(_Llm):
    """See https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "databricks.com/" + (display_model or model),
            api_key=os.getenv("DATABRICKS_TOKEN"),
            base_url="https://adb-1558081827343359.19.azuredatabricks.net/serving-endpoints",
        )


class _FireworksLlm(_Llm):
    """See https://fireworks.ai/models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "fireworks.ai/" + (display_model or model),
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1",
        )


class _GroqLlm(_Llm):
    """See https://console.groq.com/docs/models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "groq.com/" + (display_model or model),
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )


class _OctoLlm(_Llm):
    """See https://octo.ai/docs/getting-started/inference-models#serverless-endpoints"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "octo.ai/" + (display_model or model),
            api_key=os.getenv("OCTOML_API_KEY"),
            base_url="https://text.octoai.run/v1",
        )


class _PerplexityLlm(_Llm):
    """See https://docs.perplexity.ai/docs/model-cards"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "perplexity.ai/" + (display_model or model),
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
        )


class _TogetherLlm(_Llm):
    """See https://docs.together.ai/docs/inference-models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "together.ai/" + (display_model or model),
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )


def _text_models():
    AZURE_EASTUS2_OPENAI_API_KEY = os.getenv("AZURE_EASTUS2_OPENAI_API_KEY")
    return [
        # GPT-4
        _Llm(GPT_4_TURBO),
        _Llm(GPT_4_0125_PREVIEW),
        _Llm(
            GPT_4_0125_PREVIEW,
            api_key=os.getenv("AZURE_SCENTRALUS_OPENAI_API_KEY"),
            base_url="https://fixie-scentralus.openai.azure.com",
        ),
        _Llm(GPT_4_1106_PREVIEW),
        _Llm(GPT_4_1106_PREVIEW, base_url="https://fixie-westus.openai.azure.com"),
        _Llm(
            GPT_4_1106_PREVIEW,
            api_key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        _Llm(
            GPT_4_1106_PREVIEW,
            api_key=os.getenv("AZURE_FRCENTRAL_OPENAI_API_KEY"),
            base_url="https://fixie-frcentral.openai.azure.com",
        ),
        _Llm(
            GPT_4_1106_PREVIEW,
            api_key=os.getenv("AZURE_SECENTRAL_OPENAI_API_KEY"),
            base_url="https://fixie-secentral.openai.azure.com",
        ),
        _Llm(
            GPT_4_1106_PREVIEW,
            api_key=os.getenv("AZURE_UKSOUTH_OPENAI_API_KEY"),
            base_url="https://fixie-uksouth.openai.azure.com",
        ),
        # GPT-3.5
        _Llm(GPT_35_TURBO_0125),
        _Llm(GPT_35_TURBO_1106),
        _Llm(GPT_35_TURBO_1106, base_url="https://fixie-westus.openai.azure.com"),
        _Llm(
            GPT_35_TURBO,
            api_key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        # Claude
        _Llm("claude-3-opus-20240229"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("claude-3-haiku-20240307"),
        _Llm("claude-2.1"),
        _Llm("claude-instant-1.2"),
        # Cohere
        _Llm("command-r-plus"),
        _Llm("command-r"),
        _Llm("command-light"),
        # Gemini
        _Llm("gemini-pro"),
        _Llm("gemini-1.5-pro-preview-0409"),
        # Mistral 8x22b
        _Llm(
            "mistral-large",  # is this the same?
            api_key=os.getenv("AZURE_EASTUS2_MISTRAL_API_KEY"),
            base_url="https://fixie-mistral-serverless.eastus2.inference.ai.azure.com/v1",
        ),
        _AnyscaleLlm("mistralai/Mixtral-8x22B-Instruct-v0.1", MIXTRAL_8X22B_INSTRUCT),
        _FireworksLlm(
            "accounts/fireworks/models/mixtral-8x22b-instruct", MIXTRAL_8X22B_INSTRUCT
        ),
        _OctoLlm("mixtral-8x22b-instruct", MIXTRAL_8X22B_INSTRUCT),
        _TogetherLlm("mistralai/Mixtral-8x22B-Instruct-v0.1", MIXTRAL_8X22B_INSTRUCT),
        # Mistral 8x7b
        _AnyscaleLlm("mistralai/Mixtral-8x7B-Instruct-v0.1", MIXTRAL_8X7B_INSTRUCT),
        _DatabricksLlm("databricks-mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT),
        _FireworksLlm(
            "accounts/fireworks/models/mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT
        ),
        _GroqLlm("mixtral-8x7b-32768", MIXTRAL_8X7B_INSTRUCT),
        _OctoLlm("mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT),
        _PerplexityLlm("mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT),
        _PerplexityLlm("sonar-medium-chat"),
        _TogetherLlm("mistralai/Mixtral-8x7B-Instruct-v0.1", MIXTRAL_8X7B_INSTRUCT),
        # Llama 3 70b
        _AnyscaleLlm("meta-llama/Llama-3-70b-chat-hf", LLAMA_3_70B_CHAT),
        _DatabricksLlm("databricks-meta-llama-3-70b-instruct", LLAMA_3_70B_CHAT),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3-70b-instruct", LLAMA_3_70B_CHAT
        ),
        _GroqLlm("llama3-70b-8192", LLAMA_3_70B_CHAT),
        _OctoLlm("meta-llama-3-70b-instruct", LLAMA_3_70B_CHAT),
        _PerplexityLlm("llama-3-70b-instruct", LLAMA_3_70B_CHAT),
        _TogetherLlm("meta-llama/Llama-3-70b-chat-hf", LLAMA_3_70B_CHAT),
        # Llama 2 70b
        _Llm(
            LLAMA_2_70B_CHAT,
            api_key=os.getenv("AZURE_WESTUS3_LLAMA2_API_KEY"),
            base_url="https://fixie-llama-2-70b-serverless.westus3.inference.ai.azure.com/v1",
        ),
        _Llm(
            LLAMA_2_70B_CHAT,
            api_key=os.getenv("AZURE_EASTUS2_LLAMA2_API_KEY"),
            base_url="https://fixie-llama-2-70b-serverless.eastus2.inference.ai.azure.com/v1",
        ),
        _AnyscaleLlm("meta-llama/Llama-2-70b-chat-hf", LLAMA_2_70B_CHAT),
        _DatabricksLlm("databricks-llama-2-70b-chat", LLAMA_2_70B_CHAT),
        _FireworksLlm("accounts/fireworks/models/llama-v2-70b-chat", LLAMA_2_70B_CHAT),
        _GroqLlm("llama2-70b-4096", LLAMA_2_70B_CHAT),
        _OctoLlm("llama-2-70b-chat-fp16", LLAMA_2_70B_CHAT),
        _TogetherLlm("togethercomputer/llama-2-70b-chat", LLAMA_2_70B_CHAT),
        # Llama 2 13b
        _AnyscaleLlm("meta-llama/Llama-2-13b-chat-hf", LLAMA_2_13B_CHAT),
        _FireworksLlm("accounts/fireworks/models/llama-v2-13b-chat", LLAMA_2_13B_CHAT),
        _OctoLlm("llama-2-13b-chat-fp16", LLAMA_2_13B_CHAT),
        _TogetherLlm("togethercomputer/llama-2-13b-chat", LLAMA_2_13B_CHAT),
        # Llama 3 8b
        _AnyscaleLlm("meta-llama/Llama-3-8b-chat-hf", LLAMA_3_8B_CHAT),
        _CloudflareLlm("@cf/meta/llama-3-8b-instruct", LLAMA_3_8B_CHAT),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3-8b-instruct", LLAMA_3_8B_CHAT
        ),
        _GroqLlm("llama3-8b-8192", LLAMA_3_8B_CHAT),
        _OctoLlm("meta-llama-3-8b-instruct", LLAMA_3_8B_CHAT),
        _PerplexityLlm("llama-3-8b-instruct", LLAMA_3_8B_CHAT),
        _TogetherLlm("meta-llama/Llama-3-8b-chat-hf", LLAMA_3_8B_CHAT),
        # Llama 2 7b
        _AnyscaleLlm("meta-llama/Llama-2-7b-chat-hf", LLAMA_2_7B_CHAT),
        _CloudflareLlm("@cf/meta/llama-2-7b-chat-fp16", LLAMA_2_7B_CHAT),
        # _DatabricksLlm("fixie-llama-2-7b", LLAMA_2_7B_CHAT),
        _FireworksLlm("accounts/fireworks/models/llama-v2-7b-chat", LLAMA_2_7B_CHAT),
        _TogetherLlm("togethercomputer/llama-2-7b-chat", LLAMA_2_7B_CHAT),
        # Phi-2
        _CloudflareLlm("@cf/microsoft/phi-2", PHI_2),
        _TogetherLlm("microsoft/phi-2", PHI_2),
    ]


def _image_models():
    return [
        _Llm("gpt-4-turbo"),
        _Llm("gpt-4-vision-preview", base_url="https://fixie-westus.openai.azure.com"),
        _Llm("claude-3-opus-20240229"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("gemini-pro-vision"),
        _Llm("gemini-1.5-pro-preview-0409"),
        _FireworksLlm("accounts/fireworks/models/firellava-13b", "firellava-13b"),
    ]


def _av_models():
    return [
        _Llm("gemini-1.5-pro-preview-0409"),
    ]


def _get_models(mode: str, filter: Optional[str] = None):
    mode_map = {
        "text": _text_models,
        "image": _image_models,
        "audio": _av_models,
        "video": _av_models,
    }
    if mode not in mode_map:
        raise ValueError(f"Unknown mode {mode}")
    models = mode_map[mode]()
    return [m for m in models if not filter or filter in m.args["model"].lower()]


def _get_prompt(mode: str) -> List[str]:
    if mode == "text":
        return ["Write a nonet about a sunset."]
    elif mode == "image":
        return [
            "Based on the image, explain what will happen next.",
            "--file",
            "media/image/inception.jpeg",
        ]


@dataclasses.dataclass
class _Response:
    time: str
    duration: str
    region: str
    cmd: str
    results: List[Dict[str, Any]]


def _format_response(
    response: _Response, format: str, dlen: int = 0
) -> Tuple[str, str]:
    if format == "json":
        return json.dumps(vars(response), indent=2), "application/json"
    else:
        s = (
            "| Provider/Model                             | TTR  | TTFT | TPS | Tok | Total |"
            f" {'Response':{dlen}.{dlen}} |\n"
            "| :----------------------------------------- | ---: | ---: | --: | --: | ----: |"
            f" {':--':-<{dlen}.{dlen}} |\n"
        )

        for r in response.results:
            output = r["output"].replace("\n", "\\n").strip()
            s += (
                f"| {r['model']:42} | {r['ttr']:4.2f} | {r['ttft']:4.2f} | "
                f"{r['tps']:3.0f} | {r['num_tokens']:3} | {r['total_time']:5.2f} | "
                f"{output:{dlen}.{dlen}} |\n"
            )

        s += f"\ntime: {response.time}, duration: {response.duration} region: {response.region}, cmd: {response.cmd}\n"
        return s, "text/markdown"


async def _store_response(gcp_bucket: str, key: str, text: str, content_type: str):
    print(f"Storing results in {gcp_bucket}/{key}")
    storage = gcs.Storage(service_file="service_account.json")
    await storage.upload(gcp_bucket, key, text, content_type=content_type)
    await storage.close()


async def _run(argv: List[str]) -> Tuple[str, str]:
    """
    This function is invoked either from the webapp (via run) or the main function below.
    The args we know about are stored in args, and any unknown args are stored in pass_argv,
    which we'll pass to the _Llm.run function, who will turn them back into a
    single list of flags for consumption by the llm_benchmark.run function.
    """
    time_start = datetime.datetime.now()
    time_str = time_start.isoformat()
    region = os.getenv("FLY_REGION", "local")
    cmd = " ".join(argv)
    args, pass_argv = parser.parse_known_args(argv)
    pass_argv += _get_prompt(args.mode)
    models = _get_models(args.mode, args.filter)
    tasks = []
    for m in models:
        delay = random.uniform(0, args.spread)
        tasks.append(asyncio.create_task(m.run(pass_argv, delay)))
    await asyncio.gather(*tasks)
    results = [t.result() for t in tasks if t.result() is not None]
    elapsed = datetime.datetime.now() - time_start
    elapsed_str = f"{elapsed.total_seconds():.2f}s"
    response = _Response(time_str, elapsed_str, region, cmd, results)
    if args.store:
        path = f"{region}/{args.mode}/{time_str.split('T')[0]}.json"
        json, content_type = _format_response(response, "json")
        await _store_response(DEFAULT_GCS_BUCKET, path, json, content_type)
    return _format_response(response, args.format, args.display_length)


async def run(params: Dict[str, Any]) -> Tuple[str, str]:
    return await _run(_dict_to_argv(params))


async def main():
    text, _ = await _run(sys.argv[1:])
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
