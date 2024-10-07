import argparse
import asyncio
import dataclasses
import datetime
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import dataclasses_json
import gcloud.aio.storage as gcs

import llm_benchmark
import llm_request

DEFAULT_DISPLAY_LENGTH = 64
DEFAULT_GCS_BUCKET = "thefastest-data"

GPT_4O_REALTIME_PREVIEW = "gpt-4o-realtime-preview-2024-10-01"
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
GPT_4_TURBO = "gpt-4-turbo"
GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
GPT_35_TURBO = "gpt-3.5-turbo"
GPT_35_TURBO_0125 = "gpt-3.5-turbo-0125"
GPT_35_TURBO_1106 = "gpt-3.5-turbo-1106"
GEMINI_1_5_PRO = "gemini-1.5-pro"
GEMINI_1_5_FLASH = "gemini-1.5-flash"
LLAMA_31_405B_CHAT = "llama-3.1-405b-chat"
LLAMA_31_405B_CHAT_FP8 = "llama-3.1-405b-chat-fp8"
LLAMA_31_70B_CHAT = "llama-3.1-70b-chat"
LLAMA_31_70B_CHAT_FP8 = "llama-3.1-70b-chat-fp8"
LLAMA_31_8B_CHAT = "llama-3.1-8b-chat"
LLAMA_31_8B_CHAT_FP8 = "llama-3.1-8b-chat-fp8"
LLAMA_3_70B_CHAT = "llama-3-70b-chat"
LLAMA_3_70B_CHAT_FP8 = "llama-3-70b-chat-fp8"
LLAMA_3_70B_CHAT_FP4 = "llama-3-70b-chat-fp4"
LLAMA_3_8B_CHAT = "llama-3-8b-chat"
LLAMA_3_8B_CHAT_FP8 = "llama-3-8b-chat-fp8"
LLAMA_3_8B_CHAT_FP4 = "llama-3-8b-chat-fp4"
MIXTRAL_8X7B_INSTRUCT = "mixtral-8x7b-instruct"
MIXTRAL_8X7B_INSTRUCT_FP8 = "mixtral-8x7b-instruct-fp8"


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
    choices=["text", "tools", "image", "audio", "video"],
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

    def __init__(
        self,
        model: str,
        display_name: Optional[str] = None,
        peft: Optional[str] = None,
        **kwargs,
    ):
        self.args = {
            "format": "none",
            **kwargs,
        }
        if model:
            self.args["model"] = model
        if display_name:
            self.args["display_name"] = display_name
        if peft:
            self.args["peft"] = peft

    async def run(self, pass_argv: List[str], spread: float) -> asyncio.Task:
        if spread:
            await asyncio.sleep(spread)
        full_argv = _dict_to_argv(self.args) + pass_argv
        return await llm_benchmark.run(full_argv)


class _CerebrasLlm(_Llm):
    """See https://docs.cerebras.ai/en/latest/wsc/Model-zoo/MZ-overview.html#list-of-models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "cerebras.ai/" + (display_model or model),
            api_key=os.getenv("CEREBRAS_API_KEY"),
            base_url="https://api.cerebras.ai/v1",
        )


class _CloudflareLlm(_Llm):
    """See https://developers.cloudflare.com/workers-ai/models/"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "cloudflare.com/" + (display_model or model),
        )


class _DeepInfraLlm(_Llm):
    """See https://deepinfra.com/models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "deepinfra.com/" + (display_model or model),
            api_key=os.getenv("DEEPINFRA_API_TOKEN"),
            base_url="https://api.deepinfra.com/v1/openai",
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


class _MistralLlm(_Llm):
    """See https://docs.mistral.ai/getting-started/models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "mistral.ai/" + (display_model or model),
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1",
        )


class _NvidiaLlm(_Llm):
    """See https://build.nvidia.com/explore/discover"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "nvidia.com/" + (display_model or model),
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
        )


class _OvhLlm(_Llm):
    """See https://llama-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/doc"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            "",
            "cloud.ovh.net/" + display_model,
            base_url=f"https://{model}.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
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


class _UltravoxLlm(_Llm):
    """See https://docs.ultravox.ai/docs/models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "ultravox.ai/" + (display_model or model),
            api_key=os.getenv("ULTRAVOX_API_KEY"),
            base_url="https://ultravox.api.fixie.ai/v1",
        )


def _text_models():
    AZURE_EASTUS2_OPENAI_API_KEY = os.getenv("AZURE_EASTUS2_OPENAI_API_KEY")
    return [
        # GPT-4o
        _Llm(GPT_4O_REALTIME_PREVIEW),
        _Llm(GPT_4O),
        _Llm(
            GPT_4O,
            api_key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        _Llm(GPT_4O, base_url="https://fixie-westus.openai.azure.com"),
        _Llm(
            GPT_4O,
            api_key=os.getenv("AZURE_NCENTRALUS_OPENAI_API_KEY"),
            base_url="https://fixie-centralus.openai.azure.com",
        ),
        _Llm(GPT_4O_MINI),
        # GPT-4 Turbo
        _Llm(GPT_4_TURBO),
        # GPT-4 Turbo Previews
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
        _Llm("claude-3-5-sonnet-20240620"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("claude-3-haiku-20240307"),
        # Cohere
        _Llm("command-r-plus"),
        _Llm("command-r"),
        _Llm("command-light"),
        # Gemini
        _Llm("gemini-pro"),
        _Llm(GEMINI_1_5_PRO),
        _Llm(GEMINI_1_5_FLASH),
        # Mistral
        _MistralLlm("mistral-large-latest", "mistral-large"),
        _MistralLlm("open-mistral-nemo", "mistral-nemo"),
        # Mistral 8x7b
        _DatabricksLlm("databricks-mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT),
        _DeepInfraLlm("mistralai/Mixtral-8x7B-Instruct-v0.1", MIXTRAL_8X7B_INSTRUCT),
        _FireworksLlm(
            "accounts/fireworks/models/mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT_FP8
        ),
        _FireworksLlm(
            "accounts/fireworks/models/mixtral-8x7b-instruct-hf", MIXTRAL_8X7B_INSTRUCT
        ),
        _GroqLlm("mixtral-8x7b-32768", MIXTRAL_8X7B_INSTRUCT_FP8),
        _NvidiaLlm("mistralai/mixtral-8x7b-instruct-v0.1-turbo", MIXTRAL_8X7B_INSTRUCT_FP8),
        _TogetherLlm("mistralai/Mixtral-8x7B-Instruct-v0.1", MIXTRAL_8X7B_INSTRUCT),
        # Llama 3.1 405b
        _DatabricksLlm("databricks-meta-llama-3.1-405b-instruct", LLAMA_31_405B_CHAT),
        _DeepInfraLlm(
            "meta-llama/Meta-Llama-3.1-405B-Instruct", LLAMA_31_405B_CHAT_FP8
        ),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3p1-405b-instruct", LLAMA_31_405B_CHAT_FP8
        ),
        _GroqLlm("llama-3.1-405b-reasoning", LLAMA_31_405B_CHAT_FP8),
        _NvidiaLlm("meta/llama-3.1-405b-instruct-turbo", LLAMA_31_405B_CHAT_FP8),
        _TogetherLlm(
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", LLAMA_31_405B_CHAT_FP8
        ),
        # _OvhLlm("llama-3p1-405b-instruct", LLAMA_31_405B_CHAT),
        # Llama 3.1 70b
        _CerebrasLlm("llama3.1-70b", LLAMA_31_70B_CHAT),
        _CloudflareLlm("@cf/meta/llama-3.1-70b-preview", LLAMA_31_70B_CHAT),
        # _DatabricksLlm("databricks-meta-llama-3.1-70b-instruct", LLAMA_31_70B_CHAT),
        _DeepInfraLlm("meta-llama/Meta-Llama-3.1-70B-Instruct", LLAMA_31_70B_CHAT),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3p1-70b-instruct", LLAMA_31_70B_CHAT_FP8
        ),
        _GroqLlm("llama-3.1-70b-versatile", LLAMA_31_70B_CHAT_FP8),
        _NvidiaLlm("meta/llama-3.1-70b-instruct-turbo", LLAMA_31_70B_CHAT_FP8),
        _PerplexityLlm("llama-3.1-70b-instruct", LLAMA_31_70B_CHAT),
        _TogetherLlm(
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", LLAMA_31_70B_CHAT_FP8
        ),
        # _OvhLlm("llama-3p1-8b-instruct", LLAMA_31_8B_CHAT),
        # Llama 3.1 8b
        _CerebrasLlm("llama3.1-8b", LLAMA_31_8B_CHAT),
        _CloudflareLlm("@cf/meta/llama-3.1-8b-preview", LLAMA_31_8B_CHAT),
        # _DatabricksLlm("databricks-meta-llama-3.1-8b-instruct", LLAMA_31_8B_CHAT),
        _DeepInfraLlm("meta-llama/Meta-Llama-3.1-8B-Instruct", LLAMA_31_8B_CHAT),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3p1-8b-instruct", LLAMA_31_8B_CHAT_FP8
        ),
        _GroqLlm("llama-3.1-8b-instant", LLAMA_31_8B_CHAT_FP8),
        _NvidiaLlm("meta/llama-3.1-8b-instruct-turbo", LLAMA_31_8B_CHAT_FP8),
        _PerplexityLlm("llama-3.1-8b-instruct", LLAMA_31_8B_CHAT),
        _TogetherLlm(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", LLAMA_31_8B_CHAT_FP8
        ),
        # _OvhLlm("llama-3p1-70b-instruct", LLAMA_31_70B_CHAT),
        # Llama 3 70b
        _DatabricksLlm("databricks-meta-llama-3-70b-instruct", LLAMA_3_70B_CHAT),
        _DeepInfraLlm("meta-llama/Meta-Llama-3-70B-Instruct", LLAMA_3_70B_CHAT),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3-70b-instruct", LLAMA_3_70B_CHAT_FP8
        ),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3-70b-instruct-hf", LLAMA_3_70B_CHAT
        ),
        _GroqLlm("llama3-70b-8192", LLAMA_3_70B_CHAT_FP8),
        _TogetherLlm("meta-llama/Llama-3-70b-chat-hf", LLAMA_3_70B_CHAT),
        _TogetherLlm(
            "meta-llama/Meta-Llama-3-70B-Instruct-Turbo", LLAMA_3_70B_CHAT_FP8
        ),
        _TogetherLlm("meta-llama/Meta-Llama-3-70B-Instruct-Lite", LLAMA_3_70B_CHAT_FP4),
        _OvhLlm("llama-3-70b-instruct", LLAMA_3_70B_CHAT),
        # Finetunes on Llama 3 70b
        _FireworksLlm(
            "accounts/fixie/models/1b68538a063a49e2ae4513d4ef186e9a",
            LLAMA_3_70B_CHAT + "-lora-1b68",
        ),
        # Llama 3 8b
        _CloudflareLlm("@cf/meta/llama-3-8b-instruct", LLAMA_3_8B_CHAT),
        _DeepInfraLlm("meta-llama/Meta-Llama-3-8B-Instruct", LLAMA_3_8B_CHAT),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3-8b-instruct", LLAMA_3_8B_CHAT_FP8
        ),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3-8b-instruct-hf", LLAMA_3_8B_CHAT
        ),
        _GroqLlm("llama3-8b-8192", LLAMA_3_8B_CHAT_FP8),
        _TogetherLlm("meta-llama/Llama-3-8b-chat-hf", LLAMA_3_8B_CHAT),
        _TogetherLlm("meta-llama/Meta-Llama-3-8B-Instruct-Turbo", LLAMA_3_8B_CHAT_FP8),
        _TogetherLlm("meta-llama/Meta-Llama-3-8B-Instruct-Lite", LLAMA_3_8B_CHAT_FP4),
        _OvhLlm("llama-3-8b-instruct", LLAMA_3_8B_CHAT),
        # Fine-tunes on Llama 3 8b
        _FireworksLlm(
            "accounts/fixie/models/8ab03ea85d2a4b9da659ce63db36a9b1",
            LLAMA_3_8B_CHAT + "-lora-8ab0",
        ),
    ]


def _tools_models():
    return [
        _Llm(GPT_4O),
        _Llm(GPT_4O_MINI),
        _Llm(GPT_4_TURBO),
        _Llm(GPT_4O, GPT_4O + "-strict", strict=None),
        _Llm(GPT_4O_MINI, GPT_4O_MINI + "-strict", strict=None),
        _Llm(GPT_4_TURBO, GPT_4_TURBO + "-strict", strict=None),
        _Llm("claude-3-opus-20240229"),
        _Llm("claude-3-5-sonnet-20240620"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("claude-3-haiku-20240307"),
        _Llm(GEMINI_1_5_PRO),
        _Llm(GEMINI_1_5_FLASH),
        _FireworksLlm("accounts/fireworks/models/firefunction-v2", "firefunction-v2"),
        # _FireworksLlm(
        #    "accounts/fireworks/models/llama-v3p1-405b-instruct", LLAMA_31_405B_CHAT_FP8
        # ), returns "FUNCTION" and the call as text
        _GroqLlm("llama-3.1-405b-reasoning", LLAMA_31_405B_CHAT_FP8),
        _GroqLlm("llama-3.1-70b-versatile", LLAMA_31_70B_CHAT_FP8),
        _GroqLlm("llama-3.1-8b-instant", LLAMA_31_8B_CHAT_FP8),
        _GroqLlm("llama3-groq-70b-8192-tool-use-preview"),
        _GroqLlm("llama3-groq-8b-8192-tool-use-preview"),
    ]


def _image_models():
    return [
        _Llm(GPT_4O),
        _Llm(GPT_4O_MINI),
        _Llm(GPT_4_TURBO),
        _Llm("gpt-4-vision-preview", base_url="https://fixie-westus.openai.azure.com"),
        _Llm("claude-3-opus-20240229"),
        _Llm("claude-3-5-sonnet-20240620"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("gemini-pro-vision"),
        _Llm(GEMINI_1_5_PRO),
        _Llm(GEMINI_1_5_FLASH),
        _FireworksLlm(
            "accounts/fireworks/models/phi-3-vision-128k-instruct", "phi-3-vision"
        ),
        _MistralLlm("pixtral-latest", "pixtral"),
    ]


def _audio_models():
    return [
        _Llm(GPT_4O_REALTIME_PREVIEW),
        _Llm(GEMINI_1_5_PRO),
        _Llm(GEMINI_1_5_FLASH),
        _UltravoxLlm("fixie-ai/ultravox-v0.4", "ultravox-v0.4-8b"),
        _UltravoxLlm("fixie-ai/ultravox-70B", "ultravox-v0.4-70b"),
        _Llm(
            "fixie-ai/ultravox-v0.2",
            "baseten.co/ultravox-v0.2",
            base_url="https://bridge.baseten.co/5wovovzq/v1/direct",
            api_key=os.getenv("BASETEN_API_KEY"),
        ),
    ]


def _video_models():
    return [
        # _Llm(GPT_4O),
        _Llm(GEMINI_1_5_PRO),
        _Llm(GEMINI_1_5_FLASH),
    ]


def _get_models(mode: str, filter: Optional[str] = None):
    mode_map = {
        "text": _text_models,
        "tools": _tools_models,
        "image": _image_models,
        "audio": _audio_models,
        "video": _video_models,
    }
    if mode not in mode_map:
        raise ValueError(f"Unknown mode {mode}")
    models = mode_map[mode]()
    return [
        m
        for m in models
        if not filter
        or filter in (m.args.get("display_name") or m.args["model"]).lower()
    ]


def _get_prompt(mode: str) -> List[str]:
    if mode == "text":
        return ["@media/text/llama31.md"]
    elif mode == "tools":
        return [
            "I have a flight booked for July 14, 2024, and the flight number is AA100. Please check its status for me.",
            "--tool",
            "media/tools/flights.json",
        ]
    elif mode == "image":
        return [
            "Based on the image, explain what will happen next.",
            "--file",
            "media/image/inception.jpeg",
        ]
    elif mode == "audio":
        return [
            "Listen and respond to the following:",
            "--file",
            "media/audio/boolq.wav",
        ]
    elif mode == "video":
        return [
            "What color is the logo on the screen and how does it relate to what the actor is saying?",
            "--file",
            "media/video/psa.webm",
        ]
    raise ValueError(f"Unknown mode {mode}")


@dataclasses.dataclass
class _Response(dataclasses_json.DataClassJsonMixin):
    time: str
    duration: str
    region: str
    cmd: str
    results: List[llm_request.ApiMetrics]


def _format_response(
    response: _Response, format: str, dlen: int = 0
) -> Tuple[str, str]:
    if format == "json":
        return response.to_json(indent=2), "application/json"
    else:
        s = (
            "| Provider/Model                             | TTR  | TTFT | TPS  | ITk  | OTk | ITim | OTim | Total |"
            f" {'Response':{dlen}.{dlen}} |\n"
            "| :----------------------------------------- | ---: | ---: | ---: | ---: | --: | ---: | ---: | ----: |"
            f" {':--':-<{dlen}.{dlen}} |\n"
        )

        for r in response.results:
            ttr = r.ttr or 0.0
            ttft = r.ttft or 0.0
            tps = r.tps or 0.0
            in_tokens = r.input_tokens or 0
            out_tokens = r.output_tokens or 0
            in_time = r.provider_input_time or 0
            out_time = (
                r.provider_output_time or r.total_time - r.ttft
                if out_tokens
                else r.ttft
            )
            total_time = r.total_time or 0.0
            output = (r.error or r.output).strip().replace("\n", "\\n")
            s += (
                f"| {r.model[:42]:42} | {ttr:4.2f} | {ttft:4.2f} | {tps:4.0f} "
                f"| {in_tokens:4} | {out_tokens:3} | {in_time:4.2f} | {out_time:4.2f} "
                f"| {total_time:5.2f} | {output:{dlen}.{dlen}} |\n"
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
