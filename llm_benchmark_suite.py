import argparse
import asyncio
import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import llm_benchmark

DEFAULT_RESPONSE_LENGTH = 64

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
    "--response-length",
    "-l",
    type=int,
    default=DEFAULT_RESPONSE_LENGTH,
    help="Amount of the response to show",
)


def _dict_to_argv(d: Dict[str, str]) -> List[str]:
    return [f"--{k.replace('_', '-')}={v}" for k, v in d.items()]


class _Llm:
    """
    We maintain a dict of params for the llm, as well as any
    command-line flags that we didn't already handle. We'll
    turn this into a single command line for llm_benchmark.run
    to consume, which allows us to reuse the parsing logic
    from that script, rather than having to duplicate it here.
    """

    def __init__(self, model, **kwargs):
        self.pass_argv = []
        self.args = {"model": model, **kwargs, "format": "none"}

    def apply(self, pass_argv: List[str], **kwargs):
        self.pass_argv = pass_argv
        self.args.update(kwargs)
        return self

    def run(self):
        full_argv = _dict_to_argv(self.args) + self.pass_argv
        return asyncio.create_task(llm_benchmark.run(full_argv))


def _text_models():
    AZURE_EASTUS2_OPENAI_API_KEY = os.getenv("AZURE_EASTUS2_OPENAI_API_KEY")
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OCTOML_API_KEY = os.getenv("OCTOML_API_KEY")
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    return [
        _Llm("gpt-4-turbo"),
        _Llm("gpt-4-0125-preview"),
        _Llm(
            "gpt-4-0125-preview",
            api_key=os.getenv("AZURE_SCENTRALUS_OPENAI_API_KEY"),
            base_url="https://fixie-scentralus.openai.azure.com",
        ),
        _Llm("gpt-4-1106-preview"),
        _Llm("gpt-4-1106-preview", base_url="https://fixie-westus.openai.azure.com"),
        _Llm(
            "gpt-4-1106-preview",
            api_key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        _Llm(
            "gpt-4-1106-preview",
            api_key=os.getenv("AZURE_FRCENTRAL_OPENAI_API_KEY"),
            base_url="https://fixie-frcentral.openai.azure.com",
        ),
        _Llm(
            "gpt-4-1106-preview",
            api_key=os.getenv("AZURE_SECENTRAL_OPENAI_API_KEY"),
            base_url="https://fixie-secentral.openai.azure.com",
        ),
        _Llm(
            "gpt-4-1106-preview",
            api_key=os.getenv("AZURE_UKSOUTH_OPENAI_API_KEY"),
            base_url="https://fixie-uksouth.openai.azure.com",
        ),
        _Llm("gpt-3.5-turbo-0125"),
        _Llm("gpt-3.5-turbo-1106"),
        _Llm("gpt-3.5-turbo-1106", base_url="https://fixie-westus.openai.azure.com"),
        _Llm(
            "gpt-3.5-turbo",
            api_key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        _Llm("claude-3-opus-20240229"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("claude-3-haiku-20240307"),
        _Llm("claude-2.1"),
        _Llm("claude-instant-1.2"),
        _Llm("command-r-plus"),
        _Llm("command-r"),
        _Llm("command-light"),
        _Llm("gemini-pro"),
        _Llm("gemini-1.5-pro-preview-0409"),
        _Llm(
            "",
            api_key=os.getenv("AZURE_EASTUS2_MISTRAL_API_KEY"),
            base_url="https://fixie-mistral-serverless.eastus2.inference.ai.azure.com/v1",
        ),
        _Llm(
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        _Llm(
            "mixtral-8x7b-32768",
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        ),
        _Llm(
            "mixtral-8x7b-instruct",
            api_key=OCTOML_API_KEY,
            base_url="https://text.octoai.run/v1",
        ),
        _Llm(
            "sonar-medium-chat",
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai",
        ),
        _Llm(
            "",
            api_key=os.getenv("AZURE_WESTUS3_LLAMA2_API_KEY"),
            base_url="https://fixie-llama-2-70b-serverless.westus3.inference.ai.azure.com/v1",
        ),
        _Llm(
            "",
            api_key=os.getenv("AZURE_EASTUS2_LLAMA2_API_KEY"),
            base_url="https://fixie-llama-2-70b-serverless.eastus2.inference.ai.azure.com/v1",
        ),
        _Llm(
            "accounts/fireworks/models/llama-v2-70b-chat",
            api_key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        _Llm(
            "llama2-70b-4096",
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        ),
        _Llm(
            "llama-2-70b-chat-fp16",
            api_key=OCTOML_API_KEY,
            base_url="https://text.octoai.run/v1",
        ),
        _Llm(
            "pplx-70b-chat",
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai",
        ),
        _Llm("togethercomputer/llama-2-70b-chat"),
        _Llm(
            "accounts/fireworks/models/llama-v2-13b-chat",
            api_key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        _Llm(
            "llama-2-13b-chat-fp16",
            api_key=OCTOML_API_KEY,
            base_url="https://text.octoai.run/v1",
        ),
        _Llm("togethercomputer/llama-2-13b-chat"),
        _Llm(
            "accounts/fireworks/models/llama-v2-7b-chat",
            api_key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        _Llm(
            "pplx-7b-chat",
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai",
        ),
        _Llm("togethercomputer/llama-2-7b-chat"),
        _Llm("@cf/meta/llama-2-7b-chat-fp16"),
        _Llm("@cf/meta/llama-2-7b-chat-int8"),
        _Llm("Neets-7B"),
    ]


def _image_models():
    return [
        _Llm("gpt-4-turbo"),
        _Llm("gpt-4-vision-preview", base_url="https://fixie-westus.openai.azure.com"),
        _Llm("claude-3-opus-20240229"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("gemini-pro-vision"),
        _Llm("gemini-1.5-pro-preview-0409"),
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
    return [m for m in models if not filter or filter in m.args["model"]]


def _format_response(
    format: str,
    response_length: int,
    start_time: datetime.datetime,
    region: str,
    argv: List[str],
    results: List[Dict[str, Any]],
) -> Tuple[str, str]:
    time_str = start_time.isoformat()
    cmd = " ".join(argv)
    if format == "json":
        obj = {
            "time": time_str,
            "region": region,
            "cmd": cmd,
            "results": results,
        }
        return json.dumps(obj, indent=2), "application/json"
    else:
        s = (
            "| Provider/Model                             | TTR  | TTFT | TPS | Tok | Total |"
            f" {'Response':{response_length}.{response_length}} |\n"
            "| :----------------------------------------- | ---: | ---: | --: | --: | ----: |"
            f" {':--':-<{response_length}.{response_length}} |\n"
        )

        for r in results:
            output = r["output"].replace("\n", "\\n").strip()
            s += (
                f"| {r['model']:42} | {r['ttr']:4.2f} | {r['ttft']:4.2f} | "
                f"{r['tps']:3.0f} | {r['num_tokens']:3} | {r['total_time']:5.2f} | "
                f"{output:{response_length}.{response_length}} |\n"
            )

        s += f"\ntime: {time_str}, region: {region}, cmd: {cmd}\n"
        return s, "text/markdown"


async def run(
    mode: Optional[str] = "text",
    format: Optional[str] = "text",
    response_length: Optional[int] = DEFAULT_RESPONSE_LENGTH,
    filter: Optional[str] = None,
    pass_argv: Optional[List[str]] = None,
    **kwargs,
):
    """
    This function is invoked either from the webapp or the main function below.
    When invoked from the webapp, the arguments are passed as kwargs.
    When invoked from the main function, the arguments are passed as a list of flags.
    We'll give both to the _Llm.run function, which will turn them back into a
    single list of flags for consumption by the llm_benchmark.run function.
    """
    models = _get_models(mode, filter)
    tasks = []
    start_time = datetime.datetime.now()
    for m in models:
        m.apply(pass_argv or [], **kwargs)
        tasks.append(m.run())
    await asyncio.gather(*tasks)
    results = [t.result() for t in tasks if t.result() is not None]
    region = os.getenv("FLY_REGION", "local")
    argv = _dict_to_argv(kwargs) + (pass_argv or [])
    return _format_response(format, response_length, start_time, region, argv, results)


async def main(args: argparse.Namespace, pass_argv: List[str]):
    text, _ = await run(
        args.mode, args.format, args.response_length, args.filter, pass_argv
    )
    print(text)


if __name__ == "__main__":
    args, unk_args = parser.parse_known_args()
    asyncio.run(main(args, unk_args))
