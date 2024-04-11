import argparse
import asyncio
import json
import llm_benchmark
import logging
import os
import sys
import time

from typing import Dict, List

parser = argparse.ArgumentParser()
parser.add_argument("--json", action="store_true", help="Output results as JSON")
parser.add_argument(
    "--mode",
    "-m",
    choices=["text", "vision"],
    default="text",
    help="Mode to run benchmarks for",
)
args, unknown_args = parser.parse_known_args()


async def run(model, key=None, base_url=None):
    run_args = ["--format=none", f"--model={model}"]
    if key:
        run_args.append(f"--api-key={key}")
    if base_url:
        run_args.append(f"--base-url={base_url}")
    run_args.extend(unknown_args)
    try:
        return await llm_benchmark.run(run_args)
    except Exception:
        logging.exception(f"Error running {model}")
        return None


def text_tasks():
    AZURE_EASTUS2_OPENAI_API_KEY = os.getenv("AZURE_EASTUS2_OPENAI_API_KEY")
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    GCLOUD_ACCESS_TOKEN = os.popen("gcloud auth print-access-token").read().strip()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OCTOML_API_KEY = os.getenv("OCTOML_API_KEY")
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    return [
        run("gpt-4-turbo"),
        run("gpt-4-0125-preview"),
        run(
            "gpt-4-0125-preview",
            key=os.getenv("AZURE_SCENTRALUS_OPENAI_API_KEY"),
            base_url="https://fixie-scentralus.openai.azure.com",
        ),
        run("gpt-4-1106-preview"),
        run("gpt-4-1106-preview", base_url="https://fixie-westus.openai.azure.com"),
        run(
            "gpt-4-1106-preview",
            key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        run(
            "gpt-4-1106-preview",
            key=os.getenv("AZURE_FRCENTRAL_OPENAI_API_KEY"),
            base_url="https://fixie-frcentral.openai.azure.com",
        ),
        run(
            "gpt-4-1106-preview",
            key=os.getenv("AZURE_SECENTRAL_OPENAI_API_KEY"),
            base_url="https://fixie-secentral.openai.azure.com",
        ),
        run(
            "gpt-4-1106-preview",
            key=os.getenv("AZURE_UKSOUTH_OPENAI_API_KEY"),
            base_url="https://fixie-uksouth.openai.azure.com",
        ),
        run("gpt-3.5-turbo-0125"),
        run("gpt-3.5-turbo-1106"),
        run("gpt-3.5-turbo-1106", base_url="https://fixie-westus.openai.azure.com"),
        run(
            "gpt-3.5-turbo",
            key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        run("claude-3-opus-20240229"),
        run("claude-3-sonnet-20240229"),
        run("claude-3-haiku-20240307"),
        run("claude-2.1"),
        run("claude-instant-1.2"),
        run("command-r-plus"),
        run("command-r"),
        run("command-light"),
        run("gemini-pro", key=GCLOUD_ACCESS_TOKEN),
        run("gemini-1.5-pro-preview-0409", key=GCLOUD_ACCESS_TOKEN),
        run(
            "",
            key=os.getenv("AZURE_EASTUS2_MISTRAL_API_KEY"),
            base_url="https://fixie-mistral-serverless.eastus2.inference.ai.azure.com/v1",
        ),
        run(
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        run(
            "mixtral-8x7b-32768",
            key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        ),
        run(
            "mixtral-8x7b-instruct",
            key=OCTOML_API_KEY,
            base_url="https://text.octoai.run/v1",
        ),
        run(
            "sonar-medium-chat",
            key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai",
        ),
        run(
            "",
            key=os.getenv("AZURE_WESTUS3_LLAMA2_API_KEY"),
            base_url="https://fixie-llama-2-70b-serverless.westus3.inference.ai.azure.com/v1",
        ),
        run(
            "",
            key=os.getenv("AZURE_EASTUS2_LLAMA2_API_KEY"),
            base_url="https://fixie-llama-2-70b-serverless.eastus2.inference.ai.azure.com/v1",
        ),
        run(
            "accounts/fireworks/models/llama-v2-70b-chat",
            key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        run(
            "llama2-70b-4096",
            key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        ),
        run(
            "llama-2-70b-chat-fp16",
            key=OCTOML_API_KEY,
            base_url="https://text.octoai.run/v1",
        ),
        run(
            "pplx-70b-chat",
            key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai",
        ),
        run("togethercomputer/llama-2-70b-chat"),
        run(
            "accounts/fireworks/models/llama-v2-13b-chat",
            key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        run(
            "llama-2-13b-chat-fp16",
            key=OCTOML_API_KEY,
            base_url="https://text.octoai.run/v1",
        ),
        run("togethercomputer/llama-2-13b-chat"),
        run(
            "accounts/fireworks/models/llama-v2-7b-chat",
            key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
        ),
        run(
            "pplx-7b-chat", key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai"
        ),
        run("togethercomputer/llama-2-7b-chat"),
        run("@cf/meta/llama-2-7b-chat-fp16"),
        run("@cf/meta/llama-2-7b-chat-int8"),
        run("Neets-7B"),
    ]


def vision_tasks():
    GCLOUD_ACCESS_TOKEN = os.popen("gcloud auth print-access-token").read().strip()
    return [
        run("gpt-4-turbo"),
        run("gpt-4-vision-preview", base_url="https://fixie-westus.openai.azure.com"),
        run("claude-3-opus-20240229"),
        run("claude-3-sonnet-20240229"),
        run("gemini-pro-vision", key=GCLOUD_ACCESS_TOKEN),
    ]


def get_tasks(mode: str):
    if mode == "text":
        return text_tasks()
    elif mode == "vision":
        return vision_tasks()
    else:
        raise ValueError(f"Unknown mode {mode}")


async def print_results(results: List[Dict]):
    print(
        "| Provider/Model                             | TTR  | TTFT | TPS | Tok | Total |"
        " Response                                                         |\n"
        "| :----------------------------------------- | :--- | :--- | :-- | :-- | :---- |"
        " :--------------------------------------------------------------- |"
    )
    for r in results:
        output = r["output"].replace("\n", "\\n").strip()[:64]
        print(
            f"| {r['model']:<42} | {r['ttr']:4.2f} | {r['ttft']:4.2f} | "
            f"{r['tps']:3.0f} | {r['num_tokens']:3} | {r['total_time']:5.2f} | {output} |"
        )


async def main():
    start_time = time.time()
    tasks = get_tasks(args.mode)
    results = [r for r in await asyncio.gather(*tasks) if r is not None]
    if args.json:
        out = {"time": start_time(), "cmd": sys.argv, "results": results}
        print(json.dumps(out))
    else:
        await print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
