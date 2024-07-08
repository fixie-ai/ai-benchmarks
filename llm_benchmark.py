#!/usr/bin/env python
import argparse
import asyncio
from typing import List

import aiohttp

import llm_request

DEFAULT_PROMPT = "Write a nonet about a sunset."
DEFAULT_MAX_TOKENS = 100
DEFAULT_NUM_REQUESTS = 4

FMT_DEFAULT = "default"
FMT_MINIMAL = "minimal"
FMT_JSON = "json"
FMT_NONE = "none"

parser = argparse.ArgumentParser()
parser.add_argument(
    "prompt",
    type=str,
    nargs="?",
    default=DEFAULT_PROMPT,
    help="Prompt to send to the API",
)
parser.add_argument(
    "--file",
    "-f",
    type=str,
    action="append",
    help="Multimedia file(s) to include with the prompt",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="",
    help="Model to benchmark",
)
parser.add_argument(
    "--display-name",
    "-N",
    type=str,
    help="Display name for the model",
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
    "-T",
    type=int,
    default=DEFAULT_MAX_TOKENS,
    help="Max tokens for the response",
)
parser.add_argument(
    "--detail",
    "-d",
    help="Image detail level to use, low or high",
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
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    dest="verbose",
    help="Print verbose output",
)
parser.add_argument(
    "--format",
    "-F",
    type=str,
    default=FMT_DEFAULT,
)
parser.add_argument(
    "--timeout",
    type=float,
    default=30.0,
    help="Timeout for the API call",
)


class LlmTraceConfig(aiohttp.TraceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.on_request_start.append(self._on_request_start_func)
        self.on_connection_create_end.append(self._on_connection_create_end_func)
        self.on_connection_reuseconn.append(self._on_connection_reuseconn_func)

    async def _on_request_start_func(self, session, trace_ctx, params):
        self._url = params.url

    async def _on_connection_create_end_func(self, session, trace_ctx, params):
        print(f"Created connection for {self._url}")

    async def _on_connection_reuseconn_func(self, session, trace_ctx, params):
        print(f"Reused connection for {self._url}")


async def main(args: argparse.Namespace):
    if not args.model and not args.base_url:
        print("Either MODEL or BASE_URL must be specified")
        return None

    # Run the queries.
    files = [llm_request.InputFile.from_file(file) for file in args.file or []]
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    trace_configs = [LlmTraceConfig()] if args.verbose else []
    async with aiohttp.ClientSession(
        timeout=timeout, trace_configs=trace_configs, connector=aiohttp.TCPConnector()
    ) as session:
        init_ctx = llm_request.make_context(session, -1, args)
        contexts = [
            llm_request.make_context(session, i, args, args.prompt, files)
            for i in range(args.num_requests)
        ]
        chosen = None

        if args.warmup:
            # Do a warmup call to make sure the connection is ready,
            # and sleep it off to make sure it doesn't affect rate limits.
            if args.verbose:
                print("Making a warmup API call...")
            await init_ctx.run()
            await asyncio.sleep(1.0)

        def on_token(ctx: llm_request.ApiContext, token: str):
            nonlocal chosen
            if not chosen:
                chosen = ctx
                if args.format == FMT_DEFAULT:
                    ttft = chosen.metrics.ttft
                    print(f"Chosen API Call: {chosen.index} ({ttft:.2f}s)")
            if ctx == chosen:
                if args.print:
                    print(token, end="", flush=True)

        if args.format == FMT_DEFAULT:
            print(f"Racing {args.num_requests} API calls to {init_ctx.name}...")
        tasks = [asyncio.create_task(ctx.run(on_token)) for ctx in contexts]
        await asyncio.gather(*tasks)

    # Bail out if there were no successful API calls.
    task0_metrics = contexts[0].metrics
    if not chosen:
        if args.format == FMT_DEFAULT:
            print(f"No successful API calls for {init_ctx.name}")
        return task0_metrics

    # Print results.
    if args.print:
        print("\n")
    if args.verbose:
        for ctx in contexts:
            r = ctx.metrics
            if not r.error:
                print(f"API Call {ctx.index} TTFT: {r.ttft:.2f} seconds")
            else:
                print(f"API Call {ctx.index} Result: {r.error}")
        print("")

    r = chosen.metrics
    if args.format == FMT_DEFAULT:
        metrics = [ctx.metrics for ctx in contexts]
        latency_saved = task0_metrics.ttft - r.ttft
        metrics.sort(key=lambda x: x.ttft)
        med_index1 = (len(metrics) - 1) // 2
        med_index2 = len(metrics) // 2
        median_latency = (metrics[med_index1].ttft + metrics[med_index2].ttft) / 2
        print(f"Latency saved: {latency_saved:.2f} seconds")
        print(f"Optimized TTFT: {r.ttft:.2f} seconds")
        print(f"Median TTFT: {median_latency:.2f} seconds")
        if r.num_tokens:
            print(f"Tokens: {r.num_tokens} ({r.tps:.0f} tokens/sec)")
            print(f"Total time: {r.total_time:.2f} seconds")
    elif args.format == "minimal":
        assert r.output
        minimal_output = r.error or r.output.replace("\n", "\\n").strip()[:64]
        print(
            f"| {r.model:42} | {r.ttr:4.2f} | {r.ttft:4.2f} | {r.tps:3.0f} "
            f"| {r.num_tokens:3} | {r.total_time:5.2f} | {minimal_output} |"
        )
    elif args.format == "json":
        print(r.to_json(indent=2))
    return r


async def run(argv: List[str]):
    args = parser.parse_args(argv)
    return await main(args)


if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(main(args))
