# ai-benchmarks

This repo contains a handful of utilities for benchmarking the response latency of popular AI services, including:

Large Language Models (LLMs):

- OpenAI GPT-3.5, GPT-4 (from OpenAI or Azure OpenAI service)
- Anthropic Claude 3, Claude 2, Claude Instant
- Google Gemini Pro and PaLM 2 Bison
- Llama2 and 3 from several different providers, including
  - Anyscale
  - Azure
  - Cloudflare
  - Groq
  - OctoAI
  - Perplexity
  - Together
- Mixtral 8x7B from several different providers, including
  - Anyscale
  - Azure
  - Groq
  - OctoAI
  - Perplexity

Embedding Models:

- Ada-002
- Cohere

Text-to-Speech Models (TTS):

- ElevenLabs
- PlayHT

## Leaderboard

Snapshot below, click it to jump to the latest spreadsheet.
[![Screenshot 2024-03-05 at 4 08 20 PM](https://github.com/fixie-ai/ai-benchmarks/assets/1821693/97651011-fc8e-4481-bac9-cba0927aa485)](https://docs.google.com/spreadsheets/d/e/2PACX-1vTPttBIJ676Ke5eKXh8EoOe9XrMZ1kgVh-hvuO-LP41GTNIbsHwx1bcb_SsoB3BTDZLNeMspqLQMXSS/pubhtml?gid=0&single=true)

### Test methodology

- Tests are run from a Google Cloud console in us-west1.
- Input requests are short, typically a single message (~20 tokens), and typically ask for a brief output response.
- Max output tokens is set to 100, to avoid distortion of TPS values from long outputs.
- A warmup connection is made to remove any connection setup latency.
- The TTFT clock starts when the HTTP request is made and stops when the first token result is received in the response stream.
- For each provider, three separate inferences are done, and the best result is kept (to remove any outliers due to queuing etc).
- A best result is selected on 3 different days, and the median of these values is displayed.

## Initial setup

This repo uses [Poetry](https://python-poetry.org/) for dependency management. To install the dependencies, run:

```
pip install poetry
poetry install --sync
```

## Running benchmarks

To run a benchmark, first set the appropriate environment variable (e.g., OPENAI_API_KEY, ELEVEN_API_KEY) etc, and then run
the appropriate benchmark script.

### LLM benchmarks

To generate LLM benchmarks, use the `llm_benchmark.py` script. For most providers, you can just pass the model name and the script will figure out what API endpoint to invoke. e.g.,

```
poetry run python llm_benchmark.py -m gpt-3.5-turbo "Write me a haiku."
```

However, when invoking generic models like Llama2, you'll need to pass in the base_url and api_key via the -b and -k parameters, e.g.,

```
poetry run python llm_benchmark.py -k $OCTOML_API_KEY -b https://text.octoai.run/v1 \
       -m llama-2-70b-chat-fp16 "Write me a haiku."
```

Similarly, when invoking Azure OpenAI, you'll need to specify your Azure API key and the base URL of your Azure deployment, e.g.,

```
poetry run python llm_benchmark.py -b https://fixie-westus.openai.azure.com \
       -m gpt-4-1106-preview "Write me a haiku."
```

See [this script](https://github.com/fixie-ai/ai-benchmarks/blob/main/llm_benchmark_suite.sh) for more examples of how to invoke various providers.

#### Options

```
usage: llm_benchmark.py [-h] [--model MODEL] [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS] [--base-url BASE_URL]
                        [--api-key API_KEY] [--no-warmup] [--num-requests NUM_REQUESTS] [--print] [--verbose]
                        [prompt]

positional arguments:
  prompt                                       Prompt to send to the API

optional arguments:
  -h, --help                                   show this help message and exit
  --model MODEL, -m MODEL                      Model to benchmark
  --temperature TEMPERATURE, -t TEMPERATURE    Temperature for the response
  --max-tokens, -T MAX_TOKEN                   Max tokens for the response
  --base-url BASE_URL, -b BASE_URL             Base URL for the LLM API endpoint
  --api-key API_KEY, -k API_KEY                API key for the LLM API endpoint
  --no-warmup                                  Don't do a warmup call to the API
  --num-requests NUM_REQUESTS, -n NUM_REQUESTS Number of requests to make
  --print, -p                                  Print the response
  --verbose, -v                                Print verbose output
```

#### Output

By default a summary of the requests is printed:

```
Latency saved: 0.01 seconds                <---- Difference between first response time and fastest reponse time
Optimized response time: 0.14 seconds      <---- fastest(http_response_time - http_start_time) of N requests
Median response time: 0.15 seconds         <---- median(http_response_time - http_start_time) of N requests
Time to first token: 0.34 seconds          <---- first_token_time - http_start_time
Tokens: 147 (211 tokens/sec)               <---- num_generated_tokens / (last_token_time - first_token_time)
Total time: 1.03 seconds                   <---- last_token_time - http_start_time
```

You can specify -p to print the output of the LLM, or -v to see detailed timing for each request.

### TTS benchmarks

To generate TTS benchmarks, there are various scripts for the individual providers, e.g.,

```
python elevenlabs_stream_benchmark.py "Haikus I find tricky, With a 5-7-5 count, But I'll give it a go"
```

#### Playing audio

By default, only timing information for TTS is emitted. Follow the steps below to actually play out the received audio.

First, install `mpv` via

```
brew install mpv
```

Then, just pass the -p argument when generating text, e.g.,

```
python playht_benchmark.py -p "Well, basically I have intuition."
```

You can use the -v parameter to select which voice to use for generation.
