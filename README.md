# ai-benchmarks

This repo contains a handful of utilities for benchmarking the response latency of popular AI services, including:

Large Language Models (LLMs):
- OpenAI GPT-3.5, GPT-4 (from OpenAI or Azure OpenAI service)
- Anthropic Claude 2, Claude Instant
- Google PaLM 2
- Llama2 7B/70B from several different providers, including
 - Cloudflare
 - OctoML
 - Perplexity
 - Together
- Neets-7B

Embedding Models:
- Ada-002
- Cohere

Text-to-Speech Models (TTS):
- ElevenLabs
- PlayHT

## Leaderboard
Snapshot below, click it to jump to the latest spreadsheet.
[![Screenshot 2023-12-11 at 4 14 19 PM](https://github.com/fixie-ai/ai-benchmarks/assets/1821693/4613403d-a944-4dbf-9752-792453c9d13a)](https://docs.google.com/spreadsheets/d/e/2PACX-1vTPttBIJ676Ke5eKXh8EoOe9XrMZ1kgVh-hvuO-LP41GTNIbsHwx1bcb_SsoB3BTDZLNeMspqLQMXSS/pubhtml?gid=0&single=true)



## Initial setup

```
pip install -r requirements.txt
```

## Running benchmarks

To run a benchmark, first set the appropriate environment variable (e.g., OPENAI_API_KEY, ELEVEN_API_KEY) etc, and then run 
the appropriate benchmark script.

### LLM benchmarks

To generate LLM benchmarks, use the `llm_benchmark.py` script. For most providers, you can just pass the model name and the script will figure out what API endpoint to invoke. e.g., 

```
python llm_benchmark.py -m gpt-3.5-turbo "Write me a haiku."
```

However, when invoking generic models like Llama2, you'll need to pass in the base_url and api_key via the -b and -k parameters, e.g., 

```
python llm_benchmark.py -k $OCTOML_API_KEY -b https://text.octoai.run/v1 \
       -m llama-2-70b-chat-fp16 "Write me a haiku."
```

Similarly, when invoking Azure OpenAI, you'll need to specify your Azure API key and the base URL of your Azure deployment, e.g., 

```
python llm_benchmark.py -b https://fixie-westus.openai.azure.com \
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
  --max-tokens MAX_TOKENS                      Max tokens for the response                        
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
