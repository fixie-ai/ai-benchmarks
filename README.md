# ai-benchmarks

This repo contains a handful of utilities for benchmarking the response latency of popular AI services, including:
- OpenAI
- Anthropic
- ElevenLabs

## Initial setup

```
pip install -r requirements.txt
```

## Running benchmarks

To run a benchmark, first set the appropriate environment variable (e.g., OPENAI_API_KEY, ELEVEN_API_KEY) etc, and then run 
the appropriate benchmark script, e.g., 

```
python llm_benchmark.py -m gpt-3.5-turbo "Write me a haiku."
```

or

```
python elevenlabs_stream_benchmark.py "Haikus I find tricky, With a 5-7-5 count, But I'll give it a go"
```

