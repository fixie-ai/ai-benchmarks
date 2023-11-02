# ai-benchmarks

This repo contains a handful of utilities for benchmarking the response latency of popular AI services, including:

Large Language Models (LLMs):
- OpenAI (chat and embedding models)
- Anthropic (chat models)

Text-to-Speech Models (TTS):
- ElevenLabs
- PlayHT

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

## Playing audio

First, install mpv via

```
brew install mpv
```

Then, just pass the -p argument when generating text, e.g., 

```
python playht_benchmark.py -p "Well, basically I have intuition."
```

You can use the -v parameter to select which voice to use for generation.
