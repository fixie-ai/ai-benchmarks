echo "Provider/Model                                         | TTR  | TTFT | TPS  | Total | Tokens"
# gpt-4-turbo
python llm_benchmark.py --minimal -m gpt-4-0125-preview "$@"
python llm_benchmark.py --minimal -m gpt-4-1106-preview "$@"
python llm_benchmark.py --minimal -b https://fixie-westus.openai.azure.com -m gpt-4-1106-preview "$@"
python llm_benchmark.py --minimal -k $AZURE_EASTUS2_OPENAI_API_KEY -b https://fixie-openai-sub-with-gpt4.openai.azure.com -m gpt-4-1106-preview "$@"
# gpt-3.5-turbo
python llm_benchmark.py --minimal -m gpt-3.5-turbo-0125 "$@"
python llm_benchmark.py --minimal -m gpt-3.5-turbo-1106 "$@"
python llm_benchmark.py --minimal -b https://fixie-westus.openai.azure.com -m gpt-3.5-turbo-1106 "$@"
# No gpt-3.5-turbo-1106 in eastus2, so we use vanilla gpt-3.5-turbo here
python llm_benchmark.py --minimal -k $AZURE_EASTUS2_OPENAI_API_KEY -b https://fixie-openai-sub-with-gpt4.openai.azure.com -m gpt-3.5-turbo "$@"
# claude
python llm_benchmark.py --minimal -m claude-3-opus-20240229 "$@"
python llm_benchmark.py --minimal -m claude-3-sonnet-20240229 "$@"
# python llm_benchmark.py --minimal -m claude-3-haiku-20240229 "$@" (coming soon)
python llm_benchmark.py --minimal -m claude-2.1 "$@"
python llm_benchmark.py --minimal -m claude-instant-1.2 "$@"
# google
python llm_benchmark.py --minimal -k $(gcloud auth print-access-token) -m chat-bison "$@"
python llm_benchmark.py --minimal -m gemini-pro "$@"
# mixtral-8x7b
python llm_benchmark.py --minimal -k $AZURE_EASTUS2_MISTRAL_API_KEY -b https://fixie-mistral-serverless.eastus2.inference.ai.azure.com/v1 "$@"
python llm_benchmark.py --minimal -k $GROQ_API_KEY -b https://api.groq.com/openai/v1 -m mixtral-8x7b-32768 "$@"
python llm_benchmark.py --minimal -k $OCTOML_API_KEY -b https://text.octoai.run/v1 -m mixtral-8x7b-instruct "$@"
python llm_benchmark.py --minimal -k $PERPLEXITY_API_KEY -b https://api.perplexity.ai -m sonar-medium-chat "$@"
# llama2-70b
python llm_benchmark.py --minimal -k $AZURE_WESTUS3_LLAMA2_API_KEY -b https://fixie-llama-2-70b-serverless.westus3.inference.ai.azure.com/v1 "$@"
python llm_benchmark.py --minimal -k $AZURE_EASTUS2_LLAMA2_API_KEY -b https://fixie-llama-2-70b-serverless.eastus2.inference.ai.azure.com/v1 "$@"
python llm_benchmark.py --minimal -k $GROQ_API_KEY -b https://api.groq.com/openai/v1 -m llama2-70b-4096 "$@"
python llm_benchmark.py --minimal -k $OCTOML_API_KEY -b https://text.octoai.run/v1 -m llama-2-70b-chat-fp16 "$@"
python llm_benchmark.py --minimal -k $PERPLEXITY_API_KEY -b https://api.perplexity.ai -m pplx-70b-chat "$@"
python llm_benchmark.py --minimal -m togethercomputer/llama-2-70b-chat "$@"
# llama2-13b
python llm_benchmark.py --minimal -k $OCTOML_API_KEY -b https://text.octoai.run/v1 -m llama-2-13b-chat-fp16 "$@"
python llm_benchmark.py --minimal -m togethercomputer/llama-2-13b-chat "$@"
# llama2-7b
python llm_benchmark.py --minimal -k $PERPLEXITY_API_KEY -b https://api.perplexity.ai -m pplx-7b-chat "$@"
python llm_benchmark.py --minimal -m togethercomputer/llama-2-7b-chat "$@"
python llm_benchmark.py --minimal -m @cf/meta/llama-2-7b-chat-fp16 "$@"
python llm_benchmark.py --minimal -m @cf/meta/llama-2-7b-chat-int8 "$@"
python llm_benchmark.py --minimal -m Neets-7B "$@"
