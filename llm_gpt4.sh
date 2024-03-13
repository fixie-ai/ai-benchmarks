echo "Provider/Model                                         | TTR  | TTFT | TPS  | Total | Tokens"
python llm_benchmark.py --minimal -m gpt-4-0125-preview "$@"
python llm_benchmark.py --minimal -k $AZURE_SCENTRALUS_OPENAI_API_KEY -b https://fixie-scentralus.openai.azure.com -m gpt-4-0125-preview "$@"

python llm_benchmark.py --minimal -m gpt-4-1106-preview "$@"
python llm_benchmark.py --minimal -b https://fixie-westus.openai.azure.com -m gpt-4-1106-preview "$@"
python llm_benchmark.py --minimal -k $AZURE_EASTUS2_OPENAI_API_KEY -b https://fixie-openai-sub-with-gpt4.openai.azure.com -m gpt-4-1106-preview "$@"
python llm_benchmark.py --minimal -k $AZURE_FRCENTRAL_OPENAI_API_KEY -b https://fixie-frcentral.openai.azure.com -m gpt-4-1106-preview "$@"
python llm_benchmark.py --minimal -k $AZURE_SECENTRAL_OPENAI_API_KEY -b https://fixie-secentral.openai.azure.com -m gpt-4-1106-preview "$@"
python llm_benchmark.py --minimal -k $AZURE_UKSOUTH_OPENAI_API_KEY -b https://fixie-uksouth.openai.azure.com -m gpt-4-1106-preview "$@"

python llm_benchmark.py --minimal -m gpt-4-32k "$@"
#python llm_benchmark.py --minimal -b https://fixie-westus.openai.azure.com -m gpt-4-32k"$@"
python llm_benchmark.py --minimal -k $AZURE_EASTUS2_OPENAI_API_KEY -b https://fixie-openai-sub-with-gpt4.openai.azure.com -m gpt-4-32k "$@"
python llm_benchmark.py --minimal -k $AZURE_FRCENTRAL_OPENAI_API_KEY -b https://fixie-frcentral.openai.azure.com -m gpt-4-32k "$@"
python llm_benchmark.py --minimal -k $AZURE_SECENTRAL_OPENAI_API_KEY -b https://fixie-secentral.openai.azure.com -m gpt-4-32k "$@"
python llm_benchmark.py --minimal -k $AZURE_UKSOUTH_OPENAI_API_KEY -b https://fixie-uksouth.openai.azure.com -m gpt-4-32k "$@"

# EUS  (-), dall-e
# EUS2 (-0613, -1106, -32k)
# WUS  (-1106 and -vision)
# WUS3 (-)
# NCUS (-)
# SCUS (-0125 only)

# UKS (-1106 and -32k)
# FRC (-1106 and -32k)
# SEC (-1106 and -32k), dall-e
