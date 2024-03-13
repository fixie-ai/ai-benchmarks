echo "Provider/Model                                   | TTR   | Total"
python dalle.py --minimal "$@"
python dalle.py --minimal -k $AZURE_SECENTRAL_OPENAI_API_KEY -b https://fixie-secentral.openai.azure.com "$@"
python dalle.py --minimal -k $AZURE_EASTUS_OPENAI_API_KEY -b https://fixie-eastus.openai.azure.com "$@"
