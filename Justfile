# This is the main Justfile for the Fixie repo.
# To install Just, see: https://github.com/casey/just#installation

# Allow for positional arguments in Just recipes.
set positional-arguments := true

# Default recipe that runs if you type "just".
default: format check test

# Install dependencies for local development.
install:
    pip install poetry==1.7.1
    poetry install --sync

format:
    poetry run autoflake . --remove-all-unused-imports --quiet --in-place -r --exclude third_party
    poetry run isort . --force-single-line-imports
    poetry run black .

check:
    poetry run black . --check
    poetry run isort . --check --force-single-line-imports
    poetry run autoflake . --check --quiet --remove-all-unused-imports -r --exclude third_party
    poetry run mypy .

deploy *FLAGS:
    flyctl deploy {{FLAGS}}

server:
    just python app.py

curl:
    curl -X POST "https://ai-benchmarks.fly.dev/bench?max_tokens=20" -H fly-prefer-region:sea

curl_local:
    curl -X POST "http://localhost:8000/bench?max_tokens=20"

test *FLAGS:
    poetry run pytest {{FLAGS}}

python *FLAGS:
    poetry run python {{FLAGS}}

llm *FLAGS:
    poetry run python llm_benchmark.py {{FLAGS}}

llms *FLAGS:
    poetry run python llm_benchmark_suite.py {{FLAGS}}
