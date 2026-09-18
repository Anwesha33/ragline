SHELL := /bin/bash
COMPOSE := docker compose -f deploy/docker-compose.yml
API := http://localhost:8081
PY := ./.venv/bin/python

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Compile the gateway
	go build ./...

.PHONY: test
test: ## Run Go and Python unit tests
	go test ./... -count=1
	$(PY) -m unittest discover -s eval -p "test_*.py" -v

.PHONY: lint
lint: ## gofmt + go vet
	@test -z "$$(gofmt -l . | tee /dev/stderr)" || (echo "run gofmt -w ." && exit 1)
	go vet ./...

.PHONY: venv
venv: ## Create the Python environment for workers and evaluation
	python3 -m venv .venv
	$(PY) -m pip install -q -r workers/ingest/requirements.txt

.PHONY: up
up: ## Start the full stack
	$(COMPOSE) up -d --build
	@echo "gateway on $(API)"

.PHONY: down
down: ## Stop the stack and delete its volumes
	$(COMPOSE) down -v

.PHONY: infra
infra: ## Start only postgres, redis and kafka
	$(COMPOSE) up -d postgres redis kafka

.PHONY: logs
logs: ## Tail the ingest workers
	$(COMPOSE) logs -f ingest

.PHONY: ingest
ingest: ## Upload every document in corpus/ and wait for it to be indexed
	@bash scripts/ingest-corpus.sh

.PHONY: ask
ask: ## Ask a question: make ask Q="how long are idempotency keys kept?"
	@test -n "$(Q)" || (echo 'usage: make ask Q="your question"' && exit 1)
	@curl -sS -N -X POST $(API)/v1/chat -H 'content-type: application/json' \
		-d "$$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$(Q)")"

.PHONY: search
search: ## Retrieval only: make search Q="webhook retry schedule"
	@test -n "$(Q)" || (echo 'usage: make search Q="your query"' && exit 1)
	@curl -sS -X POST $(API)/v1/search -H 'content-type: application/json' \
		-d "$$(python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$(Q)")" \
		| python3 -m json.tool

.PHONY: stats
stats: ## Latency, token and cost statistics
	@curl -sS $(API)/v1/stats | python3 -m json.tool

.PHONY: eval
eval: ## Score retrieval and answer quality against the golden set
	$(PY) eval/run_eval.py --out eval/report.json
