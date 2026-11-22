.PHONY: up down demo notebook dagster test ingest snapshot-list prepare experiment campaign-plan campaign-run

up:
	docker compose up -d

down:
	docker compose down

demo:
	uv run qumin demo

notebook:
	uv run jupyter lab notebooks/research_walkthrough.ipynb

dagster:
	uv run dagster dev -w config/dagster-workspace.yaml

test:
	uv run pytest

STREAM ?= equities-daily
CAMPAIGN ?= config/campaign.example.yaml
CONFIG ?= config/pipeline.yaml

ingest:
	uv run qumin ingest $(STREAM)

snapshot-list:
	uv run qumin --json snapshot-list

prepare:
	uv run qumin --json prepare --config $(CONFIG)

experiment:
	uv run qumin --json experiment --config $(CONFIG)

campaign-plan:
	uv run qumin --json campaign-plan --campaign $(CAMPAIGN)

campaign-run:
	uv run qumin --json campaign-run --campaign $(CAMPAIGN)
