#!/usr/bin/env python3
"""Run text-only jailbreak evaluation against an OpenAI-compatible model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acz_jailbreak.io import iter_limited, load_records, save_json
from acz_jailbreak.llm import create_client, provider_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="response")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--provider", default="openai", choices=provider_names())
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None, help="Override the provider default base URL.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=None, help="Override the provider default API key env var.")
    parser.add_argument("--system", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(iter_limited(load_records(args.input), args.limit))
    client = create_client(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
    )
    request_kwargs = {}
    if args.temperature is not None:
        request_kwargs["temperature"] = args.temperature
    if args.reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.enable_thinking:
        request_kwargs["enable_thinking"] = True
    if args.thinking_budget is not None:
        request_kwargs["thinking_budget"] = args.thinking_budget

    def run(record: dict, task_id: int) -> dict:
        item_id = str(record.get(args.id_column, task_id))
        text = str(record.get(args.text_column, ""))
        result = client.chat_text(text, system=args.system, **request_kwargs)
        result.update({"id": item_id, "task_id": task_id})
        return result

    results: list[dict] = []
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run, record, index): (record, index) for index, record in enumerate(records)}
        for future in as_completed(future_map):
            record, index = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"id": record.get(args.id_column, index), "task_id": index, "error": str(exc)})

    results.sort(key=lambda item: item.get("task_id", 0))
    save_json({"results": results, "errors": errors}, args.output)
    print(f"Saved {len(results)} results and {len(errors)} errors to {args.output}")


if __name__ == "__main__":
    main()
