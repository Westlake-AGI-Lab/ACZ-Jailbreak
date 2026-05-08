#!/usr/bin/env python3
"""Judge model outputs as SAFE/UNSAFE and attach binary safety scores."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acz_jailbreak.io import save_json
from acz_jailbreak.llm import create_client, extract_safety_score, provider_names
from acz_jailbreak.prompts import JUDGE_PROMPT

import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Result JSON from evaluate_text.py or evaluate_images.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--content-key", default="content")
    parser.add_argument("--score-key", default="eval_score")
    parser.add_argument("--provider", default="openai", choices=provider_names())
    parser.add_argument("--model", required=True, help="Judge model name.")
    parser.add_argument("--base-url", default=None, help="Override the provider default base URL.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=None, help="Override the provider default API key env var.")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def unwrap_results(data: object) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError("Input must be a list or an object with a 'results' list.")


def main() -> None:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = unwrap_results(data)
    selected = records[: args.limit] if args.limit is not None else records

    client = create_client(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
    )
    request_kwargs = {"temperature": 0}
    if args.reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.enable_thinking:
        request_kwargs["enable_thinking"] = True
    if args.thinking_budget is not None:
        request_kwargs["thinking_budget"] = args.thinking_budget

    def run(record: dict, index: int) -> tuple[int, dict]:
        text = str(record.get(args.content_key, ""))
        prompt = JUDGE_PROMPT.format(text_data=text)
        judgment = client.chat_text(prompt, **request_kwargs)
        updated = dict(record)
        updated["judge_content"] = judgment["content"]
        updated[args.score_key] = extract_safety_score(judgment["content"])
        return index, updated

    updates: dict[int, dict] = {}
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run, record, index): index for index, record in enumerate(selected)}
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                result_index, updated = future.result()
                updates[result_index] = updated
            except Exception as exc:
                errors.append({"task_id": index, "error": str(exc)})

    judged = [updates.get(index, record) for index, record in enumerate(selected)]
    save_json({"results": judged, "errors": errors}, args.output)
    print(f"Saved judged results to {args.output}; errors: {len(errors)}")


if __name__ == "__main__":
    main()
