#!/usr/bin/env python3
"""
Generate and store Capital Benchmark credit memos for a benchmark grid.

This script calls /credit_memo, not /credit_memo_annotation.

Default pilot grid:
- 3 recognisable obligors with contrasting credit profiles
- context profiles: identity_only, rating_only, financials_only, rating_and_financials, full
- policy_mode: none, llm_evaluated, deterministic_evaluated
- prompt_mode: tight, loose
- model_tier: mini

Default run size:
    3 obligors × 5 context profiles × 3 policy modes × 2 prompt modes × 1 mini model = 90 memos

Output layout:
    output_dir/
      run_manifest.json
      summary.csv
      all_memos.jsonl
      raw_memos/
        one full /credit_memo JSON response per run
      memo_markdown/
        one rendered markdown memo per run
      memo_context/
        one memo_context JSON per run
      llm_context/
        one llm_context JSON per run, if returned

Example:
    python run_credit_memo_generation_grid_20_names.py \
        --base-url https://api.capitalbenchmark.net \
        --output-dir benchmark_runs/benchmark_20_mini_memos \
        --require-openai
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_CONTEXT_MODES = ["identity_only", "rating_only", "financials_only", "rating_and_financials", "full"]
DEFAULT_POLICY_MODES = ["none", "llm_evaluated", "deterministic_evaluated"]
DEFAULT_PROMPT_MODES = ["tight", "loose"]
DEFAULT_MODEL_TIERS = ["mini"]

DEFAULT_SYMBOLS = [
    "GOOG",   # Alphabet: very strong credit control
    "BP",     # BP: investment-grade cyclical / commodity exposure
    "AAL",    # American Airlines: weaker leveraged borrower / policy-trigger candidate
]

DEFAULT_CREDIT_REQUEST = {
    "requested_increase_usd": 100_000_000,
    "proposed_exposure_usd": 100_000_000,
    "facility_type": "revolving_credit_facility",
    "purpose": "general_corporate_purposes",
    "currency": "USD",
    "lgd": 0.45,
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_filename(value: str) -> str:
    keep: list[str] = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def short_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_experiment_grid(
    symbols: list[str],
    context_profiles: list[str],
    policy_modes: list[str],
    prompt_modes: list[str],
    model_tiers: list[str],
    repeats: int,
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for symbol, context_mode, policy_mode, prompt_mode, model_tier, repeat in itertools.product(
        symbols,
        context_profiles,
        policy_modes,
        prompt_modes,
        model_tiers,
        range(1, repeats + 1),
    ):
        experiments.append({
            "symbol": symbol,
            "context_mode": context_mode,
            "policy_mode": policy_mode,
            "prompt_mode": prompt_mode,
            "model_tier": model_tier,
            "run_id": repeat,
        })
    return experiments


def parse_credit_request_args(args: argparse.Namespace) -> dict[str, Any]:
    credit_request = dict(DEFAULT_CREDIT_REQUEST)

    optional_fields = [
        "existing_exposure_usd",
        "requested_increase_usd",
        "proposed_exposure_usd",
        "facility_type",
        "purpose",
        "tenor_years",
        "secured",
        "seniority",
        "currency",
        "lgd",
    ]

    for field in optional_fields:
        value = getattr(args, field, None)
        if value is not None:
            credit_request[field] = value

    return credit_request


def request_credit_memo(
    base_url: str,
    experiment: dict[str, Any],
    credit_request: dict[str, Any],
    timeout: int,
    require_openai: bool,
    method: str,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/credit_memo"

    payload = {
        **credit_request,
        "symbol": experiment["symbol"],
        "context_mode": experiment["context_mode"],
        "policy_mode": experiment["policy_mode"],
        "prompt_mode": experiment["prompt_mode"],
        "model_tier": experiment["model_tier"],
        "use_openai": "true",
        "require_openai": "true" if require_openai else "false",
        "include_llm_context": "true",
    }

    if method.upper() == "POST":
        response = requests.post(endpoint, json=payload, timeout=timeout)
    else:
        response = requests.get(endpoint, params=payload, timeout=timeout)

    response.raise_for_status()
    memo_payload = response.json()

    memo_payload.setdefault("benchmark_runner", {})
    memo_payload["benchmark_runner"].update({
        "requested_config": experiment,
        "credit_request": credit_request,
        "base_url": base_url,
        "endpoint": "/credit_memo",
        "method": method.upper(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    return memo_payload


def summarize_memo(memo_payload: dict[str, Any]) -> dict[str, Any]:
    runner = memo_payload.get("benchmark_runner", {}) or {}
    requested = runner.get("requested_config", {}) or {}
    experiment_config = memo_payload.get("experiment_config", {}) or {}
    memo_context = memo_payload.get("memo_context", {}) or {}
    borrower = memo_context.get("borrower", {}) or {}
    ratings = memo_context.get("capital_benchmark_rating", {}) or {}
    credit_request = memo_context.get("credit_request", {}) or {}
    exposure = memo_context.get("exposure_analytics", {}) or {}
    narrative = memo_payload.get("narrative", {}) or {}

    return {
        "generated_at_utc": runner.get("completed_at_utc"),
        "symbol": requested.get("symbol") or borrower.get("symbol"),
        "company_name": borrower.get("company_name"),
        "industry": borrower.get("industry"),
        "sector": borrower.get("sector"),
        "country": borrower.get("country"),
        "context_profile": experiment_config.get("context_profile") or experiment_config.get("context_mode") or requested.get("context_mode"),
        "resolved_visibility": json.dumps(experiment_config.get("resolved_visibility", {}), sort_keys=True),
        "policy_mode": experiment_config.get("policy_mode") or requested.get("policy_mode"),
        "prompt_mode": experiment_config.get("prompt_mode") or requested.get("prompt_mode"),
        "model_tier": experiment_config.get("model_tier") or requested.get("model_tier"),
        "model": memo_payload.get("openai_model") or experiment_config.get("model"),
        "run_id": requested.get("run_id"),
        "experiment_id": experiment_config.get("experiment_id"),
        "narrative_source": memo_payload.get("narrative_source"),
        "fallback_reason": memo_payload.get("fallback_reason"),
        "cb_rating": ratings.get("cb_rating"),
        "cb_pd": ratings.get("cb_pd"),
        "base_el_on_proposed_exposure": exposure.get("base_expected_loss_on_proposed_exposure"),
        "narrative_fields": ",".join(sorted(narrative.keys())) if isinstance(narrative, dict) else None,
        "raw_memo_file": runner.get("raw_memo_file"),
        "memo_markdown_file": runner.get("memo_markdown_file"),
        "memo_context_file": runner.get("memo_context_file"),
        "llm_context_file": runner.get("llm_context_file"),
        "error": None,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate stored credit memos for the benchmark grid.")
    parser.add_argument("--base-url", default="https://api.capitalbenchmark.net")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--output-dir", default=f"benchmark_runs/credit_memo_generation_{utc_stamp()}")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--require-openai", action="store_true", help="Fail an observation if OpenAI fails instead of accepting fallback output.")
    parser.add_argument("--method", choices=["GET", "POST"], default="POST")
    parser.add_argument("--resume", action="store_true", help="Skip raw memo files that already exist.")
    parser.add_argument("--print-default-symbols", action="store_true", help="Print the default 20-symbol benchmark set and exit.")

    parser.add_argument(
        "--context-profiles", "--context-modes",
        dest="context_profiles",
        nargs="+",
        default=DEFAULT_CONTEXT_MODES,
        help="Named information-visibility profiles. --context-modes remains a backward-compatible alias.",
    )
    parser.add_argument("--policy-modes", nargs="+", default=DEFAULT_POLICY_MODES)
    parser.add_argument("--prompt-modes", nargs="+", default=DEFAULT_PROMPT_MODES)
    parser.add_argument("--model-tiers", nargs="+", default=DEFAULT_MODEL_TIERS)

    parser.add_argument("--existing-exposure-usd", type=float, default=None)
    parser.add_argument("--requested-increase-usd", type=float, default=None)
    parser.add_argument("--proposed-exposure-usd", type=float, default=None)
    parser.add_argument("--facility-type", default=None)
    parser.add_argument("--purpose", default=None)
    parser.add_argument("--tenor-years", type=float, default=None)
    parser.add_argument("--secured", default=None)
    parser.add_argument("--seniority", default=None)
    parser.add_argument("--currency", default=None)
    parser.add_argument("--lgd", type=float, default=None)

    args = parser.parse_args()

    if args.print_default_symbols:
        print("\n".join(DEFAULT_SYMBOLS))
        return 0

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_memos"
    markdown_dir = output_dir / "memo_markdown"
    context_dir = output_dir / "memo_context"
    llm_context_dir = output_dir / "llm_context"
    summary_path = output_dir / "summary.csv"
    jsonl_path = output_dir / "all_memos.jsonl"
    errors_path = output_dir / "errors.jsonl"
    manifest_path = output_dir / "run_manifest.json"

    credit_request = parse_credit_request_args(args)
    experiments = build_experiment_grid(
        symbols=args.symbols,
        context_profiles=args.context_profiles,
        policy_modes=args.policy_modes,
        prompt_modes=args.prompt_modes,
        model_tiers=args.model_tiers,
        repeats=args.repeats,
    )

    manifest = {
        "runner_version": "credit_memo_generation_grid_v0.2_context_profiles",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "endpoint": "/credit_memo",
        "symbols": args.symbols,
        "context_profiles": args.context_profiles,
        "policy_modes": args.policy_modes,
        "prompt_modes": args.prompt_modes,
        "model_tiers": args.model_tiers,
        "repeats": args.repeats,
        "require_openai": bool(args.require_openai),
        "method": args.method,
        "credit_request": credit_request,
        "n_experiments": len(experiments),
    }
    write_json(manifest_path, manifest)

    print(f"Generating {len(experiments)} credit memos")
    print(f"Output directory: {output_dir}")

    summary_rows: list[dict[str, Any]] = []

    for i, experiment in enumerate(experiments, start=1):
        experiment_hash = short_hash({**experiment, "credit_request": credit_request})
        raw_name = (
            f"{i:04d}__{safe_filename(experiment['symbol'])}"
            f"__ctx_{experiment['context_mode']}"
            f"__policy_{experiment['policy_mode']}"
            f"__prompt_{experiment['prompt_mode']}"
            f"__tier_{experiment['model_tier']}"
            f"__run_{experiment['run_id']}"
            f"__{experiment_hash}.json"
        )

        raw_path = raw_dir / raw_name
        markdown_path = markdown_dir / raw_name.replace(".json", ".md")
        context_path = context_dir / raw_name.replace(".json", "__memo_context.json")
        llm_context_path = llm_context_dir / raw_name.replace(".json", "__llm_context.json")

        label = (
            f"[{i}/{len(experiments)}] "
            f"{experiment['symbol']} | "
            f"ctx={experiment['context_mode']} | "
            f"policy={experiment['policy_mode']} | "
            f"prompt={experiment['prompt_mode']} | "
            f"tier={experiment['model_tier']} | "
            f"run={experiment['run_id']}"
        )

        try:
            if args.resume and raw_path.exists():
                memo_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                memo_payload.setdefault("benchmark_runner", {})
                print(f"{label} -> skipped existing")
            else:
                print(f"{label} -> generating")
                memo_payload = request_credit_memo(
                    base_url=args.base_url,
                    experiment=experiment,
                    credit_request=credit_request,
                    timeout=args.timeout,
                    require_openai=bool(args.require_openai),
                    method=args.method,
                )

            memo_payload.setdefault("benchmark_runner", {})
            memo_payload["benchmark_runner"].update({
                "raw_memo_file": str(raw_path),
                "memo_markdown_file": str(markdown_path),
                "memo_context_file": str(context_path),
                "llm_context_file": str(llm_context_path),
            })

            write_json(raw_path, memo_payload)
            append_jsonl(jsonl_path, memo_payload)

            memo_markdown = memo_payload.get("memo_markdown")
            if isinstance(memo_markdown, str) and memo_markdown.strip():
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text(memo_markdown, encoding="utf-8")

            memo_context = memo_payload.get("memo_context")
            if isinstance(memo_context, dict):
                write_json(context_path, memo_context)

            llm_context = memo_payload.get("llm_context")
            if isinstance(llm_context, dict):
                write_json(llm_context_path, llm_context)

            row = summarize_memo(memo_payload)
            row["raw_memo_file"] = str(raw_path)
            row["memo_markdown_file"] = str(markdown_path)
            row["memo_context_file"] = str(context_path)
            row["llm_context_file"] = str(llm_context_path)
            summary_rows.append(row)

            print(
                f"    source={row.get('narrative_source')} "
                f"model={row.get('model')} "
                f"rating={row.get('cb_rating')} "
                f"fallback={row.get('fallback_reason')}"
            )

        except Exception as exc:
            error_record = {
                "error_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": experiment,
                "credit_request": credit_request,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_memo_file": str(raw_path),
            }
            append_jsonl(errors_path, error_record)

            summary_rows.append({
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": experiment["symbol"],
                "company_name": None,
                "industry": None,
                "sector": None,
                "country": None,
                "context_profile": experiment["context_mode"],
                "resolved_visibility": None,
                "policy_mode": experiment["policy_mode"],
                "prompt_mode": experiment["prompt_mode"],
                "model_tier": experiment["model_tier"],
                "model": None,
                "run_id": experiment["run_id"],
                "experiment_id": None,
                "narrative_source": None,
                "fallback_reason": None,
                "cb_rating": None,
                "cb_pd": None,
                "base_el_on_proposed_exposure": None,
                "narrative_fields": None,
                "raw_memo_file": str(raw_path),
                "memo_markdown_file": str(markdown_path),
                "memo_context_file": str(context_path),
                "llm_context_file": str(llm_context_path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"    ERROR: {type(exc).__name__}: {exc}")

        write_summary_csv(summary_path, summary_rows)

        if args.sleep_seconds > 0 and i < len(experiments):
            time.sleep(args.sleep_seconds)

    write_summary_csv(summary_path, summary_rows)

    print("Done.")
    print(f"Summary CSV: {summary_path}")
    print(f"All memos JSONL: {jsonl_path}")
    print(f"Raw memos: {raw_dir}")
    print(f"Memo markdown: {markdown_dir}")
    print(f"Memo contexts: {context_dir}")
    if errors_path.exists():
        print(f"Errors JSONL: {errors_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
