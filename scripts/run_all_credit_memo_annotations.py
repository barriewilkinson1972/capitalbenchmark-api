#!/usr/bin/env python3
"""
Batch-generate document maps and annotations for all stored benchmark memos.

Default folder layout
---------------------
benchmark_runs/benchmark_20_mini_memos/
    raw_memos/
    document_maps/
    annotations/

The script:
1. Finds all JSON memo payloads in raw_memos.
2. Generates one document map per memo.
3. Generates one annotation file per memo.
4. Writes manifests and error logs.
5. Supports resume, overwrite, limits, and strict validation.

Example
-------
python scripts/run_all_credit_memo_annotations.py \
    --benchmark-dir benchmark_runs/benchmark_20_mini_memos \
    --parser-module model/credit_memo_parser.py \
    --annotation-module model/credit_memo_annotation.py

If your parser function has a non-standard name, supply it explicitly:

python scripts/run_all_credit_memo_annotations.py \
    --parser-function parse_credit_memo \
    --annotation-function annotate_credit_memo
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_PARSER_FUNCTIONS = (
    "parse_markdown_to_blocks",
    "parse_credit_memo",
    "build_document_map",
    "create_document_map",
    "parse_memo",
)

DEFAULT_ANNOTATION_FUNCTIONS = (
    "annotate_credit_memo",
)


@dataclass
class RunResult:
    memo_file: str
    memo_id: str | None
    status: str
    document_map_file: str | None
    annotation_file: str | None
    parser_version: str | None
    annotation_engine_version: str | None
    annotation_count: int | None
    overall_score: float | int | None
    location_resolution_score: float | int | None
    validation_status: str | None
    elapsed_seconds: float
    error_type: str | None = None
    error_message: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(value, dict):
        raise TypeError(f"Expected a top-level JSON object in {path}")

    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_module(module_path: Path, label: str) -> Any:
    if not module_path.exists():
        raise FileNotFoundError(f"{label} module not found: {module_path}")

    module_name = f"_capital_benchmark_{label}_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {label} module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_function(
    module: Any,
    explicit_name: str | None,
    candidates: Iterable[str],
    label: str,
) -> Callable[..., Any]:
    names = [explicit_name] if explicit_name else list(candidates)

    for name in names:
        if not name:
            continue
        function = getattr(module, name, None)
        if callable(function):
            return function

    available = sorted(
        name
        for name, value in vars(module).items()
        if callable(value) and not name.startswith("_")
    )
    raise AttributeError(
        f"Could not find the {label} function. Tried: {names}. "
        f"Public callables in module: {available}"
    )


def memo_id_from_payload(
    memo_payload: dict[str, Any],
    fallback_stem: str,
) -> str:
    candidates = (
        memo_payload.get("memo_id"),
        (memo_payload.get("benchmark_runner") or {}).get("memo_id"),
        (memo_payload.get("experiment_config") or {}).get("memo_id"),
    )
    for value in candidates:
        if value:
            return str(value)
    return fallback_stem


def call_parser(
    parser_function: Callable[..., Any],
    memo_payload: dict[str, Any],
    fallback_memo_id: str,
) -> dict[str, Any]:
    """Call common parser interfaces without coupling the runner to one module.

    Supported parser styles include:
      * parse_markdown_to_blocks(memo_id, markdown)
      * parse_credit_memo(memo_payload)
      * parse_credit_memo(memo_payload=...)
      * functions accepting canonical fields by keyword
    """
    signature = inspect.signature(parser_function)
    parameters = signature.parameters
    parameter_names = list(parameters)

    memo_id = memo_id_from_payload(memo_payload, fallback_memo_id)
    markdown = (
        memo_payload.get("memo_markdown")
        or memo_payload.get("markdown")
        or memo_payload.get("narrative")
    )

    if (
        "memo_id" in parameters
        and ("markdown" in parameters or "memo_markdown" in parameters)
    ):
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError(
                "The parser requires Markdown, but the memo payload does not "
                "contain a non-empty memo_markdown, markdown, or narrative field."
            )

        kwargs: dict[str, Any] = {"memo_id": memo_id}
        if "markdown" in parameters:
            kwargs["markdown"] = markdown
        else:
            kwargs["memo_markdown"] = markdown

        result = parser_function(**kwargs)

    elif "memo_payload" in parameters:
        result = parser_function(memo_payload=memo_payload)

    elif "payload" in parameters:
        result = parser_function(payload=memo_payload)

    elif len(parameters) == 1:
        result = parser_function(memo_payload)

    else:
        kwargs = {}
        canonical_values = {
            **memo_payload,
            "memo_id": memo_id,
            "markdown": markdown,
            "memo_markdown": markdown,
        }

        for name in parameter_names:
            if name in canonical_values:
                kwargs[name] = canonical_values[name]

        missing_required = [
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
            and name not in kwargs
        ]

        if missing_required:
            raise TypeError(
                "Could not infer how to call parser function "
                f"{parser_function.__name__}{signature}. "
                f"Missing required arguments: {missing_required}"
            )

        result = parser_function(**kwargs)

    if not isinstance(result, dict):
        raise TypeError(
            f"Parser returned {type(result).__name__}; expected dict."
        )

    return result


def call_annotator(
    annotation_function: Callable[..., Any],
    memo_payload: dict[str, Any],
    document_map: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(annotation_function)
    parameters = signature.parameters

    if "memo_payload" in parameters and "document_map" in parameters:
        result = annotation_function(
            memo_payload=memo_payload,
            document_map=document_map,
        )
    elif len(parameters) >= 2:
        result = annotation_function(memo_payload, document_map)
    else:
        raise TypeError(
            "Annotation function must accept memo_payload and document_map. "
            f"Found {annotation_function.__name__}{signature}."
        )

    if not isinstance(result, dict):
        raise TypeError(
            f"Annotator returned {type(result).__name__}; expected dict."
        )

    return result


def existing_output_is_current(
    output_path: Path,
    memo_payload: dict[str, Any],
    expected_version_key: str,
    expected_version: str | None,
) -> bool:
    """Conservative resume check.

    If no expected version is supplied, an existing readable JSON file counts
    as complete. If a version is supplied, it must match.
    """
    if not output_path.exists():
        return False

    try:
        existing = load_json(output_path)
    except Exception:
        return False

    if expected_version is not None:
        if str(existing.get(expected_version_key)) != str(expected_version):
            return False

    source_document = existing.get("source_document") or {}
    expected_memo_id = memo_id_from_payload(memo_payload, output_path.stem)

    existing_memo_id = (
        existing.get("memo_id")
        or source_document.get("memo_id")
    )
    if existing_memo_id and str(existing_memo_id) != str(expected_memo_id):
        return False

    return True


def annotation_metrics(annotation: dict[str, Any]) -> dict[str, Any]:
    scores = annotation.get("scores") or {}
    validation = annotation.get("validation") or {}
    annotations = annotation.get("annotations") or []

    return {
        "annotation_engine_version": annotation.get(
            "annotation_engine_version"
        ),
        "annotation_count": len(annotations),
        "overall_score": scores.get("overall_score"),
        "location_resolution_score": scores.get(
            "location_resolution_score"
        ),
        "validation_status": validation.get("status"),
    }


def write_csv_manifest(path: Path, results: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RunResult.__dataclass_fields__.keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate document maps and annotations for every raw benchmark memo."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark_runs/benchmark_20_mini_memos",
        help="Benchmark run directory containing raw_memos/.",
    )
    parser.add_argument(
        "--raw-folder",
        default="raw_memos",
        help="Input folder name inside benchmark-dir.",
    )
    parser.add_argument(
        "--document-map-folder",
        default="document_maps",
        help="Output folder name for parsed document maps.",
    )
    parser.add_argument(
        "--annotation-folder",
        default="annotations",
        help="Output folder name for annotation JSON files.",
    )
    parser.add_argument(
        "--parser-module",
        default="model/credit_memo_parser.py",
        help="Path to the document parser module.",
    )
    parser.add_argument(
        "--parser-function",
        default=None,
        help="Parser function name; auto-detected when omitted.",
    )
    parser.add_argument(
        "--annotation-module",
        default="model/credit_memo_annotation.py",
        help="Path to the annotation module.",
    )
    parser.add_argument(
        "--annotation-function",
        default="annotate_credit_memo",
        help="Annotation function name.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Input filename glob, relative to raw-folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files, useful for a smoke test.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Skip files alphabetically before this filename.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat annotation validation warnings/failures as failed records."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a memo fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create folders and show planned files without running models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path.cwd()
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    raw_dir = benchmark_dir / args.raw_folder
    document_map_dir = benchmark_dir / args.document_map_folder
    annotation_dir = benchmark_dir / args.annotation_folder

    parser_module_path = Path(args.parser_module).expanduser()
    if not parser_module_path.is_absolute():
        parser_module_path = (project_root / parser_module_path).resolve()

    annotation_module_path = Path(args.annotation_module).expanduser()
    if not annotation_module_path.is_absolute():
        annotation_module_path = (project_root / annotation_module_path).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw memo directory not found: {raw_dir}")

    document_map_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    memo_files = sorted(
        path for path in raw_dir.glob(args.pattern)
        if path.is_file()
        and "__annotation_test" not in path.stem
    )

    if args.start_at:
        memo_files = [
            path for path in memo_files if path.name >= args.start_at
        ]

    if args.limit is not None:
        memo_files = memo_files[: args.limit]

    print("Capital Benchmark batch annotation run")
    print(f"  Benchmark directory: {benchmark_dir}")
    print(f"  Raw memos:          {raw_dir}")
    print(f"  Document maps:      {document_map_dir}")
    print(f"  Annotations:        {annotation_dir}")
    print(f"  Files selected:     {len(memo_files)}")
    print(f"  Overwrite:          {args.overwrite}")
    print(f"  Strict:             {args.strict}")

    if args.dry_run:
        for memo_file in memo_files:
            print(
                f"  {memo_file.name}\n"
                f"    -> {document_map_dir / memo_file.name}\n"
                f"    -> {annotation_dir / memo_file.name}"
            )
        print("\nDry run complete.")
        return 0

    parser_module = load_module(parser_module_path, "parser")
    annotation_module = load_module(
        annotation_module_path,
        "annotation",
    )

    parser_function = resolve_function(
        parser_module,
        args.parser_function,
        DEFAULT_PARSER_FUNCTIONS,
        "parser",
    )
    annotation_function = resolve_function(
        annotation_module,
        args.annotation_function,
        DEFAULT_ANNOTATION_FUNCTIONS,
        "annotation",
    )

    parser_version = getattr(parser_module, "PARSER_VERSION", None)
    annotation_engine_version = getattr(
        annotation_module,
        "ANNOTATION_ENGINE_VERSION",
        None,
    )

    print(f"  Parser function:    {parser_function.__name__}")
    print(f"  Parser version:     {parser_version or '(not declared)'}")
    print(f"  Annotator function: {annotation_function.__name__}")
    print(
        "  Annotation engine: "
        f"{annotation_engine_version or '(not declared)'}"
    )

    started_at = utc_now()
    run_started = time.perf_counter()
    results: list[RunResult] = []

    for index, memo_file in enumerate(memo_files, start=1):
        item_started = time.perf_counter()
        memo_id: str | None = None
        map_path = document_map_dir / memo_file.name
        annotation_path = annotation_dir / memo_file.name

        print(f"\n[{index}/{len(memo_files)}] {memo_file.name}")

        try:
            memo_payload = load_json(memo_file)
            memo_id = memo_id_from_payload(memo_payload, memo_file.stem)

            map_reusable = (
                not args.overwrite
                and existing_output_is_current(
                    map_path,
                    memo_payload,
                    expected_version_key="parser_version",
                    expected_version=parser_version,
                )
            )

            if map_reusable:
                document_map = load_json(map_path)
                print("  Document map: reused")
            else:
                document_map = call_parser(
                    parser_function,
                    memo_payload,
                    fallback_memo_id=memo_file.stem,
                )
                write_json(map_path, document_map)
                print("  Document map: generated")

            annotation_reusable = (
                not args.overwrite
                and existing_output_is_current(
                    annotation_path,
                    memo_payload,
                    expected_version_key="annotation_engine_version",
                    expected_version=annotation_engine_version,
                )
            )

            if annotation_reusable:
                annotation = load_json(annotation_path)
                print("  Annotation: reused")
            else:
                annotation = call_annotator(
                    annotation_function,
                    memo_payload,
                    document_map,
                )
                write_json(annotation_path, annotation)
                print("  Annotation: generated")

            metrics = annotation_metrics(annotation)
            validation_status = metrics["validation_status"]

            if args.strict and validation_status != "passed":
                raise RuntimeError(
                    "Annotation validation did not pass: "
                    f"{validation_status!r}"
                )

            elapsed = time.perf_counter() - item_started
            results.append(
                RunResult(
                    memo_file=memo_file.name,
                    memo_id=memo_id,
                    status="success",
                    document_map_file=str(map_path),
                    annotation_file=str(annotation_path),
                    parser_version=str(
                        document_map.get("parser_version")
                        or parser_version
                        or ""
                    ) or None,
                    annotation_engine_version=metrics[
                        "annotation_engine_version"
                    ],
                    annotation_count=metrics["annotation_count"],
                    overall_score=metrics["overall_score"],
                    location_resolution_score=metrics[
                        "location_resolution_score"
                    ],
                    validation_status=validation_status,
                    elapsed_seconds=round(elapsed, 4),
                )
            )

            print(
                "  Result: success; "
                f"annotations={metrics['annotation_count']}; "
                f"score={metrics['overall_score']}; "
                f"locations={metrics['location_resolution_score']}; "
                f"validation={validation_status}; "
                f"{elapsed:.2f}s"
            )

        except Exception as exc:
            elapsed = time.perf_counter() - item_started
            results.append(
                RunResult(
                    memo_file=memo_file.name,
                    memo_id=memo_id,
                    status="failed",
                    document_map_file=(
                        str(map_path) if map_path.exists() else None
                    ),
                    annotation_file=(
                        str(annotation_path)
                        if annotation_path.exists()
                        else None
                    ),
                    parser_version=None,
                    annotation_engine_version=None,
                    annotation_count=None,
                    overall_score=None,
                    location_resolution_score=None,
                    validation_status=None,
                    elapsed_seconds=round(elapsed, 4),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

            print(f"  Result: FAILED — {type(exc).__name__}: {exc}")
            traceback.print_exc()

            if args.stop_on_error:
                break

    completed_at = utc_now()
    total_elapsed = time.perf_counter() - run_started

    success_count = sum(result.status == "success" for result in results)
    failed_count = sum(result.status == "failed" for result in results)

    run_manifest = {
        "run_schema": "credit_memo_batch_annotation_run",
        "run_schema_version": "1.0.0",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_seconds": round(total_elapsed, 4),
        "benchmark_directory": str(benchmark_dir),
        "raw_memo_directory": str(raw_dir),
        "document_map_directory": str(document_map_dir),
        "annotation_directory": str(annotation_dir),
        "parser_module": str(parser_module_path),
        "parser_function": parser_function.__name__,
        "parser_version": parser_version,
        "annotation_module": str(annotation_module_path),
        "annotation_function": annotation_function.__name__,
        "annotation_engine_version": annotation_engine_version,
        "selected_file_count": len(memo_files),
        "processed_file_count": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "results": [asdict(result) for result in results],
    }

    manifest_json_path = annotation_dir / "_batch_manifest.json"
    manifest_csv_path = annotation_dir / "_batch_manifest.csv"
    write_json(manifest_json_path, run_manifest)
    write_csv_manifest(manifest_csv_path, results)

    print("\nBatch run complete")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {failed_count}")
    print(f"  Elapsed:    {total_elapsed:.2f}s")
    print(f"  JSON log:   {manifest_json_path}")
    print(f"  CSV log:    {manifest_csv_path}")

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
