#!/usr/bin/env python3
"""
End-to-end test for the revised Capital Benchmark annotation engine.

Inputs:
1. The original stored memo_payload JSON.
2. The parsed document-map JSON.

The script dynamically loads credit_memo_annotation_v1.py, runs:

    annotate_credit_memo(
        memo_payload=memo_payload,
        document_map=document_map,
    )

and writes the resulting annotation JSON.

Example:
    python test_revised_annotation.py \
        --memo-payload /path/to/raw_memo.json \
        --document-map /path/to/document_map.json \
        --annotation-module /path/to/credit_memo_annotation_v1.py \
        --output /path/to/test_annotation.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level JSON object in {path}")

    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_annotator(
    module_path: Path,
) -> Callable[..., dict[str, Any]]:
    if not module_path.exists():
        raise FileNotFoundError(f"Annotation module not found: {module_path}")

    module_name = "credit_memo_annotation_test_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    annotator = getattr(module, "annotate_credit_memo", None)
    if not callable(annotator):
        raise AttributeError(
            f"{module_path} does not define callable annotate_credit_memo()"
        )

    return annotator


def validate_inputs(
    memo_payload: dict[str, Any],
    document_map: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []

    required_payload_keys = {"memo_context", "narrative", "memo_markdown"}
    missing_payload_keys = sorted(required_payload_keys - set(memo_payload))
    if missing_payload_keys:
        warnings.append(
            "Memo payload is missing expected keys: "
            + ", ".join(missing_payload_keys)
        )

    required_map_keys = {
        "memo_id",
        "parser_version",
        "source_sha256",
        "sections",
    }
    missing_map_keys = sorted(required_map_keys - set(document_map))
    if missing_map_keys:
        warnings.append(
            "Document map is missing expected keys: "
            + ", ".join(missing_map_keys)
        )

    if not isinstance(document_map.get("sections", []), list):
        warnings.append("document_map.sections is not a list")

    payload_memo_id = (
        memo_payload.get("memo_id")
        or (memo_payload.get("benchmark_runner") or {}).get("memo_id")
        or (memo_payload.get("experiment_config") or {}).get("memo_id")
    )
    map_memo_id = document_map.get("memo_id")

    if payload_memo_id and map_memo_id and str(payload_memo_id) != str(map_memo_id):
        warnings.append(
            f"Memo ID mismatch: payload={payload_memo_id!r}, "
            f"document_map={map_memo_id!r}"
        )

    return warnings


def annotation_summary(result: dict[str, Any]) -> dict[str, Any]:
    scores = result.get("scores") or {}
    diagnostics = result.get("diagnostics") or {}
    location_diag = diagnostics.get("location_resolution") or {}
    validation = result.get("validation") or {}
    annotations = result.get("annotations") or []

    target_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}

    for annotation in annotations:
        category = str(annotation.get("category") or "unknown")
        severity = str(annotation.get("severity") or "unknown")
        location = annotation.get("location") or {}
        target_type = str(location.get("target_type") or "unknown")

        category_counts[category] = category_counts.get(category, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        target_counts[target_type] = target_counts.get(target_type, 0) + 1

    return {
        "annotation_schema": result.get("annotation_schema"),
        "annotation_schema_version": result.get("annotation_schema_version"),
        "annotation_engine_version": result.get("annotation_engine_version"),
        "memo_id": (result.get("source_document") or {}).get("memo_id"),
        "overall_score": scores.get("overall_score"),
        "policy_detection_score": scores.get("policy_detection_score"),
        "missing_information_detection_score": scores.get(
            "missing_information_detection_score"
        ),
        "location_resolution_score": scores.get("location_resolution_score"),
        "annotation_count": len(annotations),
        "resolved_annotation_count": location_diag.get(
            "resolved_annotation_count"
        ),
        "unresolved_annotation_count": location_diag.get(
            "unresolved_annotation_count"
        ),
        "validation_status": validation.get("status"),
        "validation_error_count": validation.get("error_count"),
        "validation_warning_count": validation.get("warning_count"),
        "category_counts": category_counts,
        "severity_counts": severity_counts,
        "target_type_counts": target_counts,
    }


def print_annotation_locations(result: dict[str, Any]) -> None:
    annotations = result.get("annotations") or []

    if not annotations:
        print("\nNo annotations generated.")
        return

    print("\nAnnotation locations:")
    for annotation in annotations:
        finding = annotation.get("generated_finding") or {}
        location = annotation.get("location") or {}

        title = (
            finding.get("title")
            or annotation.get("title")
            or "(untitled annotation)"
        )
        target_type = location.get("target_type")
        status = location.get("resolution_status")
        block_ids = location.get("block_ids") or []
        expected_sections = location.get("expected_section_types") or []

        location_text = ""
        if block_ids:
            location_text = ", ".join(str(x) for x in block_ids)
        elif expected_sections:
            location_text = "expected in: " + ", ".join(
                str(x) for x in expected_sections
            )
        else:
            location_text = "no concrete target"

        print(
            f"  {annotation.get('annotation_id')} "
            f"[{annotation.get('severity')}] "
            f"{title}\n"
            f"      target={target_type}; status={status}; {location_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the revised location-aware credit memo annotator."
    )
    parser.add_argument(
        "--memo-payload",
        required=True,
        help="Path to the original stored memo_payload JSON.",
    )
    parser.add_argument(
        "--document-map",
        required=True,
        help="Path to the parsed document-map JSON.",
    )
    parser.add_argument(
        "--annotation-module",
        default="credit_memo_annotation_v1.py",
        help=(
            "Path to the revised annotation module. "
            "Defaults to ./credit_memo_annotation_v1.py"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for annotation output JSON. "
            "Defaults to <memo-payload-stem>__annotation_test.json"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat input warnings or failed location validation as errors.",
    )
    args = parser.parse_args()

    memo_path = Path(args.memo_payload).expanduser().resolve()
    map_path = Path(args.document_map).expanduser().resolve()
    module_path = Path(args.annotation_module).expanduser().resolve()

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else memo_path.with_name(f"{memo_path.stem}__annotation_test.json")
    )

    print("Loading inputs...")
    print(f"  Memo payload:      {memo_path}")
    print(f"  Document map:      {map_path}")
    print(f"  Annotation module: {module_path}")
    print(f"  Output:            {output_path}")

    memo_payload = load_json(memo_path)
    document_map = load_json(map_path)

    warnings = validate_inputs(memo_payload, document_map)
    if warnings:
        print("\nInput warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        if args.strict:
            raise RuntimeError("Strict mode stopped because input warnings were found.")

    annotator = load_annotator(module_path)

    print("\nRunning revised annotation engine...")
    result = annotator(
        memo_payload=memo_payload,
        document_map=document_map,
    )

    if not isinstance(result, dict):
        raise TypeError(
            "annotate_credit_memo() returned "
            f"{type(result).__name__}; expected dict"
        )

    write_json(output_path, result)

    summary = annotation_summary(result)
    print("\nTest summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print_annotation_locations(result)

    validation = result.get("validation") or {}
    if args.strict and validation.get("status") != "passed":
        raise RuntimeError(
            "Strict mode stopped because annotation location validation failed."
        )

    print(f"\nAnnotation output written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
