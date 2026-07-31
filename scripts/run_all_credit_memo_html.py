from __future__ import annotations

"""Batch-generate HTML files from parsed Capital Benchmark document maps.

Example
-------
python scripts/run_all_credit_memo_html.py \
  --benchmark-dir benchmark_runs/benchmark_20_mini_memos \
  --renderer-module model/credit_memo_html.py \
  --overwrite

By default, matching JSON files from <benchmark-dir>/annotations are passed
to the renderer. Use --no-annotations to generate memo-only HTML.
"""

import argparse
import hashlib
import html
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping


# The batch runner may live either in the project root or in scripts/.
# Add the project root to sys.path so the shared frontend renderer can
# be imported reliably in both layouts.
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "frontend").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.index_page import render_benchmark_index


RUN_SCHEMA = "credit_memo_batch_html_run"
RUN_SCHEMA_VERSION = "1.2.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(module_path: Path, module_name: str) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")

    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_nonempty(*values: Any, default: str = "") -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def parse_filename_metadata(filename: str) -> dict[str, str]:
    """Extract useful display metadata from the benchmark filename."""
    stem = Path(filename).stem
    parts = stem.split("__")

    result = {
        "sequence": "",
        "company": "",
        "context_mode": "",
        "policy_mode": "",
        "prompt_mode": "",
        "model_tier": "",
        "run": "",
    }

    if parts:
        result["sequence"] = parts[0]
    if len(parts) > 1:
        result["company"] = parts[1]

    for part in parts[2:]:
        if part.startswith("ctx_"):
            result["context_mode"] = part.removeprefix("ctx_")
        elif part.startswith("policy_"):
            result["policy_mode"] = part.removeprefix("policy_")
        elif part.startswith("prompt_"):
            result["prompt_mode"] = part.removeprefix("prompt_")
        elif part.startswith("tier_"):
            result["model_tier"] = part.removeprefix("tier_")
        elif part.startswith("run_"):
            result["run"] = part.removeprefix("run_")

    return result


def annotation_summary(annotation_path: Path) -> dict[str, Any]:
    """Read optional annotation metadata for the static index."""
    if not annotation_path.exists():
        return {}

    try:
        payload = read_json(annotation_path)
    except Exception:
        return {}

    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        scores = {}

    benchmark_metadata = payload.get("benchmark_metadata")
    if not isinstance(benchmark_metadata, Mapping):
        benchmark_metadata = {}

    return {
        "overall_score": scores.get("overall_score"),
        "annotation_count": scores.get("annotation_count"),
        "location_resolution_score": scores.get("location_resolution_score"),
        "validation_status": (
            payload.get("validation", {}).get("status")
            if isinstance(payload.get("validation"), Mapping)
            else None
        ),
        "model": benchmark_metadata.get("model"),
    }


def full_html_document(
    *,
    fragment: str,
    title: str,
    memo_id: str,
    renderer_version: str,
) -> str:
    """Wrap the Bubble-ready fragment in a standalone HTML document."""
    safe_title = html.escape(title or "Credit Memo")
    safe_memo_id = html.escape(memo_id, quote=True)
    safe_version = html.escape(renderer_version, quote=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="Capital Benchmark HTML renderer {safe_version}">
  <title>{safe_title}</title>
  <style>
    html {{ background: #f2f4f7; }}
    body {{
      margin: 0;
      padding: 32px 20px 64px;
    }}
    .cb-preview-shell {{
      width: min(1040px, 100%);
      margin: 0 auto;
      padding: 40px 48px;
      border: 1px solid #e4e7ec;
      border-radius: 14px;
      background: #ffffff;
      box-shadow: 0 10px 30px rgba(16, 24, 40, 0.08);
    }}
    .cb-preview-meta {{
      width: min(1040px, 100%);
      margin: 0 auto 10px;
      color: #667085;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 12px;
    }}
    @media (max-width: 720px) {{
      body {{ padding: 0; }}
      .cb-preview-shell {{
        padding: 24px 18px 48px;
        border: 0;
        border-radius: 0;
        box-shadow: none;
      }}
      .cb-preview-meta {{ display: none; }}
    }}
  </style>
</head>
<body data-memo-id="{safe_memo_id}">
  <div class="cb-preview-meta">Memo ID: {safe_memo_id}</div>
  <main class="cb-preview-shell">
    {fragment}
  </main>
</body>
</html>
"""


def build_index_html(results: list[dict[str, Any]], renderer_version: str) -> str:
    successful = [item for item in results if item.get("status") == "success"]

    rows: list[str] = []
    for item in successful:
        filename_meta = item.get("filename_metadata") or {}
        score = item.get("overall_score")
        score_text = "—" if score is None else str(score)
        annotation_count = item.get("annotation_count")
        annotation_text = "—" if annotation_count is None else str(annotation_count)

        searchable = " ".join(
            str(value or "")
            for value in (
                item.get("memo_id"),
                item.get("document_title"),
                filename_meta.get("company"),
                filename_meta.get("context_mode"),
                filename_meta.get("policy_mode"),
                filename_meta.get("prompt_mode"),
                item.get("model"),
            )
        ).lower()

        rows.append(
            f"""
            <tr data-search="{html.escape(searchable, quote=True)}">
              <td>{html.escape(str(filename_meta.get("sequence") or ""))}</td>
              <td>{html.escape(str(filename_meta.get("company") or ""))}</td>
              <td class="title">{html.escape(str(item.get("document_title") or ""))}</td>
              <td>{html.escape(str(filename_meta.get("context_mode") or ""))}</td>
              <td>{html.escape(str(filename_meta.get("policy_mode") or ""))}</td>
              <td>{html.escape(str(filename_meta.get("prompt_mode") or ""))}</td>
              <td>{html.escape(str(item.get("model") or filename_meta.get("model_tier") or ""))}</td>
              <td>{score_text}</td>
              <td>{annotation_text}</td>
              <td><a href="{html.escape(str(item["html_file_name"]), quote=True)}">Open</a></td>
            </tr>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Capital Benchmark Memo HTML Index</title>
  <style>
    :root {{
      color: #172033;
      background: #f2f4f7;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 32px; }}
    main {{
      width: min(1500px, 100%);
      margin: 0 auto;
      padding: 28px;
      border: 1px solid #e4e7ec;
      border-radius: 14px;
      background: white;
      box-shadow: 0 10px 30px rgba(16, 24, 40, 0.07);
    }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    .sub {{ margin: 0 0 22px; color: #667085; }}
    input {{
      width: min(520px, 100%);
      margin: 0 0 18px;
      padding: 11px 13px;
      border: 1px solid #d0d5dd;
      border-radius: 8px;
      font: inherit;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{
      padding: 10px 9px;
      border-bottom: 1px solid #eaecf0;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f8fafc;
      color: #475467;
      font-size: 11px;
      letter-spacing: .03em;
      text-transform: uppercase;
    }}
    td.title {{ min-width: 260px; white-space: normal; }}
    a {{ color: #175cd3; font-weight: 600; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .count {{ margin-left: 10px; color: #667085; font-size: 13px; }}
    @media (max-width: 720px) {{
      body {{ padding: 0; }}
      main {{ padding: 18px; border: 0; border-radius: 0; box-shadow: none; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Capital Benchmark Memo HTML Index</h1>
  <p class="sub">
    Renderer {html.escape(renderer_version)} ·
    <span id="visible-count">{len(successful)}</span> of {len(successful)} memos shown
  </p>
  <input id="search" type="search" placeholder="Search company, title, policy mode, model…">
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Company</th>
          <th>Document</th>
          <th>Context</th>
          <th>Policy</th>
          <th>Prompt</th>
          <th>Model</th>
          <th>Score</th>
          <th>Findings</th>
          <th></th>
        </tr>
      </thead>
      <tbody id="memo-rows">
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</main>
<script>
  const input = document.getElementById("search");
  const rows = Array.from(document.querySelectorAll("#memo-rows tr"));
  const visibleCount = document.getElementById("visible-count");

  function filterRows() {{
    const query = input.value.trim().toLowerCase();
    let visible = 0;

    for (const row of rows) {{
      const match = !query || row.dataset.search.includes(query);
      row.hidden = !match;
      if (match) visible += 1;
    }}

    visibleCount.textContent = String(visible);
  }}

  input.addEventListener("input", filterRows);
</script>
</body>
</html>
"""


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    document_map_dir = (
        Path(args.document_map_dir).expanduser().resolve()
        if args.document_map_dir
        else benchmark_dir / "document_maps"
    )
    annotation_dir = (
        Path(args.annotation_dir).expanduser().resolve()
        if args.annotation_dir
        else benchmark_dir / "annotations"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else benchmark_dir / "html"
    )
    renderer_module = Path(args.renderer_module).expanduser().resolve()

    return {
        "benchmark_dir": benchmark_dir,
        "document_map_dir": document_map_dir,
        "annotation_dir": annotation_dir,
        "output_dir": output_dir,
        "renderer_module": renderer_module,
    }


def select_files(
    document_map_dir: Path,
    *,
    limit: int | None,
    contains: str | None,
) -> list[Path]:
    files = sorted(path for path in document_map_dir.glob("*.json") if path.is_file())

    if contains:
        needle = contains.lower()
        files = [path for path in files if needle in path.name.lower()]

    if limit is not None:
        files = files[:limit]

    return files



def build_variant_navigation(
    files: list[Path],
    *,
    current_path: Path,
    annotation_dir: Path,
) -> dict[str, Any]:
    """Build links among pre-generated HTML variants for one company.

    Each dropdown changes one experimental dimension while preserving the
    current values of the other dimensions where an exact variant exists.
    """
    records: list[dict[str, Any]] = []
    for path in files:
        metadata = parse_filename_metadata(path.name)
        annotation_path = annotation_dir / path.name
        summary = annotation_summary(annotation_path)
        records.append(
            {
                "path": path,
                "href": f"{path.stem}.html",
                **metadata,
                "model": summary.get("model") or metadata.get("model_tier") or "",
            }
        )

    current = next((item for item in records if item["path"] == current_path), None)
    if current is None:
        return {}

    company_records = [
        item for item in records
        if item.get("company") == current.get("company")
    ]
    company_records.sort(key=lambda item: (
        item.get("model", ""),
        item.get("context_mode", ""),
        item.get("policy_mode", ""),
        item.get("prompt_mode", ""),
        item.get("run", ""),
        item.get("sequence", ""),
    ))

    dimensions: dict[str, list[dict[str, Any]]] = {}
    keys = ("model", "context_mode", "policy_mode", "prompt_mode", "run")

    for key in keys:
        values = sorted({
            str(item.get(key) or "")
            for item in company_records
            if str(item.get(key) or "")
        })
        options: list[dict[str, Any]] = []

        for value in values:
            candidates = [
                item for item in company_records
                if str(item.get(key) or "") == value
            ]

            # Prefer an exact match on all dimensions other than the one being
            # changed. Fall back to the first available variant for the value.
            exact = next(
                (
                    item for item in candidates
                    if all(
                        str(item.get(other) or "") == str(current.get(other) or "")
                        for other in keys
                        if other != key
                    )
                ),
                None,
            )
            destination = exact or (candidates[0] if candidates else None)
            if destination:
                options.append(
                    {
                        "value": value,
                        "href": destination["href"],
                        "selected": value == str(current.get(key) or ""),
                    }
                )

        if options:
            dimensions[key] = options

    current_index = company_records.index(current)
    previous_href = (
        company_records[current_index - 1]["href"]
        if current_index > 0
        else None
    )
    next_href = (
        company_records[current_index + 1]["href"]
        if current_index + 1 < len(company_records)
        else None
    )

    return {
        "dimensions": dimensions,
        "previous_href": previous_href,
        "next_href": next_href,
        "note": (
            "Scores reflect only the information and policies available to the "
            "model in this experiment."
        ),
    }

def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args)
    benchmark_dir = paths["benchmark_dir"]
    document_map_dir = paths["document_map_dir"]
    annotation_dir = paths["annotation_dir"]
    output_dir = paths["output_dir"]
    renderer_module_path = paths["renderer_module"]

    if not document_map_dir.exists():
        raise FileNotFoundError(f"Document-map directory not found: {document_map_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    renderer_module = load_module(renderer_module_path, "_capital_benchmark_html_renderer")
    render_fn = getattr(renderer_module, args.renderer_function, None)
    if not callable(render_fn):
        raise AttributeError(
            f"Renderer function {args.renderer_function!r} not found in {renderer_module_path}"
        )

    renderer_version = first_nonempty(
        getattr(renderer_module, "HTML_RENDERER_VERSION", None),
        default="unknown",
    )

    files = select_files(
        document_map_dir,
        limit=args.limit,
        contains=args.contains,
    )

    started_at = utc_now_iso()
    started_perf = time.perf_counter()
    results: list[dict[str, Any]] = []

    for index, document_map_path in enumerate(files, start=1):
        item_started = time.perf_counter()
        output_path = output_dir / f"{document_map_path.stem}.html"
        result: dict[str, Any] = {
            "document_map_file": str(document_map_path),
            "html_file": str(output_path),
            "html_file_name": output_path.name,
            "status": "pending",
            "error_type": None,
            "error_message": None,
        }

        try:
            document_map = read_json(document_map_path)
            memo_id = first_nonempty(
                document_map.get("memo_id"),
                document_map_path.stem,
            )
            title = first_nonempty(
                document_map.get("document_title"),
                default="Credit Memo",
            )

            result.update(
                {
                    "memo_id": memo_id,
                    "document_title": title,
                    "parser_version": document_map.get("parser_version"),
                    "filename_metadata": parse_filename_metadata(document_map_path.name),
                }
            )

            annotation_path = annotation_dir / document_map_path.name
            result.update(annotation_summary(annotation_path))

            annotation_payload: dict[str, Any] | None = None
            if not args.no_annotations and annotation_path.exists():
                annotation_payload = read_json(annotation_path)

            result["annotation_file"] = (
                str(annotation_path) if annotation_path.exists() else None
            )
            result["annotations_rendered"] = annotation_payload is not None

            if output_path.exists() and not args.overwrite:
                result["status"] = "skipped_existing"
            else:
                navigation = (
                    build_variant_navigation(
                        files,
                        current_path=document_map_path,
                        annotation_dir=annotation_dir,
                    )
                    if not args.no_navigation
                    else None
                )

                fragment = render_fn(
                    document_map,
                    annotations=annotation_payload,
                    navigation=navigation,
                    include_title=not args.no_title,
                    include_styles=True,
                )
                if not isinstance(fragment, str) or not fragment.strip():
                    raise ValueError("Renderer returned empty or non-string HTML")

                standalone = full_html_document(
                    fragment=fragment,
                    title=title,
                    memo_id=memo_id,
                    renderer_version=renderer_version,
                )
                write_text(output_path, standalone)

                result.update(
                    {
                        "status": "success",
                        "html_sha256": sha256_text(standalone),
                        "html_bytes": len(standalone.encode("utf-8")),
                    }
                )

        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            if args.fail_fast:
                results.append(result)
                raise

        finally:
            result["elapsed_seconds"] = round(time.perf_counter() - item_started, 4)
            results.append(result)

        if args.verbose:
            print(
                f"[{index:>3}/{len(files)}] "
                f"{result.get('status', 'unknown'):<16} "
                f"{document_map_path.name}"
            )

    elapsed = time.perf_counter() - started_perf
    success_count = sum(item["status"] == "success" for item in results)
    skipped_count = sum(item["status"] == "skipped_existing" for item in results)
    failed_count = sum(item["status"] == "failed" for item in results)

    manifest = {
        "run_schema": RUN_SCHEMA,
        "run_schema_version": RUN_SCHEMA_VERSION,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now_iso(),
        "elapsed_seconds": round(elapsed, 4),
        "benchmark_directory": str(benchmark_dir),
        "document_map_directory": str(document_map_dir),
        "annotation_directory": str(annotation_dir),
        "annotations_enabled": not args.no_annotations,
        "variant_navigation_enabled": not args.no_navigation,
        "html_directory": str(output_dir),
        "renderer_module": str(renderer_module_path),
        "renderer_function": args.renderer_function,
        "html_renderer_version": renderer_version,
        "selected_file_count": len(files),
        "processed_file_count": len(results),
        "success_count": success_count,
        "skipped_existing_count": skipped_count,
        "failed_count": failed_count,
        "results": results,
    }

    manifest_path = output_dir / "_html_manifest.json"
    write_json(manifest_path, manifest)

    index_path = output_dir / "index.html"
    if not args.no_index:
        # Include both newly generated and already-existing memo pages.
        # This prevents the index from becoming empty when the batch runner
        # is executed again without --overwrite.
        index_records = [
            item
            for item in results
            if item.get("status") in {"success", "skipped_existing"}
        ]
        write_text(
            index_path,
            render_benchmark_index(index_records),
        )

    print()
    print("HTML generation complete")
    print(f"  Selected:  {len(files)}")
    print(f"  Generated: {success_count}")
    print(f"  Skipped:   {skipped_count}")
    print(f"  Failed:    {failed_count}")
    print(f"  Elapsed:   {elapsed:.3f}s")
    print(f"  Output:    {output_dir}")
    print(f"  Manifest:  {manifest_path}")
    if not args.no_index:
        print(f"  Index:     {index_path}")

    return 1 if failed_count else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate standalone HTML files from parsed credit-memo document maps."
    )
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help="Benchmark root containing document_maps/ and annotations/.",
    )
    parser.add_argument(
        "--document-map-dir",
        help="Override the input document-map directory.",
    )
    parser.add_argument(
        "--annotation-dir",
        help=(
            "Override the annotation directory used for rendered review findings "
            "and index metadata."
        ),
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Render memo HTML without annotation findings or the review panel.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to <benchmark-dir>/html.",
    )
    parser.add_argument(
        "--renderer-module",
        default="model/credit_memo_html.py",
        help="Path to the HTML renderer Python module.",
    )
    parser.add_argument(
        "--renderer-function",
        default="render_document_map_to_html",
        help="Renderer function name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate HTML files that already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate only the first N matching files.",
    )
    parser.add_argument(
        "--contains",
        help="Only process document-map filenames containing this text.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not render the document title inside the memo fragment.",
    )
    parser.add_argument(
        "--no-navigation",
        action="store_true",
        help="Do not render model/context/policy/prompt/run navigation controls.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not generate the searchable index.html.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one document fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one status line per document.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
