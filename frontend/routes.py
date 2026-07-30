from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from flask import abort, make_response, send_from_directory
from werkzeug.utils import safe_join

from .index_page import render_benchmark_index


ALLOWED_FRONTEND_SUFFIXES = {
    ".html",
    ".css",
    ".js",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".map",
}

_MEMOS_PATTERN = re.compile(
    r"const\s+MEMOS\s*=\s*(\[.*?\])\s*;",
    re.DOTALL,
)


def _discover_frontends(root: Path) -> dict[str, Path]:
    """Discover model-specific HTML directories."""
    frontends: dict[str, Path] = {}

    if (root / "index.html").is_file():
        frontends[root.name or "default"] = root

    if (root / "html" / "index.html").is_file():
        frontends[root.name or "default"] = root / "html"

    for child in sorted(root.iterdir()):
        html_dir = child / "html"

        if child.is_dir() and (html_dir / "index.html").is_file():
            frontends[child.name] = html_dir

    return frontends


def _load_index_records(
    run_slug: str,
    html_dir: Path,
) -> list[dict[str, Any]]:
    """
    Read the memo records embedded by run_all_credit_memo_html.py and
    prefix each memo link with its model/run slug.
    """
    index_path = html_dir / "index.html"

    try:
        source = index_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Could not read benchmark index: {index_path}"
        ) from exc

    match = _MEMOS_PATTERN.search(source)

    if match is None:
        raise RuntimeError(
            f"Could not find the embedded MEMOS array in {index_path}"
        )

    try:
        records = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid embedded MEMOS JSON in {index_path}: {exc}"
        ) from exc

    normalised_records: list[dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue

        item = dict(record)
        filename = Path(str(item.get("href", "")).strip()).name

        if not filename:
            continue

        item["href"] = f"memos/{run_slug}/{filename}"
        item["benchmark_run"] = run_slug
        normalised_records.append(item)

    return normalised_records


def register_frontend_routes(app):
    html_root = Path(app.config["HTML_DIR"]).expanduser().resolve()
    frontend_dirs = _discover_frontends(html_root)

    if not frontend_dirs:
        raise RuntimeError(
            f"No benchmark HTML frontends were found under {html_root}"
        )

    def resolve_frontend_file(
        run_slug: str,
        filename: str,
    ) -> tuple[Path, Path]:
        html_dir = frontend_dirs.get(run_slug)

        if html_dir is None:
            abort(
                404,
                description=f"Unknown benchmark run: {run_slug}",
            )

        requested_filename = str(filename).strip()

        if not requested_filename:
            abort(
                400,
                description="A frontend filename is required.",
            )

        resolved_value = safe_join(
            str(html_dir),
            requested_filename,
        )

        if resolved_value is None:
            abort(
                400,
                description="Invalid frontend file path.",
            )

        resolved_path = Path(resolved_value)

        if resolved_path.suffix.lower() not in ALLOWED_FRONTEND_SUFFIXES:
            abort(
                403,
                description="Unsupported frontend file type.",
            )

        if not resolved_path.is_file():
            abort(
                404,
                description=f"Frontend file not found: {requested_filename}",
            )

        return html_dir, resolved_path

    @app.get("/")
    def frontend_home():
        records: list[dict[str, Any]] = []

        for run_slug, html_dir in frontend_dirs.items():
            records.extend(_load_index_records(run_slug, html_dir))

        page = render_benchmark_index(records)
        response = make_response(page)
        response.mimetype = "text/html"
        response.headers["Cache-Control"] = "public, max-age=300"
        response.headers["X-Content-Type-Options"] = "nosniff"

        return response

    @app.get("/memos/<run_slug>/<path:filename>")
    def frontend_memo(run_slug: str, filename: str):
        html_dir, requested_path = resolve_frontend_file(
            run_slug,
            filename,
        )

        relative_path = requested_path.relative_to(html_dir)

        response = send_from_directory(
            directory=str(html_dir),
            path=str(relative_path),
            conditional=True,
            max_age=3600,
        )

        response.headers["X-Content-Type-Options"] = "nosniff"

        return response
