from __future__ import annotations

"""HTML renderer for parsed Capital Benchmark credit memos.

The canonical source remains the parsed document map. This module creates a
safe, self-contained HTML fragment for display inside Bubble's HTML element.
Every section and block receives stable IDs/data attributes so later releases
can add scroll-to-block and highlighting without changing the payload shape.
"""

from html import escape
from typing import Any, Iterable, Mapping, Sequence

HTML_RENDERER_VERSION = "1.0.0"

DEFAULT_CSS = r"""
.cb-memo {
  --cb-text: #172033;
  --cb-muted: #667085;
  --cb-border: #e4e7ec;
  --cb-surface: #ffffff;
  --cb-soft: #f8fafc;
  --cb-heading: #101828;
  color: var(--cb-text);
  background: var(--cb-surface);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.62;
  overflow-wrap: anywhere;
}
.cb-memo * { box-sizing: border-box; }
.cb-memo__header { margin: 0 0 28px; }
.cb-memo__title {
  margin: 0;
  color: var(--cb-heading);
  font-size: 28px;
  line-height: 1.22;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.cb-section { margin: 0 0 30px; scroll-margin-top: 24px; }
.cb-section__heading {
  margin: 0 0 12px;
  color: var(--cb-heading);
  font-weight: 700;
  letter-spacing: -0.012em;
}
h2.cb-section__heading { font-size: 21px; line-height: 1.3; }
h3.cb-section__heading { font-size: 17px; line-height: 1.35; }
h4.cb-section__heading,
h5.cb-section__heading,
h6.cb-section__heading { font-size: 15px; line-height: 1.4; }
.cb-block {
  position: relative;
  margin: 0 0 12px;
  scroll-margin-top: 24px;
  transition: background-color 120ms ease, box-shadow 120ms ease;
}
.cb-block:last-child { margin-bottom: 0; }
.cb-block--paragraph { white-space: normal; }
.cb-block--bullet,
.cb-block--numbered-item { padding-left: 24px; }
.cb-block--bullet::before {
  content: "•";
  position: absolute;
  left: 7px;
  color: var(--cb-muted);
}
.cb-block--numbered-item::before {
  content: attr(data-list-number) ".";
  position: absolute;
  left: 0;
  color: var(--cb-muted);
  font-variant-numeric: tabular-nums;
}
.cb-block--quote {
  margin-left: 0;
  padding: 10px 14px;
  border-left: 3px solid #98a2b3;
  background: var(--cb-soft);
  color: #344054;
}
.cb-block--code {
  margin: 0 0 12px;
  padding: 14px;
  overflow-x: auto;
  border: 1px solid var(--cb-border);
  border-radius: 8px;
  background: var(--cb-soft);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre;
}
.cb-block--rule { border: 0; border-top: 1px solid var(--cb-border); margin: 22px 0; }
.cb-table-wrap {
  margin: 4px 0 16px;
  overflow-x: auto;
  border: 1px solid var(--cb-border);
  border-radius: 10px;
}
.cb-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.cb-table th,
.cb-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--cb-border);
  text-align: left;
  vertical-align: top;
}
.cb-table th {
  background: var(--cb-soft);
  color: #344054;
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.025em;
  text-transform: uppercase;
}
.cb-table tr:last-child td { border-bottom: 0; }
.cb-table td:first-child { color: #344054; font-weight: 550; }
.cb-empty { color: var(--cb-muted); font-style: italic; }
.cb-block.is-selected,
.cb-block[data-highlighted="true"] {
  border-radius: 6px;
  background: #fff7e8;
  box-shadow: 0 0 0 4px #fff7e8;
}
.cb-insertion-marker {
  display: none;
  margin: 8px 0 14px;
  padding: 10px 12px;
  border: 1px dashed #f79009;
  border-radius: 8px;
  background: #fffaeb;
  color: #7a2e0e;
  font-size: 13px;
}
.cb-insertion-marker.is-visible { display: block; }
@media (max-width: 720px) {
  .cb-memo { font-size: 14px; }
  .cb-memo__title { font-size: 23px; }
  h2.cb-section__heading { font-size: 19px; }
  .cb-table th, .cb-table td { padding: 8px 9px; }
}
""".strip()


def _attr(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _text(value: Any) -> str:
    return escape("" if value is None else str(value), quote=False)


def _class_token(value: Any) -> str:
    raw = "" if value is None else str(value).strip().lower()
    token = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)
    return token.strip("-") or "unknown"


def _normalise_rows(value: Any) -> list[list[Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[list[Any]] = []
    for row in value:
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            rows.append(list(row))
    return rows


def _render_table(block: Mapping[str, Any]) -> str:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), Mapping) else {}
    headers = list(metadata.get("headers") or [])
    rows = _normalise_rows(metadata.get("rows"))

    # A parser should normally supply structured metadata. The fallback keeps
    # malformed/legacy maps visible without trying to parse pipe-delimited text.
    if not headers and not rows:
        return f'<div class="cb-empty">{_text(block.get("text") or "Table unavailable")}</div>'

    width = max([len(headers), *(len(row) for row in rows)], default=0)
    if width == 0:
        return '<div class="cb-empty">Empty table</div>'

    if len(headers) < width:
        headers.extend([""] * (width - len(headers)))

    head_html = "".join(f"<th scope=\"col\">{_text(cell)}</th>" for cell in headers)
    body_parts: list[str] = []
    for row in rows:
        padded = row + [""] * (width - len(row))
        cells = "".join(f"<td>{_text(cell)}</td>" for cell in padded[:width])
        body_parts.append(f"<tr>{cells}</tr>")

    return (
        '<div class="cb-table-wrap">'
        '<table class="cb-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_parts)}</tbody>"
        "</table>"
        "</div>"
    )


def _block_attributes(block: Mapping[str, Any]) -> str:
    lines = block.get("source_lines") if isinstance(block.get("source_lines"), Mapping) else {}
    attrs = {
        "id": block.get("block_id"),
        "data-block-id": block.get("block_id"),
        "data-block-uuid": block.get("block_uuid"),
        "data-section-id": block.get("section_id"),
        "data-block-type": block.get("block_type"),
        "data-document-order": block.get("document_order"),
        "data-source-line-start": lines.get("start"),
        "data-source-line-end": lines.get("end"),
        "data-text-sha256": block.get("text_sha256"),
    }
    return " ".join(f'{key}="{_attr(value)}"' for key, value in attrs.items() if value is not None)


def render_block(block: Mapping[str, Any]) -> str:
    """Render one parsed block as safe HTML."""
    block_type = str(block.get("block_type") or "paragraph")
    token = _class_token(block_type).replace("_", "-")
    attrs = _block_attributes(block)
    content = _text(block.get("text"))

    if block_type == "table":
        return f'<div class="cb-block cb-block--table" {attrs}>{_render_table(block)}</div>'
    if block_type == "horizontal_rule":
        return f'<hr class="cb-block cb-block--rule" {attrs}>'
    if block_type == "code_block":
        return f'<pre class="cb-block cb-block--code" {attrs}><code>{content}</code></pre>'
    if block_type == "quote":
        return f'<blockquote class="cb-block cb-block--quote" {attrs}>{content}</blockquote>'
    if block_type == "numbered_item":
        number = block.get("order") or 1
        return (
            f'<div class="cb-block cb-block--numbered-item" data-list-number="{_attr(number)}" '
            f'{attrs}>{content}</div>'
        )
    if block_type == "bullet":
        return f'<div class="cb-block cb-block--bullet" {attrs}>{content}</div>'

    return f'<div class="cb-block cb-block--{token}" {attrs}>{content}</div>'


def render_section(section: Mapping[str, Any]) -> str:
    """Render one parsed section, retaining canonical identifiers."""
    section_id = section.get("section_id") or "section"
    section_type = section.get("section_type") or "other"
    level_raw = section.get("level")
    try:
        level = int(level_raw)
    except (TypeError, ValueError):
        level = 2

    title = str(section.get("title") or "")
    blocks = section.get("blocks") if isinstance(section.get("blocks"), Sequence) else []
    block_html = "".join(render_block(block) for block in blocks if isinstance(block, Mapping))

    # The parser's synthetic Document Introduction has level 0. It is useful
    # structurally, but its heading need not be shown to end users.
    heading_html = ""
    if title and not (level == 0 and section_type == "document_introduction"):
        heading_level = min(max(level, 2), 6)
        heading_html = (
            f'<h{heading_level} class="cb-section__heading">{_text(title)}</h{heading_level}>'
        )

    return (
        f'<section id="{_attr(section_id)}" class="cb-section cb-section--{_class_token(section_type)}" '
        f'data-section-id="{_attr(section_id)}" data-section-type="{_attr(section_type)}" '
        f'data-section-order="{_attr(section.get("order"))}">'
        f"{heading_html}{block_html}"
        "</section>"
    )


def render_document_map_to_html(
    document_map: Mapping[str, Any],
    *,
    include_title: bool = True,
    include_styles: bool = True,
    css: str | None = None,
) -> str:
    """Return a Bubble-ready HTML fragment from a parsed document map.

    All memo content is HTML-escaped. The optional CSS is trusted application
    code supplied by the backend developer, not user content.
    """
    sections = document_map.get("sections")
    if not isinstance(sections, Sequence):
        raise ValueError("document_map.sections must be a list")

    memo_id = document_map.get("memo_id") or ""
    title = str(document_map.get("document_title") or "Credit Memo")
    parser_version = document_map.get("parser_version") or ""

    style_html = f"<style>{css if css is not None else DEFAULT_CSS}</style>" if include_styles else ""
    title_html = (
        f'<header class="cb-memo__header"><h1 class="cb-memo__title">{_text(title)}</h1></header>'
        if include_title
        else ""
    )
    sections_html = "".join(
        render_section(section) for section in sections if isinstance(section, Mapping)
    )

    return (
        f"{style_html}"
        f'<article class="cb-memo" data-memo-id="{_attr(memo_id)}" '
        f'data-parser-version="{_attr(parser_version)}" '
        f'data-html-renderer-version="{_attr(HTML_RENDERER_VERSION)}">'
        f"{title_html}{sections_html}"
        "</article>"
    )


def build_html_payload(
    document_map: Mapping[str, Any],
    *,
    include_title: bool = True,
    include_styles: bool = True,
) -> dict[str, Any]:
    """Build a JSON-serialisable API payload for Bubble."""
    return {
        "memo_id": document_map.get("memo_id"),
        "document_title": document_map.get("document_title"),
        "parser_version": document_map.get("parser_version"),
        "html_renderer_version": HTML_RENDERER_VERSION,
        "memo_html": render_document_map_to_html(
            document_map,
            include_title=include_title,
            include_styles=include_styles,
        ),
    }
