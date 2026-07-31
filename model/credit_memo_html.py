from __future__ import annotations

"""HTML renderer for parsed Capital Benchmark credit memos.

The canonical source remains the parsed document map. This module creates a
safe, self-contained HTML fragment for display inside Bubble's HTML element.

Version 1.1 adds optional annotation rendering:
- a memo/review two-column layout;
- annotation badges beside linked memo blocks;
- a review findings panel;
- click-to-scroll and hover highlighting;
- graceful handling of findings that do not yet contain block identifiers.

All memo and annotation content is HTML-escaped.
"""

from html import escape
from typing import Any, Mapping, Sequence

HTML_RENDERER_VERSION = "1.2.1"

DEFAULT_CSS = r"""
.cb-review-shell {
  --cb-text: #172033;
  --cb-muted: #667085;
  --cb-border: #e4e7ec;
  --cb-surface: #ffffff;
  --cb-soft: #f8fafc;
  --cb-heading: #101828;
  --cb-critical: #b42318;
  --cb-critical-soft: #fef3f2;
  --cb-high: #d92d20;
  --cb-high-soft: #fef3f2;
  --cb-medium: #dc6803;
  --cb-medium-soft: #fffaeb;
  --cb-low: #175cd3;
  --cb-low-soft: #eff8ff;
  --cb-info: #475467;
  --cb-info-soft: #f2f4f7;
  color: var(--cb-text);
  background: transparent;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.62;
  overflow-wrap: anywhere;
}
.cb-review-shell * { box-sizing: border-box; }


.cb-page-controls {
  display: grid;
  gap: 12px;
  margin: 0 0 22px;
  padding: 14px 16px;
  border: 1px solid var(--cb-border);
  border-radius: 12px;
  background: var(--cb-surface);
  box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
}
.cb-page-controls__row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  align-items: end;
}
.cb-page-controls__field {
  display: grid;
  gap: 4px;
  min-width: 150px;
}
.cb-page-controls__label {
  color: var(--cb-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.cb-page-controls__select {
  min-height: 36px;
  padding: 7px 32px 7px 10px;
  border: 1px solid #d0d5dd;
  border-radius: 8px;
  background: #fff;
  color: var(--cb-text);
  font: inherit;
  font-size: 12px;
}
.cb-page-controls__nav {
  display: flex;
  gap: 8px;
  margin-left: auto;
}
.cb-page-controls__link {
  display: inline-flex;
  min-height: 36px;
  align-items: center;
  justify-content: center;
  padding: 7px 11px;
  border: 1px solid #d0d5dd;
  border-radius: 8px;
  background: #fff;
  color: #344054;
  font-size: 12px;
  font-weight: 650;
  text-decoration: none;
}
.cb-page-controls__link:hover { background: #f9fafb; }
.cb-page-controls__link[aria-disabled="true"] {
  cursor: default;
  opacity: .45;
  pointer-events: none;
}
.cb-page-controls__note {
  margin: 0;
  color: var(--cb-muted);
  font-size: 11px;
}

.cb-score-strip {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0;
  margin: 0 0 22px;
  overflow: hidden;
  border: 1px solid var(--cb-border);
  border-radius: 12px;
  background: var(--cb-surface);
  box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
}
.cb-score-card {
  min-width: 0;
  padding: 14px 16px;
  border-right: 1px solid var(--cb-border);
  background: transparent;
}
.cb-score-card:last-child { border-right: 0; }
.cb-score-card__label {
  display: block;
  margin-bottom: 4px;
  color: var(--cb-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .03em;
  text-transform: uppercase;
}
.cb-score-card__value {
  display: block;
  color: var(--cb-heading);
  font-size: 19px;
  line-height: 1.2;
  font-weight: 750;
}
.cb-score-card__value--muted {
  color: var(--cb-muted);
  font-size: 14px;
  font-weight: 650;
}

.cb-review-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 360px);
  gap: 28px;
  align-items: start;
}
.cb-review-layout--memo-only { display: block; }

.cb-memo {
  min-width: 0;
  padding: 38px 42px 48px;
  border: 1px solid var(--cb-border);
  border-radius: 14px;
  color: var(--cb-text);
  background: var(--cb-surface);
  box-shadow: 0 10px 30px rgba(16, 24, 40, 0.07);
}
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
  transition:
    background-color 140ms ease,
    box-shadow 140ms ease,
    border-color 140ms ease;
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
.cb-block--rule {
  border: 0;
  border-top: 1px solid var(--cb-border);
  margin: 22px 0;
}

.cb-block.has-annotations {
  margin-left: -12px;
  padding-left: 12px;
  border-left: 3px solid var(--cb-medium);
  border-radius: 0 6px 6px 0;
  background: #fffdf8;
}
.cb-block.has-annotations[data-max-severity="critical"],
.cb-block.has-annotations[data-max-severity="high"] {
  border-left-color: var(--cb-high);
  background: #fffafa;
}
.cb-block.has-annotations[data-max-severity="low"] {
  border-left-color: var(--cb-low);
  background: #fbfdff;
}
.cb-block.has-annotations[data-max-severity="info"] {
  border-left-color: #98a2b3;
  background: #fcfcfd;
}
.cb-block.is-selected,
.cb-block[data-highlighted="true"] {
  border-radius: 6px;
  background: #fff7e8;
  box-shadow: 0 0 0 4px #fff7e8;
}

.cb-block__annotation-gutter {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin: 7px 0 2px;
}
.cb-block__annotation-link {
  appearance: none;
  border: 1px solid transparent;
  border-radius: 999px;
  padding: 3px 8px;
  cursor: pointer;
  font: inherit;
  font-size: 11px;
  font-weight: 650;
  line-height: 1.35;
}
.cb-block__annotation-link:hover,
.cb-block__annotation-link:focus-visible {
  filter: brightness(0.97);
  outline: 2px solid #84adff;
  outline-offset: 1px;
}

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

.cb-review {
  position: sticky;
  top: 18px;
  max-height: calc(100vh - 36px);
  overflow: auto;
  border: 1px solid var(--cb-border);
  border-radius: 14px;
  background: var(--cb-surface);
  box-shadow: 0 10px 30px rgba(16, 24, 40, 0.07);
}
.cb-review__header {
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 16px 16px 12px;
  border-bottom: 1px solid var(--cb-border);
  background: rgba(255, 255, 255, 0.96);
  backdrop-filter: blur(5px);
}
.cb-review__title {
  margin: 0;
  color: var(--cb-heading);
  font-size: 17px;
  line-height: 1.35;
  font-weight: 700;
}
.cb-review__summary {
  margin: 4px 0 0;
  color: var(--cb-muted);
  font-size: 12px;
}
.cb-review__list {
  display: grid;
  gap: 10px;
  padding: 12px;
}
.cb-annotation {
  width: 100%;
  border: 1px solid var(--cb-border);
  border-radius: 10px;
  background: var(--cb-surface);
  overflow: hidden;
  scroll-margin-top: 18px;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}
.cb-annotation.is-selected {
  border-color: #84adff;
  box-shadow: 0 0 0 3px #d1e9ff;
}
.cb-annotation__button {
  display: block;
  width: 100%;
  padding: 12px;
  border: 0;
  background: transparent;
  color: inherit;
  text-align: left;
  cursor: pointer;
  font: inherit;
}
.cb-annotation__button:hover { background: #f9fafb; }
.cb-annotation__button:focus-visible {
  outline: 2px solid #84adff;
  outline-offset: -2px;
}
.cb-annotation__topline {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.cb-annotation__id {
  margin-left: auto;
  color: #98a2b3;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}
.cb-annotation__title {
  margin: 0;
  color: var(--cb-heading);
  font-size: 13px;
  line-height: 1.4;
  font-weight: 650;
}
.cb-annotation__detail {
  margin: 6px 0 0;
  color: #475467;
  font-size: 12px;
  line-height: 1.45;
}
.cb-annotation__meta {
  margin: 8px 0 0;
  color: var(--cb-muted);
  font-size: 10px;
}
.cb-annotation--unlinked .cb-annotation__meta::before {
  content: "Not linked to a memo block · ";
}
.cb-review__empty {
  padding: 18px;
  color: var(--cb-muted);
  font-size: 13px;
  text-align: center;
}

.cb-severity {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 3px 7px;
  font-size: 10px;
  font-weight: 700;
  line-height: 1.25;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}
.cb-severity--critical,
.cb-severity--high {
  color: var(--cb-high);
  background: var(--cb-high-soft);
}
.cb-severity--medium {
  color: var(--cb-medium);
  background: var(--cb-medium-soft);
}
.cb-severity--low {
  color: var(--cb-low);
  background: var(--cb-low-soft);
}
.cb-severity--info,
.cb-severity--unknown {
  color: var(--cb-info);
  background: var(--cb-info-soft);
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

@media (max-width: 980px) {
  .cb-review-layout {
    grid-template-columns: minmax(0, 1fr) minmax(250px, 310px);
    gap: 20px;
  }
}
@media (max-width: 760px) {
  .cb-review-shell { font-size: 14px; }
  .cb-review-layout { display: block; }
  .cb-memo {
    padding: 26px 20px 36px;
    border-radius: 12px;
  }
  .cb-review {
    position: static;
    max-height: none;
    margin-top: 20px;
    border-radius: 12px;
  }
  .cb-score-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .cb-score-card {
    border-right: 1px solid var(--cb-border);
    border-bottom: 1px solid var(--cb-border);
  }
  .cb-score-card:nth-child(2n) { border-right: 0; }
  .cb-score-card:last-child { border-bottom: 0; }
  .cb-page-controls__field { min-width: min(100%, 180px); flex: 1 1 150px; }
  .cb-page-controls__nav { width: 100%; margin-left: 0; }
  .cb-page-controls__link { flex: 1; }
  .cb-memo__title { font-size: 23px; }
  h2.cb-section__heading { font-size: 19px; }
  .cb-table th, .cb-table td { padding: 8px 9px; }
}
""".strip()

DEFAULT_JS = r"""
(function () {
  function findShell(node) {
    return node && node.closest ? node.closest(".cb-review-shell") : null;
  }

  function clearSelection(shell) {
    if (!shell) return;
    shell.querySelectorAll(".is-selected").forEach(function (node) {
      node.classList.remove("is-selected");
    });
  }

  function selectPair(shell, annotationId, targetId, shouldScroll) {
    if (!shell) return;
    clearSelection(shell);

    var annotation = annotationId
      ? shell.querySelector('[data-annotation-id="' + CSS.escape(annotationId) + '"]')
      : null;
    var target = targetId
      ? shell.querySelector("#" + CSS.escape(targetId))
      : null;

    if (annotation) annotation.classList.add("is-selected");
    if (target) target.classList.add("is-selected");

    if (shouldScroll) {
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
      } else if (annotation) {
        annotation.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
  }

  document.addEventListener("click", function (event) {
    var trigger = event.target.closest("[data-cb-annotation-trigger]");
    if (!trigger) return;

    event.preventDefault();
    var shell = findShell(trigger);
    selectPair(
      shell,
      trigger.getAttribute("data-annotation-id"),
      trigger.getAttribute("data-target-id"),
      true
    );
  });

  document.addEventListener("mouseover", function (event) {
    var trigger = event.target.closest("[data-cb-annotation-trigger]");
    if (!trigger) return;

    var shell = findShell(trigger);
    selectPair(
      shell,
      trigger.getAttribute("data-annotation-id"),
      trigger.getAttribute("data-target-id"),
      false
    );
  });

  document.addEventListener("mouseout", function (event) {
    var trigger = event.target.closest("[data-cb-annotation-trigger]");
    if (!trigger) return;

    var related = event.relatedTarget;
    if (related && trigger.contains(related)) return;
    clearSelection(findShell(trigger));
  });

  document.addEventListener("change", function (event) {
    var select = event.target.closest("[data-cb-variant-select]");
    if (!select) return;
    var destination = select.value;
    if (destination) window.location.href = destination;
  });
})();
""".strip()


def _attr(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _text(value: Any) -> str:
    return escape("" if value is None else str(value), quote=False)


def _class_token(value: Any) -> str:
    raw = "" if value is None else str(value).strip().lower()
    token = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)
    return token.strip("-") or "unknown"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _normalise_rows(value: Any) -> list[list[Any]]:
    if not _is_sequence(value):
        return []
    rows: list[list[Any]] = []
    for row in value:
        if _is_sequence(row):
            rows.append(list(row))
    return rows


def _normalise_annotations(value: Any) -> list[dict[str, Any]]:
    """Accept either an annotation payload or a direct annotation list."""
    if isinstance(value, Mapping):
        value = value.get("annotations")

    if not _is_sequence(value):
        return []

    annotations: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            annotations.append(dict(item))
    return annotations


def _annotation_target(annotation: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return (target kind, target identifier) for a finding.

    The renderer accepts identifiers either directly or inside ``location`` or
    ``target`` mappings. A string location is treated as an identifier only
    when it resembles a canonical block/section id.
    """
    direct_fields = (
        ("block", annotation.get("block_id")),
        ("block_uuid", annotation.get("block_uuid")),
        ("section", annotation.get("section_id")),
    )
    for kind, value in direct_fields:
        if value not in (None, ""):
            return kind, str(value)

    for container_name in ("target", "location"):
        container = annotation.get(container_name)
        if isinstance(container, Mapping):
            for kind, key in (
                ("block", "block_id"),
                ("block_uuid", "block_uuid"),
                ("section", "section_id"),
            ):
                value = container.get(key)
                if value not in (None, ""):
                    return kind, str(value)

    location = annotation.get("location")
    if isinstance(location, str):
        value = location.strip()
        lowered = value.lower()
        if lowered.startswith(("block-", "block_", "section-", "section_")):
            return ("section" if lowered.startswith("section") else "block"), value

    return None, None


def _target_html_id(
    annotation: Mapping[str, Any],
    *,
    block_id_by_uuid: Mapping[str, str],
    known_element_ids: set[str],
) -> str | None:
    kind, identifier = _annotation_target(annotation)
    if not identifier:
        return None

    if kind == "block_uuid":
        return block_id_by_uuid.get(identifier)

    return identifier if identifier in known_element_ids else None


def _severity_rank(value: Any) -> int:
    return {
        "critical": 5,
        "high": 4,
        "medium": 3,
        "low": 2,
        "info": 1,
    }.get(_class_token(value), 0)


def _max_severity(annotations: Sequence[Mapping[str, Any]]) -> str:
    if not annotations:
        return "unknown"
    return max(
        (_class_token(item.get("severity")) for item in annotations),
        key=_severity_rank,
        default="unknown",
    )


def _severity_label(annotation: Mapping[str, Any]) -> str:
    severity = str(annotation.get("severity") or "info").strip()
    return severity.title()


def _annotation_id(annotation: Mapping[str, Any], index: int) -> str:
    value = annotation.get("annotation_id")
    return str(value) if value not in (None, "") else f"annotation-{index + 1}"


def _render_table(block: Mapping[str, Any]) -> str:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), Mapping) else {}
    headers = list(metadata.get("headers") or [])
    rows = _normalise_rows(metadata.get("rows"))

    if not headers and not rows:
        return f'<div class="cb-empty">{_text(block.get("text") or "Table unavailable")}</div>'

    width = max([len(headers), *(len(row) for row in rows)], default=0)
    if width == 0:
        return '<div class="cb-empty">Empty table</div>'

    if len(headers) < width:
        headers.extend([""] * (width - len(headers)))

    head_html = "".join(f'<th scope="col">{_text(cell)}</th>' for cell in headers)
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


def _block_attributes(
    block: Mapping[str, Any],
    *,
    annotations: Sequence[Mapping[str, Any]] = (),
) -> str:
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
        "data-annotation-count": len(annotations) if annotations else None,
        "data-max-severity": _max_severity(annotations) if annotations else None,
    }
    return " ".join(
        f'{key}="{_attr(value)}"' for key, value in attrs.items() if value is not None
    )


def _render_block_annotation_gutter(
    annotations: Sequence[Mapping[str, Any]],
    *,
    target_id: str,
    annotation_index: Mapping[int, str],
) -> str:
    if not annotations:
        return ""

    parts: list[str] = []
    for annotation in annotations:
        annotation_id = annotation_index[id(annotation)]
        severity = _class_token(annotation.get("severity"))
        title = str(annotation.get("title") or annotation.get("public_label") or "Review finding")
        parts.append(
            f'<button type="button" '
            f'class="cb-block__annotation-link cb-severity cb-severity--{severity}" '
            f'data-cb-annotation-trigger="true" '
            f'data-annotation-id="{_attr(annotation_id)}" '
            f'data-target-id="{_attr(target_id)}" '
            f'title="{_attr(title)}">'
            f'{_text(_severity_label(annotation))}'
            f"</button>"
        )

    return f'<div class="cb-block__annotation-gutter">{"".join(parts)}</div>'


def render_block(
    block: Mapping[str, Any],
    *,
    annotations: Sequence[Mapping[str, Any]] = (),
    annotation_index: Mapping[int, str] | None = None,
) -> str:
    """Render one parsed block as safe HTML."""
    annotation_index = annotation_index or {}
    block_type = str(block.get("block_type") or "paragraph")
    token = _class_token(block_type).replace("_", "-")
    block_id = str(block.get("block_id") or "")
    classes = f"cb-block cb-block--{token}"
    if annotations:
        classes += " has-annotations"

    attrs = _block_attributes(block, annotations=annotations)
    content = _text(block.get("text"))
    gutter = (
        _render_block_annotation_gutter(
            annotations,
            target_id=block_id,
            annotation_index=annotation_index,
        )
        if block_id
        else ""
    )

    if block_type == "table":
        return f'<div class="{classes}" {attrs}>{_render_table(block)}{gutter}</div>'
    if block_type == "horizontal_rule":
        return f'<div class="{classes}" {attrs}><hr class="cb-block--rule">{gutter}</div>'
    if block_type == "code_block":
        return f'<pre class="{classes}" {attrs}><code>{content}</code>{gutter}</pre>'
    if block_type == "quote":
        return f'<blockquote class="{classes}" {attrs}>{content}{gutter}</blockquote>'
    if block_type == "numbered_item":
        number = block.get("order") or 1
        return (
            f'<div class="{classes}" data-list-number="{_attr(number)}" '
            f'{attrs}>{content}{gutter}</div>'
        )
    if block_type == "bullet":
        return f'<div class="{classes}" {attrs}>{content}{gutter}</div>'

    return f'<div class="{classes}" {attrs}>{content}{gutter}</div>'


def render_section(
    section: Mapping[str, Any],
    *,
    annotations_by_target: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    annotation_index: Mapping[int, str] | None = None,
) -> str:
    """Render one parsed section, retaining canonical identifiers."""
    annotations_by_target = annotations_by_target or {}
    annotation_index = annotation_index or {}

    section_id = str(section.get("section_id") or "section")
    section_type = section.get("section_type") or "other"
    level_raw = section.get("level")
    try:
        level = int(level_raw)
    except (TypeError, ValueError):
        level = 2

    title = str(section.get("title") or "")
    blocks = section.get("blocks") if _is_sequence(section.get("blocks")) else []
    block_html = "".join(
        render_block(
            block,
            annotations=annotations_by_target.get(str(block.get("block_id") or ""), ()),
            annotation_index=annotation_index,
        )
        for block in blocks
        if isinstance(block, Mapping)
    )

    heading_html = ""
    if title and not (level == 0 and section_type == "document_introduction"):
        heading_level = min(max(level, 2), 6)
        heading_html = (
            f'<h{heading_level} class="cb-section__heading">{_text(title)}</h{heading_level}>'
        )

    return (
        f'<section id="{_attr(section_id)}" '
        f'class="cb-section cb-section--{_class_token(section_type)}" '
        f'data-section-id="{_attr(section_id)}" '
        f'data-section-type="{_attr(section_type)}" '
        f'data-section-order="{_attr(section.get("order"))}">'
        f"{heading_html}{block_html}"
        "</section>"
    )


def _document_identifiers(
    document_map: Mapping[str, Any],
) -> tuple[set[str], dict[str, str]]:
    known_ids: set[str] = set()
    block_id_by_uuid: dict[str, str] = {}

    sections = document_map.get("sections")
    if not _is_sequence(sections):
        return known_ids, block_id_by_uuid

    for section in sections:
        if not isinstance(section, Mapping):
            continue

        section_id = section.get("section_id")
        if section_id not in (None, ""):
            known_ids.add(str(section_id))

        blocks = section.get("blocks")
        if not _is_sequence(blocks):
            continue

        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            block_id = block.get("block_id")
            block_uuid = block.get("block_uuid")
            if block_id not in (None, ""):
                block_id_string = str(block_id)
                known_ids.add(block_id_string)
                if block_uuid not in (None, ""):
                    block_id_by_uuid[str(block_uuid)] = block_id_string

    return known_ids, block_id_by_uuid


def _prepare_annotation_rendering(
    document_map: Mapping[str, Any],
    annotations: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[int, str],
    dict[int, str | None],
]:
    normalised = _normalise_annotations(annotations)
    known_ids, block_id_by_uuid = _document_identifiers(document_map)

    annotations_by_target: dict[str, list[dict[str, Any]]] = {}
    annotation_index: dict[int, str] = {}
    target_index: dict[int, str | None] = {}

    for index, annotation in enumerate(normalised):
        annotation_id = _annotation_id(annotation, index)
        target_id = _target_html_id(
            annotation,
            block_id_by_uuid=block_id_by_uuid,
            known_element_ids=known_ids,
        )
        annotation_index[id(annotation)] = annotation_id
        target_index[id(annotation)] = target_id
        if target_id:
            annotations_by_target.setdefault(target_id, []).append(annotation)

    return normalised, annotations_by_target, annotation_index, target_index


def _render_annotation_card(
    annotation: Mapping[str, Any],
    *,
    annotation_id: str,
    target_id: str | None,
) -> str:
    severity = _class_token(annotation.get("severity"))
    title = annotation.get("title") or annotation.get("public_label") or "Review finding"
    detail = annotation.get("detail") or ""
    category = annotation.get("category") or ""
    policy_id = annotation.get("policy_id")

    metadata = " · ".join(
        str(value)
        for value in (category, policy_id)
        if value not in (None, "")
    )
    linked_class = "" if target_id else " cb-annotation--unlinked"
    target_attribute = f'data-target-id="{_attr(target_id)}"' if target_id else ""

    detail_html = (
        f'<p class="cb-annotation__detail">{_text(detail)}</p>' if detail else ""
    )
    meta_html = (
        f'<p class="cb-annotation__meta">{_text(metadata)}</p>' if metadata else ""
    )

    return (
        f'<article id="annotation-{_attr(annotation_id)}" '
        f'class="cb-annotation{linked_class}" '
        f'data-annotation-id="{_attr(annotation_id)}" '
        f'data-severity="{_attr(severity)}">'
        f'<button type="button" class="cb-annotation__button" '
        f'data-cb-annotation-trigger="true" '
        f'data-annotation-id="{_attr(annotation_id)}" {target_attribute}>'
        f'<div class="cb-annotation__topline">'
        f'<span class="cb-severity cb-severity--{severity}">'
        f'{_text(_severity_label(annotation))}</span>'
        f'<span class="cb-annotation__id">{_text(annotation_id)}</span>'
        f"</div>"
        f'<h3 class="cb-annotation__title">{_text(title)}</h3>'
        f"{detail_html}{meta_html}"
        f"</button>"
        f"</article>"
    )


def render_annotation_panel(
    annotations: Sequence[Mapping[str, Any]],
    *,
    annotation_index: Mapping[int, str],
    target_index: Mapping[int, str | None],
) -> str:
    """Render the right-hand review panel."""
    if not annotations:
        return (
            '<aside class="cb-review" aria-label="Review findings">'
            '<div class="cb-review__header">'
            '<h2 class="cb-review__title">Review Findings</h2>'
            '<p class="cb-review__summary">No annotations</p>'
            "</div>"
            '<div class="cb-review__empty">No review findings were generated.</div>'
            "</aside>"
        )

    linked_count = sum(1 for annotation in annotations if target_index[id(annotation)])
    cards = "".join(
        _render_annotation_card(
            annotation,
            annotation_id=annotation_index[id(annotation)],
            target_id=target_index[id(annotation)],
        )
        for annotation in annotations
    )

    return (
        '<aside class="cb-review" aria-label="Review findings">'
        '<div class="cb-review__header">'
        '<h2 class="cb-review__title">Review Findings</h2>'
        f'<p class="cb-review__summary">{len(annotations)} findings · '
        f'{linked_count} linked to memo blocks</p>'
        "</div>"
        f'<div class="cb-review__list">{cards}</div>'
        "</aside>"
    )



def _score_value(value: Any, *, suffix: str = "") -> tuple[str, bool]:
    if value is None:
        return "Not scored", True
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{suffix}", False


def render_score_strip(annotation_payload: Mapping[str, Any] | None) -> str:
    """Render a compact summary of the evaluated dimensions."""
    if not isinstance(annotation_payload, Mapping):
        return ""

    scores = annotation_payload.get("scores")
    if not isinstance(scores, Mapping):
        return ""

    cards = [
        ("Overall", scores.get("overall_score"), ""),
        ("Policy", scores.get("policy_detection_score"), ""),
        ("Missing info", scores.get("missing_information_detection_score"), ""),
        ("Material issues", scores.get("critical_or_high_issue_count"), ""),
        ("Findings", scores.get("annotation_count"), ""),
    ]

    parts: list[str] = []
    for label, raw_value, suffix in cards:
        value, muted = _score_value(raw_value, suffix=suffix)
        value_class = " cb-score-card__value--muted" if muted else ""
        parts.append(
            '<div class="cb-score-card">'
            f'<span class="cb-score-card__label">{_text(label)}</span>'
            f'<span class="cb-score-card__value{value_class}">{_text(value)}</span>'
            "</div>"
        )

    return (
        '<section class="cb-score-strip" aria-label="Benchmark summary scores">'
        f'{"".join(parts)}'
        "</section>"
    )


def _variant_label(value: Any) -> str:
    raw = str(value or "").replace("_", " ").strip()
    return raw.title() if raw else "—"


def render_variant_controls(navigation: Mapping[str, Any] | None) -> str:
    """Render static-page navigation controls for benchmark variants."""
    if not isinstance(navigation, Mapping):
        return ""

    dimensions = navigation.get("dimensions")
    if not isinstance(dimensions, Mapping):
        dimensions = {}

    fields: list[str] = []
    field_order = (
        ("model", "Model"),
        ("context_mode", "Context"),
        ("policy_mode", "Policy"),
        ("prompt_mode", "Prompt"),
        ("run", "Run"),
    )

    for key, label in field_order:
        options = dimensions.get(key)
        if not _is_sequence(options) or not options:
            continue

        option_html: list[str] = []
        for option in options:
            if not isinstance(option, Mapping):
                continue
            value = option.get("value")
            href = option.get("href")
            selected = bool(option.get("selected"))
            option_html.append(
                f'<option value="{_attr(href)}"'
                f'{" selected" if selected else ""}>'
                f'{_text(_variant_label(value))}</option>'
            )

        if option_html:
            fields.append(
                '<label class="cb-page-controls__field">'
                f'<span class="cb-page-controls__label">{_text(label)}</span>'
                f'<select class="cb-page-controls__select" '
                f'data-cb-variant-select="{_attr(key)}">'
                f'{"".join(option_html)}'
                "</select>"
                "</label>"
            )

    previous_href = navigation.get("previous_href")
    next_href = navigation.get("next_href")
    nav_html = (
        '<div class="cb-page-controls__nav">'
        f'<a class="cb-page-controls__link" '
        f'href="{_attr(previous_href or "#")}" '
        f'aria-disabled="{str(not bool(previous_href)).lower()}">Previous variation</a>'
        f'<a class="cb-page-controls__link" '
        f'href="{_attr(next_href or "#")}" '
        f'aria-disabled="{str(not bool(next_href)).lower()}">Next variation</a>'
        "</div>"
    )

    note = navigation.get("note") or (
        "Scores reflect only the information and policies available to the model "
        "in this experiment."
    )

    return (
        '<nav class="cb-page-controls" aria-label="Benchmark variation controls">'
        f'<div class="cb-page-controls__row">{"".join(fields)}{nav_html}</div>'
        f'<p class="cb-page-controls__note">{_text(note)}</p>'
        "</nav>"
    )

def render_document_map_to_html(
    document_map: Mapping[str, Any],
    *,
    annotations: Any = None,
    navigation: Mapping[str, Any] | None = None,
    include_title: bool = True,
    include_styles: bool = True,
    include_scripts: bool = True,
    include_annotation_panel: bool = True,
    css: str | None = None,
    javascript: str | None = None,
) -> str:
    """Return a Bubble-ready HTML fragment from a parsed document map.

    ``annotations`` may be either:
    - an annotation result payload containing an ``annotations`` list; or
    - the annotation list itself.

    Existing callers remain compatible because annotations are optional.
    """
    sections = document_map.get("sections")
    if not _is_sequence(sections):
        raise ValueError("document_map.sections must be a list")

    annotation_payload = annotations if isinstance(annotations, Mapping) else None
    (
        normalised_annotations,
        annotations_by_target,
        annotation_index,
        target_index,
    ) = _prepare_annotation_rendering(document_map, annotations)

    memo_id = document_map.get("memo_id") or ""
    title = str(document_map.get("document_title") or "Credit Memo")
    parser_version = document_map.get("parser_version") or ""

    style_html = (
        f"<style>{css if css is not None else DEFAULT_CSS}</style>"
        if include_styles
        else ""
    )
    script_html = (
        f"<script>{javascript if javascript is not None else DEFAULT_JS}</script>"
        if include_scripts and normalised_annotations
        else ""
    )
    title_html = (
        f'<header class="cb-memo__header">'
        f'<h1 class="cb-memo__title">{_text(title)}</h1>'
        f"</header>"
        if include_title
        else ""
    )
    sections_html = "".join(
        render_section(
            section,
            annotations_by_target=annotations_by_target,
            annotation_index=annotation_index,
        )
        for section in sections
        if isinstance(section, Mapping)
    )

    memo_html = (
        f'<article class="cb-memo" data-memo-id="{_attr(memo_id)}" '
        f'data-parser-version="{_attr(parser_version)}" '
        f'data-html-renderer-version="{_attr(HTML_RENDERER_VERSION)}">'
        f"{title_html}{sections_html}"
        "</article>"
    )

    panel_html = (
        render_annotation_panel(
            normalised_annotations,
            annotation_index=annotation_index,
            target_index=target_index,
        )
        if include_annotation_panel and normalised_annotations
        else ""
    )
    layout_class = (
        "cb-review-layout"
        if panel_html
        else "cb-review-layout cb-review-layout--memo-only"
    )
    controls_html = render_variant_controls(navigation)
    score_html = render_score_strip(annotation_payload)

    return (
        f"{style_html}"
        f'<div class="cb-review-shell" '
        f'data-annotation-count="{len(normalised_annotations)}">'
        f"{controls_html}{score_html}"
        f'<div class="{layout_class}">{memo_html}{panel_html}</div>'
        f"</div>"
        f"{script_html}"
    )


def build_html_payload(
    document_map: Mapping[str, Any],
    *,
    annotations: Any = None,
    navigation: Mapping[str, Any] | None = None,
    include_title: bool = True,
    include_styles: bool = True,
    include_scripts: bool = True,
    include_annotation_panel: bool = True,
) -> dict[str, Any]:
    """Build a JSON-serialisable API payload for Bubble."""
    normalised_annotations = _normalise_annotations(annotations)
    return {
        "memo_id": document_map.get("memo_id"),
        "document_title": document_map.get("document_title"),
        "parser_version": document_map.get("parser_version"),
        "html_renderer_version": HTML_RENDERER_VERSION,
        "annotation_count": len(normalised_annotations),
        "memo_html": render_document_map_to_html(
            document_map,
            annotations=annotations,
            navigation=navigation,
            include_title=include_title,
            include_styles=include_styles,
            include_scripts=include_scripts,
            include_annotation_panel=include_annotation_panel,
        ),
    }
