from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _json_for_script(value: Any) -> str:
    """
    Serialize data safely for embedding inside a <script> element.
    """
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    # Prevent strings inside the data from prematurely closing the script.
    return serialized.replace("</", "<\\/")


def render_benchmark_index(manifest: dict[str, Any] | list[dict[str, Any]]) -> str:
    """
    Render the static Credit Memo Benchmark Explorer index page.

    The page:
    - displays a Capital Benchmark header;
    - provides client-side filters;
    - displays a live results count;
    - links to existing memo HTML files.

    The function accepts either:
    - a manifest containing an `items` or `memos` list; or
    - a list of memo records directly.
    """

    if isinstance(manifest, list):
        raw_records = manifest
    else:
        raw_records = (
            manifest.get("results")
            or manifest.get("items")
            or manifest.get("memos")
            or manifest.get("records")
            or manifest.get("documents")
            or []
        )

    records = [
        _normalise_manifest_record(record)
        for record in raw_records
        if isinstance(record, dict)
    ]

    records_json = _json_for_script(records)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <title>Credit Memo Benchmark Explorer</title>

    <style>
        :root {{
            --navy: #173d70;
            --navy-dark: #12325d;
            --navy-light: #285b96;
            --page-background: #f5f7fa;
            --panel-background: #ffffff;
            --text-primary: #161b22;
            --text-secondary: #67738a;
            --border: #dce2ea;
            --border-strong: #cbd3df;
            --row-hover: #f5f8fc;
            --focus-ring: rgba(23, 61, 112, 0.18);
            --shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            --radius: 14px;
            --content-width: 1680px;
        }}

        * {{
            box-sizing: border-box;
        }}

        html {{
            min-height: 100%;
        }}

        body {{
            margin: 0;
            min-height: 100%;
            background: var(--page-background);
            color: var(--text-primary);
            font-family:
                Inter,
                ui-sans-serif,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
            font-size: 16px;
            line-height: 1.5;
        }}

        button,
        select,
        input {{
            font: inherit;
        }}

        a {{
            color: inherit;
        }}

        .site-header {{
            background: var(--navy);
            color: #ffffff;
        }}

        .site-header__inner {{
            width: min(var(--content-width), calc(100% - 40px));
            min-height: 78px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 32px;
        }}

        .brand {{
            display: flex;
            flex-direction: column;
            text-decoration: none;
            color: #ffffff;
        }}

        .brand__name {{
            display: flex;
            align-items: center;
            gap: 2px;
            font-size: 23px;
            line-height: 1.05;
            letter-spacing: 0.01em;
        }}

        .brand__mark {{
            display: inline-flex;
            width: 25px;
            height: 30px;
            align-items: center;
            justify-content: center;
            margin-right: 1px;
            border-radius: 4px;
            background: #ffffff;
            color: var(--navy);
            font-size: 22px;
            font-weight: 800;
        }}

        .brand__capital {{
            font-weight: 750;
        }}

        .brand__benchmark {{
            font-weight: 350;
            color: #d9e4f2;
        }}

        .brand__tagline {{
            margin-top: 3px;
            font-size: 14px;
            color: #eef4fb;
        }}

        .site-nav {{
            display: flex;
            align-items: center;
            gap: 30px;
        }}

        .site-nav a {{
            color: #ffffff;
            text-decoration: none;
            font-size: 16px;
        }}

        .site-nav a:hover {{
            text-decoration: underline;
            text-underline-offset: 4px;
        }}

        .page-shell {{
            width: min(var(--content-width), calc(100% - 40px));
            margin: 0 auto;
            padding: 30px 0 60px;
        }}

        .page-heading {{
            margin-bottom: 26px;
        }}

        .page-heading h1 {{
            margin: 0;
            font-size: clamp(28px, 2vw, 34px);
            line-height: 1.2;
            letter-spacing: -0.02em;
        }}

        .page-heading p {{
            margin: 4px 0 0;
            color: var(--text-secondary);
            font-size: 19px;
        }}

        .filter-panel {{
            padding: 24px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--panel-background);
            box-shadow: var(--shadow);
        }}

        .filter-panel__heading {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 22px;
        }}

        .filter-panel__heading h2 {{
            margin: 0;
            font-size: 23px;
            line-height: 1.2;
        }}

        .reset-button {{
            appearance: none;
            border: 0;
            background: transparent;
            color: var(--navy);
            cursor: pointer;
            font-weight: 650;
            padding: 6px 0;
        }}

        .reset-button:hover {{
            text-decoration: underline;
            text-underline-offset: 3px;
        }}

        .filter-grid {{
            display: grid;
            grid-template-columns:
                minmax(190px, 1.45fr)
                minmax(160px, 0.9fr)
                minmax(145px, 0.8fr)
                minmax(180px, 1fr)
                minmax(210px, 1.2fr);
            gap: 16px;
            align-items: end;
        }}

        .filter-field {{
            min-width: 0;
        }}

        .filter-field label {{
            display: block;
            margin-bottom: 6px;
            font-size: 14px;
            font-weight: 650;
        }}

        .filter-field select {{
            width: 100%;
            height: 54px;
            padding: 0 42px 0 15px;
            border: 1px solid var(--border-strong);
            border-radius: 5px;
            background-color: #ffffff;
            color: var(--text-primary);
            cursor: pointer;
            outline: none;
        }}

        .filter-field select:focus {{
            border-color: var(--navy-light);
            box-shadow: 0 0 0 4px var(--focus-ring);
        }}

        .results-toolbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            margin: 28px 0 18px;
        }}

        .results-count {{
            margin: 0;
            font-size: 18px;
        }}

        .table-panel {{
            overflow: hidden;
            border: 1px solid var(--border);
            background: var(--panel-background);
            box-shadow: var(--shadow);
        }}

        .memo-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}

        .memo-table th {{
            padding: 16px 22px;
            border-bottom: 1px solid var(--border);
            background: #ffffff;
            color: var(--text-primary);
            font-size: 14px;
            font-weight: 700;
            text-align: left;
        }}

        .memo-table td {{
            padding: 19px 22px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}

        .memo-table tbody tr {{
            cursor: pointer;
            transition: background-color 120ms ease;
        }}

        .memo-table tbody tr:hover {{
            background: var(--row-hover);
        }}

        .memo-table tbody tr:focus {{
            outline: 3px solid var(--focus-ring);
            outline-offset: -3px;
        }}

        .memo-table tbody tr:last-child td {{
            border-bottom: 0;
        }}

        .memo-table__company {{
            width: 25%;
        }}

        .memo-table__model {{
            width: 15%;
        }}

        .memo-table__prompt {{
            width: 12%;
        }}

        .memo-table__context {{
            width: 15%;
        }}

        .memo-table__evaluation {{
            width: 18%;
        }}

        .memo-table__issues {{
            width: 9%;
        }}

        .memo-table__open {{
            width: 6%;
            text-align: right;
        }}

        .open-arrow {{
            display: inline-flex;
            width: 30px;
            height: 30px;
            align-items: center;
            justify-content: center;
            color: var(--navy);
            font-size: 27px;
            font-weight: 300;
            line-height: 1;
        }}

        .empty-state {{
            padding: 64px 24px;
            text-align: center;
            color: var(--text-secondary);
        }}

        .load-more-wrap {{
            display: flex;
            justify-content: center;
            padding: 24px 0 0;
        }}

        .load-more-button {{
            min-width: 150px;
            padding: 11px 20px;
            border: 1px solid var(--navy);
            border-radius: 6px;
            background: #ffffff;
            color: var(--navy);
            cursor: pointer;
            font-weight: 650;
        }}

        .load-more-button:hover {{
            background: var(--navy);
            color: #ffffff;
        }}

        .load-more-button[hidden] {{
            display: none;
        }}

        @media (max-width: 1050px) {{
            .filter-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}

            .memo-table__context,
            .memo-table__evaluation {{
                display: none;
            }}

            .memo-table__company {{
                width: 32%;
            }}

            .memo-table__model {{
                width: 23%;
            }}

            .memo-table__prompt {{
                width: 18%;
            }}

            .memo-table__issues {{
                width: 15%;
            }}

            .memo-table__open {{
                width: 12%;
            }}
        }}

        @media (max-width: 700px) {{
            .site-header__inner,
            .page-shell {{
                width: min(100% - 24px, var(--content-width));
            }}

            .site-header__inner {{
                min-height: 72px;
            }}

            .site-nav {{
                display: none;
            }}

            .page-shell {{
                padding-top: 24px;
            }}

            .page-heading p {{
                font-size: 16px;
            }}

            .filter-panel {{
                padding: 18px;
            }}

            .filter-grid {{
                grid-template-columns: 1fr;
            }}

            .table-panel {{
                overflow-x: auto;
            }}

            .memo-table {{
                min-width: 700px;
            }}
        }}
    </style>
</head>

<body>
    <header class="site-header">
        <div class="site-header__inner">
            <a class="brand" href="/">
                <span class="brand__name">
                    <span class="brand__mark">C</span>
                    <span class="brand__capital">apital</span>
                    <span class="brand__benchmark">Benchmark</span>
                </span>

                <span class="brand__tagline">
                    Credit Memo Benchmark Explorer
                </span>
            </a>

            <nav class="site-nav" aria-label="Primary navigation">
                <a href="/methodology">Methodology</a>
                <a href="/about">About</a>
            </nav>
        </div>
    </header>

    <main class="page-shell">
        <section class="page-heading">
            <h1>Explore benchmark credit memos</h1>

            <p>
                Compare AI-generated credit memos across companies,
                models, prompts and experimental configurations.
            </p>
        </section>

        <section class="filter-panel" aria-labelledby="filters-heading">
            <div class="filter-panel__heading">
                <h2 id="filters-heading">Filters</h2>

                <button
                    class="reset-button"
                    id="reset-filters"
                    type="button"
                >
                    Reset
                </button>
            </div>

            <div class="filter-grid">
                <div class="filter-field">
                    <label for="filter-company">Company</label>
                    <select id="filter-company">
                        <option value="">All Companies</option>
                    </select>
                </div>

                <div class="filter-field">
                    <label for="filter-model">Model</label>
                    <select id="filter-model">
                        <option value="">All Models</option>
                    </select>
                </div>

                <div class="filter-field">
                    <label for="filter-prompt">Prompt</label>
                    <select id="filter-prompt">
                        <option value="">All Prompts</option>
                    </select>
                </div>

                <div class="filter-field">
                    <label for="filter-context">Financial Context</label>
                    <select id="filter-context">
                        <option value="">All Context</option>
                    </select>
                </div>

                <div class="filter-field">
                    <label for="filter-evaluation">Policy Evaluation</label>
                    <select id="filter-evaluation">
                        <option value="">All Evaluations</option>
                    </select>
                </div>
            </div>
        </section>

        <div class="results-toolbar">
            <p class="results-count" id="results-count"></p>
        </div>

        <section class="table-panel">
            <table class="memo-table">
                <thead>
                    <tr>
                        <th class="memo-table__company">Company</th>
                        <th class="memo-table__model">Model</th>
                        <th class="memo-table__prompt">Prompt</th>
                        <th class="memo-table__context">Context</th>
                        <th class="memo-table__evaluation">Evaluation</th>
                        <th class="memo-table__issues">Issues</th>
                        <th
                            class="memo-table__open"
                            aria-label="Open memo"
                        ></th>
                    </tr>
                </thead>

                <tbody id="memo-table-body"></tbody>
            </table>

            <div
                class="empty-state"
                id="empty-state"
                hidden
            >
                No benchmark credit memos match these filters.
            </div>
        </section>

        <div class="load-more-wrap">
            <button
                class="load-more-button"
                id="load-more"
                type="button"
                hidden
            >
                Show more
            </button>
        </div>
    </main>

    <script>
        "use strict";

        const MEMOS = {records_json};

        const INITIAL_PAGE_SIZE = 50;
        const PAGE_INCREMENT = 50;

        const state = {{
            visibleCount: INITIAL_PAGE_SIZE,
            filteredMemos: [...MEMOS],
        }};

        const elements = {{
            company: document.getElementById("filter-company"),
            model: document.getElementById("filter-model"),
            prompt: document.getElementById("filter-prompt"),
            context: document.getElementById("filter-context"),
            evaluation: document.getElementById("filter-evaluation"),
            reset: document.getElementById("reset-filters"),
            count: document.getElementById("results-count"),
            tableBody: document.getElementById("memo-table-body"),
            emptyState: document.getElementById("empty-state"),
            loadMore: document.getElementById("load-more"),
        }};

        function normaliseValue(value) {{
            return String(value ?? "").trim();
        }}

        function memoHref(value) {{
            const raw = normaliseValue(value);
            return raw.split("/").filter(Boolean).pop() || "";
        }}

        function uniqueSortedValues(fieldName) {{
            return [
                ...new Set(
                    MEMOS
                        .map((memo) => normaliseValue(memo[fieldName]))
                        .filter(Boolean)
                )
            ].sort((left, right) =>
                left.localeCompare(right, undefined, {{
                    numeric: true,
                    sensitivity: "base",
                }})
            );
        }}

        function addOptions(selectElement, values) {{
            const fragment = document.createDocumentFragment();

            values.forEach((value) => {{
                const option = document.createElement("option");
                option.value = value;
                option.textContent = value;
                fragment.appendChild(option);
            }});

            selectElement.appendChild(fragment);
        }}

        function initialiseFilters() {{
            addOptions(elements.company, uniqueSortedValues("company"));
            addOptions(elements.model, uniqueSortedValues("model"));
            addOptions(elements.prompt, uniqueSortedValues("prompt"));
            addOptions(elements.context, uniqueSortedValues("context"));
            addOptions(
                elements.evaluation,
                uniqueSortedValues("evaluation")
            );
        }}

        function memoMatchesFilters(memo) {{
            return (
                (!elements.company.value ||
                    memo.company === elements.company.value) &&
                (!elements.model.value ||
                    memo.model === elements.model.value) &&
                (!elements.prompt.value ||
                    memo.prompt === elements.prompt.value) &&
                (!elements.context.value ||
                    memo.context === elements.context.value) &&
                (!elements.evaluation.value ||
                    memo.evaluation === elements.evaluation.value)
            );
        }}

        function applyFilters() {{
            state.visibleCount = INITIAL_PAGE_SIZE;
            state.filteredMemos = MEMOS.filter(memoMatchesFilters);
            render();
        }}

        function createCell(value, className = "") {{
            const cell = document.createElement("td");

            if (className) {{
                cell.className = className;
            }}

            cell.textContent = normaliseValue(value) || "—";
            return cell;
        }}

        function createMemoRow(memo) {{
            const row = document.createElement("tr");
            row.tabIndex = 0;
            row.setAttribute(
                "aria-label",
                `Open credit memo for ${{memo.company}}`
            );

            row.appendChild(
                createCell(memo.company, "memo-table__company")
            );

            row.appendChild(
                createCell(memo.model, "memo-table__model")
            );

            row.appendChild(
                createCell(memo.prompt, "memo-table__prompt")
            );

            row.appendChild(
                createCell(memo.context, "memo-table__context")
            );

            row.appendChild(
                createCell(memo.evaluation, "memo-table__evaluation")
            );

            row.appendChild(
                createCell(memo.issues, "memo-table__issues")
            );

            const openCell = document.createElement("td");
            openCell.className = "memo-table__open";

            const arrow = document.createElement("span");
            arrow.className = "open-arrow";
            arrow.setAttribute("aria-hidden", "true");
            arrow.textContent = "›";

            openCell.appendChild(arrow);
            row.appendChild(openCell);

            const openMemo = () => {{
                const href = memoHref(memo.href);

                if (href) {{
                    window.location.href = "/" + encodeURIComponent(href);
                }}
            }};

            row.addEventListener("click", openMemo);

            row.addEventListener("keydown", (event) => {{
                if (event.key === "Enter" || event.key === " ") {{
                    event.preventDefault();
                    openMemo();
                }}
            }});

            return row;
        }}

        function updateResultsCount() {{
            const total = state.filteredMemos.length;
            const showing = Math.min(state.visibleCount, total);

            elements.count.textContent =
                `Showing ${{showing}} of ${{total}} ` +
                "benchmark credit memos";
        }}

        function render() {{
            const visibleMemos = state.filteredMemos.slice(
                0,
                state.visibleCount
            );

            const fragment = document.createDocumentFragment();

            visibleMemos.forEach((memo) => {{
                fragment.appendChild(createMemoRow(memo));
            }});

            elements.tableBody.replaceChildren(fragment);

            const hasResults = state.filteredMemos.length > 0;
            elements.emptyState.hidden = hasResults;
            elements.tableBody.parentElement.hidden = !hasResults;

            elements.loadMore.hidden =
                state.visibleCount >= state.filteredMemos.length;

            updateResultsCount();
        }}

        function resetFilters() {{
            elements.company.value = "";
            elements.model.value = "";
            elements.prompt.value = "";
            elements.context.value = "";
            elements.evaluation.value = "";

            applyFilters();
        }}

        [
            elements.company,
            elements.model,
            elements.prompt,
            elements.context,
            elements.evaluation,
        ].forEach((selectElement) => {{
            selectElement.addEventListener("change", applyFilters);
        }});

        elements.reset.addEventListener("click", resetFilters);

        elements.loadMore.addEventListener("click", () => {{
            state.visibleCount += PAGE_INCREMENT;
            render();
        }});

        initialiseFilters();
        applyFilters();
    </script>
</body>
</html>
"""


def _normalise_manifest_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Map a batch-run manifest result into the frontend index format.
    """

    filename = _first_value(
        record,
        "html_file_name",
        "html_filename",
        "filename",
        "output_filename",
        "file",
        default="",
    )

    # Never expose an absolute filesystem path in a browser link.
    filename = Path(str(filename).strip()).name

    parsed_filename = _parse_filename_metadata(filename)

    company = _first_value(
        record,
        "company_name",
        "company",
        "obligor_name",
        "issuer_name",
        default=parsed_filename.get("company", ""),
    )

    model = _first_value(
        record,
        "model_display_name",
        "model_name",
        "model",
        "llm",
        default=parsed_filename.get("model", ""),
    )

    prompt = _first_value(
        record,
        "prompt_display_name",
        "prompt_name",
        "prompt",
        "prompt_type",
        default=parsed_filename.get("prompt", ""),
    )

    context = _first_value(
        record,
        "context_display_name",
        "financial_context",
        "context",
        "context_type",
        default=parsed_filename.get("context", ""),
    )

    evaluation = _first_value(
        record,
        "evaluation_display_name",
        "policy_evaluation",
        "evaluation",
        "policy",
        default=parsed_filename.get("evaluation", ""),
    )

    issues = _first_value(
        record,
        "issue_count",
        "issues",
        "annotation_count",
        "total_annotations",
        default=0,
    )

    return {
        "company": _display_text(company),
        "model": _display_text(model),
        "prompt": _display_text(prompt),
        "context": _display_text(context),
        "evaluation": _display_text(evaluation),
        "issues": issues,
        "href": filename,
    }


def _first_value(
    record: dict[str, Any],
    *keys: str,
    default: Any = "",
) -> Any:
    for key in keys:
        value = record.get(key)

        if value is not None and value != "":
            return value

    return default


def _display_text(value: Any) -> str:
    text = str(value or "").strip()

    replacements = {
        "ctx_full": "Full Context",
        "ctx_minimal": "Minimal Context",
        "context_full": "Full Context",
        "context_minimal": "Minimal Context",
        "prompt_tight": "Tight",
        "prompt_loose": "Loose",
        "policy_none": "None",
        "policy_llm": "LLM Evaluated",
        "policy_deterministic": "Deterministic Evaluated",
        "tier_mini": "GPT-4o-mini",
    }

    if text in replacements:
        return replacements[text]

    return text.replace("_", " ").strip().title()


def _parse_filename_metadata(filename: str) -> dict[str, str]:
    """
    Extract fallback metadata from filenames such as:

    0002__GOOG__ctx_full__policy_none__prompt_loose__
    tier_mini__run_1__hash.html
    """

    stem = Path(filename).stem
    parts = stem.split("__")

    parsed: dict[str, str] = {}

    for part in parts:
        if part.startswith("ctx_"):
            parsed["context"] = part

        elif part.startswith("policy_"):
            parsed["evaluation"] = part

        elif part.startswith("prompt_"):
            parsed["prompt"] = part

        elif part.startswith("tier_"):
            parsed["model"] = part

    if len(parts) >= 2:
        parsed["company"] = parts[1]

    return parsed