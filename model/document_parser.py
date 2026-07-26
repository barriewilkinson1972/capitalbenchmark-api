from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token


PARSER_VERSION = "1.1.0"


BLOCK_CLASSES = {
    "paragraph": "narrative",
    "bullet": "list",
    "numbered_item": "list",
    "table": "table",
    "horizontal_rule": "layout",
    "code_block": "code",
    "quote": "quote",
}


SECTION_TYPES = {
    "document introduction": "document_introduction",
    "introduction": "introduction",
    "executive summary": "executive_summary",
    "borrower and request summary": "borrower_request_summary",
    "business profile": "business_profile",
    "capital benchmark rating assessment": "rating_assessment",
    "rating driver commentary": "rating_driver_commentary",
    "positive rating drivers": "positive_rating_drivers",
    "negative rating drivers": "negative_rating_drivers",
    "neutral rating diagnostics": "neutral_rating_diagnostics",
    "financial risk assessment": "financial_risk",
    "financial watchpoints": "financial_watchpoints",
    "deterministically evaluated policy assessment": "policy_assessment",
    "llm evaluated policy assessment": "policy_assessment",
    "llm-applied policy assessment": "policy_assessment",
    "llm applied policy assessment": "policy_assessment",
    "policy breaches and triggers": "policy_breaches_triggers",
    "llm-identified policy breaches and triggers": "policy_breaches_triggers",
    "llm identified policy breaches and triggers": "policy_breaches_triggers",
    "required policy actions": "required_policy_actions",
    "llm-recommended policy actions": "required_policy_actions",
    "llm recommended policy actions": "required_policy_actions",
    "policy missing information": "policy_missing_information",
    "llm-identified missing information": "policy_missing_information",
    "llm identified missing information": "policy_missing_information",
    "policy escalation assessment": "policy_escalation_assessment",
    "llm escalation assessment": "policy_escalation_assessment",
    "peer and anchor context": "peer_anchor_context",
    "key credit strengths": "credit_strengths",
    "key credit watchpoints": "credit_watchpoints",
    "questions for relationship manager": "relationship_manager_questions",
    "credit committee focus areas": "credit_committee_focus",
    "conclusion and recommendation": "conclusion_recommendation",
    "data quality and limitations": "data_quality_limitations",
}


@dataclass
class SourceLines:
    start: int
    end: int


@dataclass
class DocumentBlock:
    block_id: str
    block_uuid: str
    section_id: str
    block_type: str
    block_class: str
    order: int
    document_order: int
    text: str
    raw_markdown: str
    source_lines: SourceLines | None
    text_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSection:
    section_id: str
    title: str
    section_type: str
    level: int
    order: int
    blocks: list[DocumentBlock] = field(default_factory=list)


@dataclass
class ParsedDocument:
    memo_id: str
    document_title: str | None
    parser_version: str
    source_sha256: str
    sections: list[DocumentSection]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalise_text(value: str) -> str:
    """Normalise whitespace without changing wording."""
    return re.sub(r"\s+", " ", value).strip()


def classify_section(title: str) -> str:
    """Map a section heading to a stable semantic section type."""
    return SECTION_TYPES.get(normalise_text(title).casefold(), "other")


def classify_block(block_type: str) -> str:
    """Map a structural block type to a broader block class."""
    return BLOCK_CLASSES.get(block_type, "other")


def make_block_uuid(memo_id: str, block_id: str, text_sha256: str) -> str:
    """Create a deterministic UUID for a block within an immutable memo."""
    value = f"{memo_id}:{block_id}:{text_sha256}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def token_text(token: Token) -> str:
    """Extract readable text from an inline Markdown token."""
    if token.type == "inline" and token.children:
        parts: list[str] = []

        for child in token.children:
            if child.type in {"text", "code_inline"}:
                parts.append(child.content)
            elif child.type in {"softbreak", "hardbreak"}:
                parts.append("\n")
            elif child.type == "image":
                parts.append(child.content)

        return normalise_text("".join(parts))

    return normalise_text(token.content)


def source_slice(markdown_lines: list[str], token: Token) -> tuple[str, SourceLines | None]:
    """
    markdown-it uses zero-based start lines and an exclusive end line.
    The JSON output uses one-based, inclusive line numbering.
    """
    if not token.map:
        return "", None

    start_zero, end_exclusive = token.map
    raw = "\n".join(markdown_lines[start_zero:end_exclusive]).strip()

    return raw, SourceLines(
        start=start_zero + 1,
        end=end_exclusive,
    )


def make_block_id(
    section_id: str,
    block_type: str,
    block_number: int,
) -> str:
    prefixes = {
        "paragraph": "p",
        "bullet": "b",
        "numbered_item": "n",
        "table": "t",
        "horizontal_rule": "hr",
        "code_block": "code",
        "quote": "q",
    }

    prefix = prefixes.get(block_type, "x")
    return f"{section_id}_{prefix}_{block_number:03d}"


def parse_table(
    tokens: list[Token],
    start_index: int,
) -> tuple[dict[str, Any], int]:
    """
    Parse a markdown-it table token sequence into headers and rows.

    Returns:
        table_data
        index of the closing table token
    """
    headers: list[str] = []
    rows: list[list[str]] = []
    current_row: list[str] = []
    inside_header = False
    index = start_index

    while index < len(tokens):
        token = tokens[index]

        if token.type == "thead_open":
            inside_header = True

        elif token.type == "thead_close":
            inside_header = False

        elif token.type == "tr_open":
            current_row = []

        elif token.type == "inline":
            current_row.append(token_text(token))

        elif token.type == "tr_close":
            if inside_header:
                headers = current_row
            else:
                rows.append(current_row)

        elif token.type == "table_close":
            return {
                "headers": headers,
                "rows": rows,
            }, index

        index += 1

    raise ValueError("Table was opened but no table_close token was found.")


def parse_markdown_to_blocks(
    memo_id: str,
    markdown: str,
) -> dict[str, Any]:
    if not memo_id.strip():
        raise ValueError("memo_id must not be empty.")

    if not markdown.strip():
        raise ValueError("markdown must not be empty.")

    document_title: str | None = None

    parser = MarkdownIt(
        "commonmark",
        {
            "html": False,
            "linkify": False,
            "typographer": False,
        },
    ).enable("table")

    tokens = parser.parse(markdown)
    markdown_lines = markdown.splitlines()

    sections: list[DocumentSection] = []

    # Content appearing before the first heading is retained.
    current_section = DocumentSection(
        section_id="sec_000",
        title="Document Introduction",
        section_type=classify_section("Document Introduction"),
        level=0,
        order=0,
    )
    sections.append(current_section)

    section_number = 0
    block_number = 0
    document_order = 0
    list_stack: list[str] = []

    index = 0

    while index < len(tokens):
        token = tokens[index]

        # Heading
        if token.type == "heading_open":
            heading_level = int(token.tag[1:])
            inline_token = tokens[index + 1]
            heading_text = token_text(inline_token)

            # Treat the first H1 as the document title rather than a section.
            if (
                heading_level == 1
                and document_title is None
                and section_number == 0
                and not current_section.blocks
            ):
                document_title = heading_text
                index += 3
                continue

            section_number += 1
            block_number = 0

            current_section = DocumentSection(
                section_id=f"sec_{section_number:03d}",
                title=heading_text,
                section_type=classify_section(heading_text),
                level=heading_level,
                order=section_number,
            )
            sections.append(current_section)

            index += 3
            continue

        # Track whether list items are ordered or unordered.
        if token.type == "bullet_list_open":
            list_stack.append("bullet")
            index += 1
            continue

        if token.type == "ordered_list_open":
            list_stack.append("numbered_item")
            index += 1
            continue

        if token.type in {"bullet_list_close", "ordered_list_close"}:
            if list_stack:
                list_stack.pop()
            index += 1
            continue

        # Paragraph or list item content
        if token.type == "paragraph_open":
            inline_token = tokens[index + 1]
            raw_markdown, source_lines = source_slice(markdown_lines, token)

            block_type = list_stack[-1] if list_stack else "paragraph"
            block_number += 1
            document_order += 1
            text = token_text(inline_token)
            block_id = make_block_id(
                current_section.section_id,
                block_type,
                block_number,
            )
            text_hash = sha256_text(text)

            current_section.blocks.append(
                DocumentBlock(
                    block_id=block_id,
                    block_uuid=make_block_uuid(memo_id, block_id, text_hash),
                    section_id=current_section.section_id,
                    block_type=block_type,
                    block_class=classify_block(block_type),
                    order=block_number,
                    document_order=document_order,
                    text=text,
                    raw_markdown=raw_markdown,
                    source_lines=source_lines,
                    text_sha256=text_hash,
                )
            )

            index += 3
            continue

        # Markdown table
        if token.type == "table_open":
            table_data, closing_index = parse_table(tokens, index)

            raw_markdown, source_lines = source_slice(markdown_lines, token)
            table_text = " | ".join(
                table_data["headers"]
                + [
                    " | ".join(row)
                    for row in table_data["rows"]
                ]
            )

            block_number += 1
            document_order += 1
            block_id = make_block_id(
                current_section.section_id,
                "table",
                block_number,
            )
            normalised_table_text = normalise_text(table_text)
            text_hash = sha256_text(normalised_table_text)

            current_section.blocks.append(
                DocumentBlock(
                    block_id=block_id,
                    block_uuid=make_block_uuid(memo_id, block_id, text_hash),
                    section_id=current_section.section_id,
                    block_type="table",
                    block_class=classify_block("table"),
                    order=block_number,
                    document_order=document_order,
                    text=normalised_table_text,
                    raw_markdown=raw_markdown,
                    source_lines=source_lines,
                    text_sha256=text_hash,
                    metadata=table_data,
                )
            )

            index = closing_index + 1
            continue

        # Horizontal rule
        if token.type == "hr":
            raw_markdown, source_lines = source_slice(markdown_lines, token)
            block_number += 1
            document_order += 1
            block_id = make_block_id(
                current_section.section_id,
                "horizontal_rule",
                block_number,
            )
            text_hash = sha256_text("")

            current_section.blocks.append(
                DocumentBlock(
                    block_id=block_id,
                    block_uuid=make_block_uuid(memo_id, block_id, text_hash),
                    section_id=current_section.section_id,
                    block_type="horizontal_rule",
                    block_class=classify_block("horizontal_rule"),
                    order=block_number,
                    document_order=document_order,
                    text="",
                    raw_markdown=raw_markdown,
                    source_lines=source_lines,
                    text_sha256=text_hash,
                )
            )

            index += 1
            continue

        # Fenced or indented code
        if token.type in {"fence", "code_block"}:
            raw_markdown, source_lines = source_slice(markdown_lines, token)
            text = token.content.rstrip()
            block_number += 1
            document_order += 1
            block_id = make_block_id(
                current_section.section_id,
                "code_block",
                block_number,
            )
            text_hash = sha256_text(text)

            current_section.blocks.append(
                DocumentBlock(
                    block_id=block_id,
                    block_uuid=make_block_uuid(memo_id, block_id, text_hash),
                    section_id=current_section.section_id,
                    block_type="code_block",
                    block_class=classify_block("code_block"),
                    order=block_number,
                    document_order=document_order,
                    text=text,
                    raw_markdown=raw_markdown,
                    source_lines=source_lines,
                    text_sha256=text_hash,
                    metadata={
                        "language": token.info.strip() if token.info else None,
                    },
                )
            )

            index += 1
            continue

        index += 1

    # Remove the synthetic introduction when nothing preceded the first heading.
    sections = [
        section
        for section in sections
        if section.blocks or section.section_id != "sec_000"
    ]

    document = ParsedDocument(
        memo_id=memo_id,
        document_title=document_title,
        parser_version=PARSER_VERSION,
        source_sha256=sha256_text(markdown),
        sections=sections,
    )

    return asdict(document)
