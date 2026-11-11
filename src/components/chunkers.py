"""Chunkers: split a cleaned document into small searchable pieces.

SecFilingChunker is built for 10-K annual reports:
- A new chunk always starts at a section heading such as "Item 7. ...".
- Page decoration (page numbers, "Table of Contents", running headers like
  "PART II", page footers) is removed.
- Table rows are never split. A table too long for one chunk is split between
  rows, and each piece repeats the table's caption and heading rows.
- Heading rows that are shifted one column left are lined up again.

ParagraphChunker works on any text: it packs paragraphs into chunks and keeps
table rows together the same way, but knows nothing about filings.
"""

import re
from collections import Counter

from interfaces import Chunker

# Single lines that are page decoration, not content: page numbers, "Table of
# Contents" links, running headers like "PART II" or "Item 1B, 1C", and page
# footers like "Apple Inc. | 2025 Form 10-K | 21".
PAGE_JUNK = re.compile(
    r"\d{1,3}"
    r"|table of contents"
    r"|part [ivx]+\.?"
    r"|item \d+[a-c]?(, ?\d+[a-c]?)*"
    r"|.+\| ?\d{4} form 10-k ?\| ?\d+",
    re.IGNORECASE,
)

# A real section heading, such as "Item 7. Management's Discussion and Analysis".
SECTION_HEADING = re.compile(r"item \d+[a-c]?\.\s+\S", re.IGNORECASE)

YEAR = re.compile(r"\b(19|20)\d\d\b")


def filing_info(text, ticker):
    """Find the company name and fiscal year end on the filing's cover page."""
    lines = text.splitlines()
    company = ticker
    for i, line in enumerate(lines):
        if line.lower().startswith("commission file number"):
            company = next(l for l in lines[i + 1:] if l.strip())  # the next non-empty line
            break
    if company.isupper():
        company = company.title()  # "MICROSOFT CORPORATION" -> "Microsoft Corporation"

    match = re.search(r"fiscal year ended\s+(\w+ \d+\s*,\s*\d{4})", text, re.IGNORECASE)
    year_end = re.sub(r"\s*,\s*", ", ", match.group(1)) if match else "unknown"
    return company, year_end


def split_blocks(text, drop_page_junk):
    """Split text at blank lines, optionally dropping blocks that are page decoration."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text)]
    return [b for b in blocks
            if b and not (drop_page_junk and "\n" not in b and PAGE_JUNK.fullmatch(b))]


def is_table(block):
    return " | " in block


def is_section_heading(block):
    return bool(SECTION_HEADING.match(block)) and not is_table(block) and len(block) < 300


def typical_width(rows):
    """The most common number of cells per row, ignoring one-cell label rows."""
    counts = Counter(len(row) for row in rows if len(row) > 1)
    return counts.most_common(1)[0][0] if counts else 1


def fix_shifted_headers(rows, width):
    """Line up heading rows that lost their empty top-left cell.

    In a table like
        2025 | 2024 | 2023
        Products | $112,887 | $109,633 | $108,803
    the heading row has no blank cell above the row labels, so each year sits
    one column too far left. Heading rows come before the first full-width row;
    one that is exactly one cell short and starts with a year gets a blank cell.
    """
    for row in rows:
        if len(row) == width:
            break
        if len(row) == width - 1 and YEAR.search(row[0]):
            row.insert(0, "")
    return rows


def split_table(table, caption, max_chars):
    """Turn one table into one or more chunk-sized pieces, never splitting a row."""
    rows = [line.split(" | ") for line in table.splitlines()]
    width = typical_width(rows)
    rows = fix_shifted_headers(rows, width)
    lines = [" | ".join(row).lstrip() for row in rows]  # a blank first cell shows as "| 2025 | ..."

    # Heading rows are the ones before the first full-width row; repeat them in every piece.
    n_heading = next((i for i, row in enumerate(rows) if len(row) == width), 0)
    heading = ([caption, ""] if caption else []) + lines[:n_heading]

    pieces, current = [], []
    for line in lines[n_heading:]:
        if current and len("\n".join(heading + current + [line])) > max_chars:
            pieces.append("\n".join(heading + current))
            current = []
        current.append(line)
    pieces.append("\n".join(heading + current))
    return pieces


def pack(pieces, max_chars, separator):
    """Join pieces in order into groups of at most max_chars, never cutting a piece."""
    groups, current = [], ""
    for piece in pieces:
        if current and len(current) + len(separator) + len(piece) > max_chars:
            groups.append(current)
            current = piece
        else:
            current = current + separator + piece if current else piece
    if current:
        groups.append(current)
    return groups


def split_text(block, max_chars):
    """Split a long paragraph between sentences."""
    if len(block) <= max_chars:
        return [block]
    sentences = re.split(r"(?<=[.!?])\s+", block)
    return pack(sentences, max_chars, " ")


class SecFilingChunker(Chunker):
    def chunk(self, text, doc_name):
        max_chars = self.options["max_chars"]
        ticker = doc_name.split("_")[0]  # "AAPL_10K_....txt" -> "AAPL"
        company, year_end = filing_info(text, ticker)

        # First group the blocks into sections, each a list of chunk-sized pieces.
        sections = []
        title, pieces = "Cover page", []
        for block in split_blocks(text, drop_page_junk=True):
            if is_section_heading(block):
                sections.append((title, pieces))
                title, pieces = block, []
            elif is_table(block):
                # A short line ending in ":" just before a table is its caption, e.g.
                # "The following table shows net sales ... (dollars in millions):"
                has_caption = pieces and pieces[-1].endswith(":") and len(pieces[-1]) < 500
                caption = pieces.pop() if has_caption else None
                pieces.extend(split_table(block, caption, max_chars))
            else:
                pieces.extend(split_text(block, max_chars))
        sections.append((title, pieces))

        # Then pack neighboring pieces of the same section into chunks.
        chunks = []
        for title, pieces in sections:
            for chunk_text in pack(pieces, max_chars, "\n\n"):
                chunks.append({
                    "id": f"{ticker}-{len(chunks) + 1:04d}",
                    "heading": f"{company} 10-K, fiscal year ended {year_end}, {title}",
                    "text": chunk_text,
                    "ticker": ticker,
                    "company": company,
                    "year_end": year_end,
                    "section": title,
                })
        return chunks


class ParagraphChunker(Chunker):
    """Packs paragraphs into chunks of at most max_chars. Works on any text."""

    def chunk(self, text, doc_name):
        max_chars = self.options["max_chars"]
        pieces = []
        for block in split_blocks(text, drop_page_junk=False):
            if is_table(block):
                pieces.extend(split_table(block, None, max_chars))
            else:
                pieces.extend(split_text(block, max_chars))
        return [
            {"id": f"{doc_name}-{number:04d}", "heading": doc_name, "text": chunk_text}
            for number, chunk_text in enumerate(pack(pieces, max_chars, "\n\n"), start=1)
        ]
