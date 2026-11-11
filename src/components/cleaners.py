"""Cleaners: turn a source file into plain text.

HtmlCleaner removes everything a reader would not see (scripts, styles, hidden
tags, machine-readable XBRL data) and keeps tables as one line per row, with
cells separated by " | ". Paragraphs are separated by a blank line.
"""

import re

from bs4 import BeautifulSoup

from interfaces import Cleaner

# Tags whose contents are never text meant for a reader.
# "ix:header" holds the filing's hidden XBRL data (tagged facts for computers).
JUNK_TAGS = ["script", "style", "head", "ix:header"]

# Tags that a browser shows as a separate block, so they get a blank line after.
PARAGRAPH_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "table"]

# Tags that only start a new line.
LINE_TAGS = ["br", "li"]

# A character that never appears in filings. We mark line breaks with it while
# cleaning, because real newlines in HTML source are just spaces to a browser.
LINE_BREAK = "\x00"

HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)


def is_hidden(tag):
    """Return True if a browser would hide this tag from the reader."""
    style = tag.get("style", "")
    return (
        bool(HIDDEN_STYLE.search(style))
        or tag.has_attr("hidden")
        or tag.get("aria-hidden") == "true"
    )


def remove_junk(soup):
    """Delete scripts, styles, XBRL data and hidden tags from the page."""
    for tag in soup.find_all(JUNK_TAGS) + soup.find_all(is_hidden):
        if not tag.decomposed:  # skip tags already deleted along with a parent
            tag.decompose()


def join_number_pieces(cells):
    """Glue lone "$" cells to the next cell, and lone ")" or "%" to the previous one.

    Financial tables often put these symbols in their own cells, which would
    otherwise give "$ | 1,234" instead of "$1,234". A "$" is only glued to an
    amount, so column headings like "$ | %" stay as two cells.
    """
    joined = []
    for cell in cells:
        if joined and cell in (")", "%", ")%") and joined[-1] != "$":
            joined[-1] += cell
        elif joined and joined[-1] == "$" and re.match(r"[\d(—-]", cell):
            joined[-1] += cell
        else:
            joined.append(cell)
    return joined


def flatten_tables(soup):
    """Replace each table with plain text: one line per row, cells joined by " | "."""
    for table in soup.find_all("table"):
        for tag in table.find_all(["br", "p", "div"]):  # a line break inside a cell is a space here
            tag.insert_after(" ")
        lines = []
        for row in table.find_all("tr"):
            # .split() then " ".join() squeezes each cell's whitespace to single spaces
            cells = [" ".join(cell.get_text().split()) for cell in row.find_all(["td", "th"])]
            cells = join_number_pieces([c for c in cells if c])  # drop empty spacer cells
            if cells:
                lines.append(" | ".join(cells))
        table.replace_with(LINE_BREAK * 2 + LINE_BREAK.join(lines) + LINE_BREAK * 2)


def add_line_breaks(soup):
    """Mark where a browser would start a new line or paragraph."""
    for tag in soup.find_all(PARAGRAPH_TAGS):
        tag.insert_after(LINE_BREAK * 2)
    for tag in soup.find_all(LINE_TAGS):
        tag.insert_after(LINE_BREAK)


def tidy_whitespace(text):
    """Turn the marked-up text into neat lines and paragraphs."""
    text = re.sub(r"\s+", " ", text)  # any run of spaces/newlines/tabs -> one space, like a browser
    text = text.replace(LINE_BREAK, "\n")  # now put in the real line breaks
    text = re.sub(r" *\n *", "\n", text)  # no spaces at the start or end of lines
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
    return text.strip() + "\n"


class HtmlCleaner(Cleaner):
    extensions = (".html", ".htm")

    def clean(self, raw_bytes):
        soup = BeautifulSoup(raw_bytes, "html.parser")
        remove_junk(soup)
        flatten_tables(soup)  # must run before add_line_breaks, which would split table cells
        add_line_breaks(soup)
        return tidy_whitespace(soup.get_text())


class TextCleaner(Cleaner):
    """For files that are already plain text: just tidies the whitespace."""

    extensions = (".txt", ".md")

    def clean(self, raw_bytes):
        text = raw_bytes.decode("utf-8", errors="replace")
        text = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n"))  # no trailing spaces
        text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
        return text.strip() + "\n"
