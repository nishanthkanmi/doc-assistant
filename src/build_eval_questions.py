"""Builds the test set for step 7: data/eval/questions.json (40 questions).

- 25 numeric questions. The correct answers come from the SEC's official
  structured XBRL data (the data.sec.gov "companyfacts" API), not from the
  model and not from our own text extraction. Each value is also checked to
  appear in the cleaned filing, so the question really is answerable.
- 10 text questions, with short reference answers written by hand from the filings.
- 5 questions the reports cannot answer. The correct behavior is "I don't know".

Run from the project root, after step 2:
    py src/build_eval_questions.py --email you@example.com

Use the same email you gave to download_filings.py: the SEC requires every
automated request to say who is making it.
"""

import argparse
import json
import os
import time
from datetime import date

import requests

from config import load_config

CONFIG = load_config()
CLEAN_DIR = CONFIG.clean_dir
RAW_DIR = CONFIG.source_dir
PAUSE_SECONDS = CONFIG.section("download")["pause_seconds"]

NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla"}

# (ticker, XBRL concept name, question). The concept name says exactly which
# number in the SEC's data is the correct answer.
REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
OPERATING_CASH = "NetCashProvidedByUsedInOperatingActivities"
NUMERIC = [
    ("AAPL", REVENUE, "What were Apple's total net sales in fiscal year 2025?"),
    ("AAPL", "NetIncomeLoss", "What was Apple's net income in fiscal year 2025?"),
    ("AAPL", "OperatingIncomeLoss", "What was Apple's operating income in fiscal year 2025?"),
    ("AAPL", "ResearchAndDevelopmentExpense", "How much did Apple spend on research and development in fiscal year 2025?"),
    ("AAPL", "Assets", "What were Apple's total assets as of September 27, 2025?"),
    ("AAPL", OPERATING_CASH, "How much cash did Apple generate from operating activities in fiscal year 2025?"),
    ("AAPL", "EarningsPerShareDiluted", "What was Apple's diluted earnings per share in fiscal year 2025?"),
    ("AAPL", "PaymentsForRepurchaseOfCommonStock", "How much did Apple spend repurchasing its common stock in fiscal year 2025?"),
    ("MSFT", REVENUE, "What was Microsoft's total revenue in fiscal year 2026?"),
    ("MSFT", "NetIncomeLoss", "What was Microsoft's net income in fiscal year 2026?"),
    ("MSFT", "OperatingIncomeLoss", "What was Microsoft's operating income in fiscal year 2026?"),
    ("MSFT", "ResearchAndDevelopmentExpense", "How much did Microsoft spend on research and development in fiscal year 2026?"),
    ("MSFT", "Assets", "What were Microsoft's total assets as of June 30, 2026?"),
    ("MSFT", OPERATING_CASH, "How much cash did Microsoft generate from operating activities in fiscal year 2026?"),
    ("MSFT", "PaymentsToAcquirePropertyPlantAndEquipment", "How much did Microsoft spend on additions to property and equipment in fiscal year 2026?"),
    ("MSFT", "EarningsPerShareDiluted", "What was Microsoft's diluted earnings per share in fiscal year 2026?"),
    ("TSLA", "Revenues", "What were Tesla's total revenues in 2025?"),
    ("TSLA", "NetIncomeLoss", "What was Tesla's net income attributable to common stockholders in 2025?"),
    ("TSLA", "OperatingIncomeLoss", "What was Tesla's income from operations in 2025?"),
    ("TSLA", "ResearchAndDevelopmentExpense", "How much did Tesla spend on research and development in 2025?"),
    ("TSLA", "Assets", "What were Tesla's total assets as of December 31, 2025?"),
    ("TSLA", "CashAndCashEquivalentsAtCarryingValue", "How much cash and cash equivalents did Tesla have as of December 31, 2025?"),
    ("TSLA", OPERATING_CASH, "How much cash did Tesla generate from operating activities in 2025?"),
    ("TSLA", "PaymentsToAcquirePropertyPlantAndEquipment", "How much did Tesla spend to acquire property and equipment in 2025?"),
    ("TSLA", "GrossProfit", "What was Tesla's total gross profit in 2025?"),
]

# (ticker, question, reference answer, key facts). Each key fact is a list of
# acceptable spellings; the answer is correct if EVERY key fact has at least
# one spelling in it (ignoring capitals).
TEXT = [
    ("AAPL", "Where is Apple's headquarters located?",
     "Cupertino, California.", [["cupertino"]]),
    ("AAPL", "How does Apple define its fiscal year?",
     "A 52- or 53-week period that ends on the last Saturday of September.",
     [["last saturday"], ["september"]]),
    ("AAPL", "What kinds of products does Apple design, manufacture and market?",
     "Smartphones, personal computers, tablets, wearables and accessories, plus related services.",
     [["smartphone", "iphone"], ["computer", "mac"], ["tablet", "ipad"], ["wearable"]]),
    ("MSFT", "What are Microsoft's three reportable segments?",
     "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
     [["productivity and business processes"], ["intelligent cloud"], ["more personal computing"]]),
    ("MSFT", "What is Microsoft's mission?",
     "To empower every person and every organization on the planet to achieve more.",
     [["empower every person"], ["every organization"], ["achieve more"]]),
    ("MSFT", "About how many people did Microsoft employ full-time as of June 30, 2026?",
     "Approximately 223,000 people.", [["223,000", "223000", "223 thousand"]]),
    ("TSLA", "What are Tesla's two reportable segments?",
     "Automotive, and energy generation and storage.",
     [["automotive"], ["energy generation and storage"]]),
    ("TSLA", "In which city and state is Tesla's principal executive office?",
     "Austin, Texas.", [["austin"], ["texas"]]),
    ("TSLA", "Which energy storage products does Tesla name in its annual report?",
     "Megapack and Powerwall.", [["megapack"], ["powerwall"]]),
    ("TSLA", "On which stock exchange is Tesla's common stock listed?",
     "The Nasdaq Global Select Market.", [["nasdaq"]]),
]

# Things the three reports cannot tell us.
UNANSWERABLE = [
    ("TSLA", "What is Tesla's stock price today?",
     "Not in the reports; a 10-K does not contain today's price."),
    ("MSFT", "What will Microsoft's revenue be in fiscal year 2030?",
     "Not in the reports; they do not forecast that far ahead."),
    (None, "What were Amazon's total net sales in 2025?",
     "Amazon's report is not in the corpus."),
    ("AAPL", "How many iPhone units did Apple sell in fiscal year 2025?",
     "Apple does not report unit sales."),
    ("MSFT", "What was the total compensation of Microsoft's chief executive officer in fiscal year 2026?",
     "Executive pay is in the proxy statement, not in the 10-K."),
]


def get_json(url, headers):
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    time.sleep(PAUSE_SECONDS)
    return response.json()


def find_cik(ticker, headers):
    data = get_json("https://www.sec.gov/files/company_tickers.json", headers)
    return next(str(e["cik_str"]).zfill(10) for e in data.values() if e["ticker"] == ticker)


def filing_accession(ticker):
    """The accession number of the filing we downloaded, taken from its file name."""
    path = next(RAW_DIR.glob(f"{ticker}_10K_*.html"))
    return path.stem.split("_10K_")[1]


def pick_fact(facts, concept, accession):
    """Find the full-year value of a concept as reported in our downloaded filing."""
    units = facts["facts"]["us-gaap"][concept]["units"]
    unit = "USD/shares" if "USD/shares" in units else "USD"
    rows = [r for r in units[unit] if r["accn"] == accession and r["form"] == "10-K" and r["fp"] == "FY"]
    if any("start" in r for r in rows):
        # An amount over a period (like revenue): keep rows covering about one year.
        # (Amounts at a single date, like total assets, have no "start".)
        rows = [r for r in rows
                if 350 <= (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days <= 380]
    latest_end = max(r["end"] for r in rows)  # the current year, not the earlier years also shown
    row = next(r for r in rows if r["end"] == latest_end)
    return unit, row


def appears_in_filing(text, value, unit):
    """Check that the number is printed in the cleaned filing (in millions, or as EPS)."""
    shown = f"{value:.2f}" if unit == "USD/shares" else f"{abs(value) / 1e6:,.0f}"
    return shown in text


def main():
    parser = argparse.ArgumentParser(description="Build data/eval/questions.json")
    parser.add_argument("--email", default=os.environ.get("SEC_EMAIL"),
                        help="Your email (SEC requires it); or set SEC_EMAIL")
    args = parser.parse_args()
    if not args.email:
        raise SystemExit("Pass --email you@example.com (the SEC requires it).")
    headers = {"User-Agent": f"doc-assistant-portfolio {args.email}"}

    clean_text = {t: next(CLEAN_DIR.glob(f"{t}_10K_*.txt")).read_text(encoding="utf-8") for t in NAMES}
    facts = {}
    for ticker in NAMES:
        cik = find_cik(ticker, headers)
        facts[ticker] = get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers)
        print(f"{ticker}: downloaded XBRL facts")

    questions = []
    for ticker, concept, question in NUMERIC:
        unit, row = pick_fact(facts[ticker], concept, filing_accession(ticker))
        if not appears_in_filing(clean_text[ticker], row["val"], unit):
            raise SystemExit(f"{ticker} {concept} = {row['val']} is not printed in the cleaned filing.")
        questions.append({
            "id": f"num-{len(questions) + 1:02d}",
            "category": "numeric",
            "ticker": ticker,
            "question": question,
            "expected_value": row["val"],
            "unit": unit,
            "source": {
                "api": "data.sec.gov companyfacts",
                "concept": f"us-gaap:{concept}",
                "period_end": row["end"],
                "accession": row["accn"],
            },
        })

    for number, (ticker, question, reference, key_facts) in enumerate(TEXT, start=1):
        questions.append({
            "id": f"txt-{number:02d}", "category": "text", "ticker": ticker, "question": question,
            "reference_answer": reference, "key_facts": key_facts,
        })
    for number, (ticker, question, why) in enumerate(UNANSWERABLE, start=1):
        questions.append({
            "id": f"unans-{number:02d}", "category": "unanswerable", "ticker": ticker, "question": question,
            "why_unanswerable": why,
        })

    out = CONFIG.questions_file
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(questions)} questions to {out}")


if __name__ == "__main__":
    main()
