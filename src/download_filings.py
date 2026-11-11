"""Step 1 of the pipeline: LOAD.

Downloads the most recent 10-K annual report for a few public companies
from SEC EDGAR and saves the raw HTML into data/raw/.

Run from the project root:
    python src/download_filings.py --email you@example.com

The SEC requires every automated request to say who is making it, so you must
pass your own email address. They also limit how fast you can request pages,
so this script waits between requests.
"""

import argparse
import time

import requests

from config import load_config

# Companies to download, the save folder and the pause between requests all
# come from config.yaml (the "download" and "documents" sections).
CONFIG = load_config()
TICKERS = CONFIG.section("download")["tickers"]
RAW_DIR = CONFIG.source_dir
PAUSE_SECONDS = CONFIG.section("download")["pause_seconds"]


def get_json(url, headers):
    """Fetch a URL and return the parsed JSON, or stop with a clear error."""
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()  # raises an error if the request failed
    time.sleep(PAUSE_SECONDS)
    return response.json()


def find_cik(ticker, headers):
    """Look up a company's CIK (the SEC's ID number) from its ticker symbol."""
    data = get_json("https://www.sec.gov/files/company_tickers.json", headers)
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)  # SEC wants 10 digits
    raise ValueError(f"Ticker not found: {ticker}")


def latest_10k(cik, headers):
    """Return (accession_number, document_name, filing_date) of the newest 10-K."""
    data = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json", headers)
    recent = data["filings"]["recent"]  # parallel lists, one entry per filing
    for i, form in enumerate(recent["form"]):
        if form == "10-K":  # filings are listed newest first
            return (
                recent["accessionNumber"][i],
                recent["primaryDocument"][i],
                recent["filingDate"][i],
            )
    raise ValueError(f"No 10-K found for CIK {cik}")


def download_filing(ticker, cik, accession, document, headers):
    """Download the filing's main document and save it to data/raw/."""
    folder = accession.replace("-", "")  # the URL uses the number without dashes
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{document}"
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    time.sleep(PAUSE_SECONDS)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{ticker}_10K_{accession}.html"
    out_path.write_bytes(response.content)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download recent 10-K filings.")
    parser.add_argument("--email", required=True, help="Your email (SEC requires it)")
    args = parser.parse_args()

    headers = {"User-Agent": f"doc-assistant-portfolio {args.email}"}

    for ticker in TICKERS:
        print(f"{ticker}: looking up company...")
        cik = find_cik(ticker, headers)
        accession, document, filing_date = latest_10k(cik, headers)
        path = download_filing(ticker, cik, accession, document, headers)
        size_mb = path.stat().st_size / 1_000_000
        print(f"{ticker}: saved {path} (filed {filing_date}, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
