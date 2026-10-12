import os
import re
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from backend.db.connection import ROOT_DIR
from backend.ingestion.pipeline.common import (
    extract_sentences_from_pdf,
)


def fetch_lingua_pdf_catalog(
    base_url: str = "https://lingua.com/french/reading/"
) -> List[Dict[str, str]]:
    """
    Parses the Lingua.com French reading index page and retrieves all free reading texts,
    their CEFR levels, titles, and direct PDF download links.
    """
    req = urllib.request.Request(
        base_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")

    # Split page by Level headings (e.g., Level A1, Level A2, Level B1, Level B2)
    level_sections = re.findall(
        r'<h2[^>]*>(Level [^<]+)</h2>(.*?)(?=<h2|\Z)',
        html,
        re.DOTALL
    )

    catalog = []
    for level_heading, sec_html in level_sections:
        # Extract normalized level name e.g. "A1", "A2", "B1", "B2"
        lvl_m = re.search(r'([A-C][1-2])', level_heading)
        level = lvl_m.group(1) if lvl_m else level_heading.strip()

        # Find items with reading links and PDF links
        # Lingua structure: <a href="..." class="reading-link">Title</a> ... <a href="...pdf"
        items = re.findall(
            r'<a href=\"([^\"]+)\" class=\"reading-link\">([^<]+)</a>.*?<a href=\"([^\"]+\.pdf)\"',
            sec_html,
            re.DOTALL
        )

        for rel_page, title, rel_pdf in items:
            pdf_url = urllib.parse.urljoin(base_url, rel_pdf)
            catalog.append({
                "title": title.strip(),
                "level": level,
                "pdf_url": pdf_url,
                "filename": os.path.basename(urllib.parse.urlparse(pdf_url).path)
            })

    return catalog


def download_lingua_pdfs(
    catalog: Optional[List[Dict[str, str]]] = None,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Downloads PDFs from the catalog into output_dir. Caches existing files on disk.
    Returns the list of metadata items augmented with local 'path'.
    """
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "data", "books", "lingua")
    os.makedirs(output_dir, exist_ok=True)

    if catalog is None:
        print("Fetching Lingua reading catalog...")
        catalog = fetch_lingua_pdf_catalog()

    if limit is not None and limit > 0:
        catalog = catalog[:limit]

    print(f"Downloading {len(catalog)} PDFs to {output_dir}...")
    downloaded_items = []
    for item in tqdm(catalog, desc="Downloading Lingua PDFs"):
        local_path = os.path.join(output_dir, item["filename"])
        if not os.path.exists(local_path):
            req = urllib.request.Request(
                item["pdf_url"],
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
                f.write(resp.read())

        item_with_path = dict(item)
        item_with_path["path"] = local_path
        downloaded_items.append(item_with_path)

    return downloaded_items


def run_mine_lingua(
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
    interactive: bool = False
) -> List[Dict[str, Any]]:
    """
    High-level entry point to mine candidate sentences from Lingua.com:
    1. Downloads available PDFs (cached in output_dir).
    2. Extracts candidate sentences (stopping at comprehension questions).
    3. Returns candidate records formatted for the ingestion pipeline.
    """
    print("\n--- Mining French Texts from Lingua.com ---")
    downloaded = download_lingua_pdfs(output_dir=output_dir, limit=limit)
    if not downloaded:
        print("No Lingua PDFs downloaded.")
        return []

    print(f"\nExtracting sentences from {len(downloaded)} PDFs...")
    candidate_records: List[Dict[str, Any]] = []
    for item in downloaded:
        sents = extract_sentences_from_pdf(
            item["path"],
            stop_phrase="Questions de compréhension:"
        )
        for sent in sents:
            candidate_records.append({
                "fr": sent,
                "es": None,
                "source": "lingua_pdf",
                "metadata": {
                    "title": item.get("title", ""),
                    "level": item.get("level", ""),
                },
            })

    print(f"\nSuccessfully extracted {len(candidate_records)} candidate sentences from Lingua.com.")
    return candidate_records


if __name__ == "__main__":
    run_mine_lingua()
