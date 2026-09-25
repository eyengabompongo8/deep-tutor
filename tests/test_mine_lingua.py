import os
import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.pipeline.mine_lingua import (
    fetch_lingua_pdf_catalog,
    download_lingua_pdfs,
    run_mine_lingua,
)

SAMPLE_HTML = """
<html>
<body>
<h2>Level A1</h2>
<ul>
  <li>
    <a href="jessica/" class="reading-link">Je m’appelle Jessica</a>
    <a href="../../pdf/francais-texte-jessica.pdf">PDF</a>
  </li>
  <li>
    <a href="comment-vas-tu/" class="reading-link">Comment vas-tu ?</a>
    <a href="../../pdf/francais-texte-comment-vas-tu.pdf">PDF</a>
  </li>
</ul>
<h2>Level B2</h2>
<ul>
  <li>
    <a href="lyon/" class="reading-link">Lyon</a>
    <a href="../../pdf/francais-texte-lyon.pdf">PDF</a>
  </li>
</ul>
</body>
</html>
"""


@patch("backend.ingestion.pipeline.mine_lingua.urllib.request.urlopen")
def test_fetch_lingua_pdf_catalog(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = SAMPLE_HTML.encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    catalog = fetch_lingua_pdf_catalog("https://lingua.com/french/reading/")

    assert len(catalog) == 3
    assert catalog[0]["level"] == "A1"
    assert catalog[0]["title"] == "Je m’appelle Jessica"
    assert catalog[0]["pdf_url"] == "https://lingua.com/pdf/francais-texte-jessica.pdf"
    assert catalog[0]["filename"] == "francais-texte-jessica.pdf"

    assert catalog[2]["level"] == "B2"
    assert catalog[2]["title"] == "Lyon"
    assert catalog[2]["pdf_url"] == "https://lingua.com/pdf/francais-texte-lyon.pdf"


@patch("backend.ingestion.pipeline.mine_lingua.urllib.request.urlopen")
def test_download_lingua_pdfs_caching(mock_urlopen, tmp_path):
    mock_response = MagicMock()
    mock_response.read.return_value = b"%PDF-1.4 mock content"
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    catalog = [
        {
            "title": "Text 1",
            "level": "A1",
            "pdf_url": "https://lingua.com/pdf/text1.pdf",
            "filename": "text1.pdf"
        },
        {
            "title": "Text 2",
            "level": "A1",
            "pdf_url": "https://lingua.com/pdf/text2.pdf",
            "filename": "text2.pdf"
        }
    ]

    out_dir = str(tmp_path)

    # 1. First download (both fetched)
    downloaded = download_lingua_pdfs(catalog=catalog, output_dir=out_dir)
    assert len(downloaded) == 2
    assert os.path.exists(downloaded[0]["path"])
    assert mock_urlopen.call_count == 2

    # 2. Second download with pre-existing files (cached, no network call)
    mock_urlopen.reset_mock()
    downloaded_again = download_lingua_pdfs(catalog=catalog, output_dir=out_dir)
    assert len(downloaded_again) == 2
    assert mock_urlopen.call_count == 0


@patch("backend.ingestion.pipeline.mine_lingua.download_lingua_pdfs")
@patch("backend.ingestion.pipeline.mine_lingua.extract_sentences_from_pdf")
@patch("backend.ingestion.pipeline.mine_lingua.deduplicate_against_db")
@patch("backend.ingestion.pipeline.mine_lingua.process_and_score_sentences")
def test_run_mine_lingua(
    mock_process,
    mock_dedup,
    mock_extract,
    mock_download
):
    mock_download.return_value = [
        {"path": "/tmp/test1.pdf", "level": "A1", "filename": "test1.pdf"},
        {"path": "/tmp/test2.pdf", "level": "A2", "filename": "test2.pdf"}
    ]
    mock_extract.side_effect = [
        ["Bonjour le monde.", "Je m'appelle Pierre."],
        ["C'est une belle journée."]
    ]
    mock_dedup.return_value = [
        "Bonjour le monde.",
        "Je m'appelle Pierre.",
        "C'est une belle journée."
    ]
    mock_process.return_value = [
        {"fr": "Bonjour le monde.", "level": "A1"},
        {"fr": "Je m'appelle Pierre.", "level": "A1"},
        {"fr": "C'est une belle journée.", "level": "A2"}
    ]

    results = run_mine_lingua(limit=2)

    assert mock_download.called
    assert mock_extract.call_count == 2
    assert mock_dedup.called
    assert mock_process.called
    assert len(results) == 3
    assert results[0]["fr"] == "Bonjour le monde."
