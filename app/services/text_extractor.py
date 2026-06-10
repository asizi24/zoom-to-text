"""
Text extraction from supplementary materials (PDF, DOCX, HTML, TXT).

Called by the upload routes to convert uploaded supplementary files to plain text
before injecting them into the Gemini summarization prompt.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: set[str] = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}
MAX_PER_FILE_BYTES: int = 10 * 1024 * 1024  # 10 MB — refuse larger files
MAX_TOTAL_CHARS: int = 30_000               # cap combined extracted text sent to Gemini


def extract_text_from_file(path: str | Path, original_filename: str) -> str:
    """
    Extract plain text from a single file.

    Returns the extracted text (possibly empty string on failure — caller
    decides whether to propagate or skip). Never raises for known formats;
    unknown/unsupported extensions return "".
    """
    path = Path(path)
    ext = Path(original_filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Unsupported extension '%s' for file '%s'; skipping.", ext, original_filename)
        return ""

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        logger.warning("Cannot stat file '%s': %s", original_filename, exc)
        return ""

    if file_size > MAX_PER_FILE_BYTES:
        logger.warning(
            "File '%s' is %d bytes, exceeding %d byte limit; skipping.",
            original_filename,
            file_size,
            MAX_PER_FILE_BYTES,
        )
        return ""

    if ext == ".pdf":
        return _extract_pdf(path, original_filename)
    elif ext == ".docx":
        return _extract_docx(path, original_filename)
    elif ext in {".html", ".htm"}:
        return _extract_html(path, original_filename)
    elif ext in {".txt", ".md"}:
        return _extract_plaintext(path, original_filename)

    # Should not reach here given the ALLOWED_EXTENSIONS check above, but be safe.
    logger.warning("Unhandled extension '%s' for file '%s'; skipping.", ext, original_filename)
    return ""


def _extract_pdf(path: Path, original_filename: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    try:
        import pdfplumber  # noqa: PLC0415 — optional heavy dep, imported lazily

        pages_text: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract text from PDF '%s': %s", original_filename, exc)
        return ""


def _extract_docx(path: Path, original_filename: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        import docx  # noqa: PLC0415 — optional heavy dep, imported lazily

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract text from DOCX '%s': %s", original_filename, exc)
        return ""


def _extract_html(path: Path, original_filename: str) -> str:
    """Extract visible text from an HTML file using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415 — optional dep, imported lazily

        content = path.read_bytes()
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract text from HTML '%s': %s", original_filename, exc)
        return ""


def _extract_plaintext(path: Path, original_filename: str) -> str:
    """Read a plain-text or Markdown file directly."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read text file '%s': %s", original_filename, exc)
        return ""


def extract_text_from_files(file_pairs: list[tuple[str, str]]) -> str:
    """
    Extract and concatenate text from multiple supplementary files.

    Args:
        file_pairs: list of (temp_path, original_filename) pairs

    Returns:
        Combined plain text with per-file headers, capped at MAX_TOTAL_CHARS.
        Returns "" if file_pairs is empty.
    """
    if not file_pairs:
        return ""

    parts: list[str] = []
    for temp_path, original_filename in file_pairs:
        text = extract_text_from_file(temp_path, original_filename)
        if text:
            parts.append(f"\n\n=== {original_filename} ===\n{text}")

    combined = "".join(parts)

    if len(combined) > MAX_TOTAL_CHARS:
        combined = combined[:MAX_TOTAL_CHARS] + "\n[... קוצר לצורך אורך]"

    return combined
