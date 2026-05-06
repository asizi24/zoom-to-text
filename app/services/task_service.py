"""
Task creation helpers — thin layer between route handlers and the pipeline.

Extracted from app/api/routes.py to keep route handlers under ~20 lines each.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.services import text_extractor

logger = logging.getLogger(__name__)

_ALLOWED_SUPP_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}
MAX_SUPP_FILES = 5
MAX_SUPP_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per file


async def prepare_supplementary_context(
    task_id: str,
    supplementary_files: list[UploadFile],
) -> Optional[str]:
    """
    Validate, save, and extract text from supplementary material files.

    Returns the concatenated extracted text, or None when the list is empty.
    Temp files are always deleted in the finally block even if extraction fails.
    Raises HTTPException (400 / 413) on validation errors — bubbles up to the
    route handler unchanged.
    """
    if not supplementary_files:
        return None

    if len(supplementary_files) > MAX_SUPP_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"יותר מדי קבצי עזר. מקסימום {MAX_SUPP_FILES} קבצים.",
        )

    supp_pairs: list[tuple[str, str]] = []
    settings.downloads_dir.mkdir(parents=True, exist_ok=True)

    for sf in supplementary_files:
        sf_name = Path(sf.filename).name if sf.filename else "material"
        sf_ext = Path(sf_name).suffix.lower()
        if sf_ext not in _ALLOWED_SUPP_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"סוג קובץ עזר לא נתמך: {sf_ext}. קבצים נתמכים: PDF, DOCX, HTML, TXT, MD",
            )
        sf_content = await sf.read()
        if len(sf_content) > MAX_SUPP_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"קובץ עזר גדול מדי: {sf_name}. מקסימום 10 MB לקובץ.",
            )
        sf_path = settings.downloads_dir / f"{task_id}_supp_{sf_name}"
        async with aiofiles.open(sf_path, "wb") as f_supp:
            await f_supp.write(sf_content)
        supp_pairs.append((str(sf_path), sf_name))

    try:
        return text_extractor.extract_text_from_files(supp_pairs)
    finally:
        for sf_path_str, _ in supp_pairs:
            try:
                Path(sf_path_str).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Could not delete temp supp file %s: %s", sf_path_str, exc)
