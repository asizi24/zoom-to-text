"""
PDF exporter for task results.

Pipeline: TaskResponse → Obsidian Markdown → HTML → PDF (WeasyPrint)

WeasyPrint requires system libs (Pango, Cairo, GLib, fonts-noto) that are
installed in the Docker image. On a local dev machine without those libs
the import inside build_pdf() will raise ImportError, which the route
handler catches and converts to HTTP 503.
"""
from app.models import TaskResponse
from app.services.exporters.markdown import build_obsidian_markdown

_RTL_STYLE = """
  body {
    font-family: Arial, Helvetica, sans-serif;
    margin: 2cm;
    direction: rtl;
    line-height: 1.6;
    color: #222;
  }
  h1, h2, h3 {
    color: #2c3e50;
    border-bottom: 1px solid #eee;
    padding-bottom: 4px;
  }
  code {
    background: #f4f4f4;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 0.9em;
  }
  pre {
    background: #f4f4f4;
    padding: 12px;
    border-radius: 5px;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
  }
  th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: right;
  }
  th { background: #f2f2f2; }
  details {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
  }
  summary { cursor: pointer; font-weight: bold; }
  blockquote {
    border-right: 4px solid #ccc;
    margin: 0;
    padding: 0 15px;
    color: #666;
  }
"""


def _to_html(markdown_text: str) -> str:
    """Convert Obsidian-flavored Markdown to a styled, RTL HTML document."""
    import markdown as md_lib  # pure-Python, always available

    body = md_lib.markdown(
        markdown_text,
        extensions=["tables", "fenced_code"],
    )
    return (
        "<!DOCTYPE html>"
        '<html dir="rtl" lang="he">'
        "<head>"
        '<meta charset="utf-8">'
        f"<style>{_RTL_STYLE}</style>"
        "</head>"
        f"<body>{body}</body>"
        "</html>"
    )


def build_pdf(task: TaskResponse) -> bytes:
    """Convert a completed task to a PDF byte string.

    Raises ImportError if weasyprint is not installed or its system libs
    (Pango/Cairo) are missing — the route handler maps this to HTTP 503.
    """
    from weasyprint import HTML  # lazy: avoids hard error at module load time

    md = build_obsidian_markdown(task)
    html = _to_html(md)
    return HTML(string=html).write_pdf()
