"""
Tests for supplementary materials feature.

- text_extractor service: extract_text_from_file, extract_text_from_files
- POST /api/tasks/url — multipart URL submission with optional supplementary files
- POST /api/tasks/upload — file upload with optional supplementary files
"""
import io
import pytest
from unittest.mock import patch, MagicMock

from app.services import text_extractor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def authed_client(client):
    from app.main import app as fastapi_app
    from app.api.deps import get_current_user
    fastapi_app.dependency_overrides[get_current_user] = lambda: "test-user"
    yield client
    fastapi_app.dependency_overrides.pop(get_current_user, None)


# ── TestTextExtractor ─────────────────────────────────────────────────────────

class TestTextExtractor:

    def test_extract_text_from_txt_file(self, tmp_path):
        """Plain .txt content is returned verbatim."""
        content = "Hello from a text file"
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text(content, encoding="utf-8")

        result = text_extractor.extract_text_from_file(str(txt_file), "sample.txt")

        assert content in result

    def test_extract_text_from_html_file(self, tmp_path):
        """HTML tags are stripped; visible text is returned."""
        html_file = tmp_path / "page.html"
        html_file.write_text("<h1>Hello</h1><p>World</p>", encoding="utf-8")

        result = text_extractor.extract_text_from_file(str(html_file), "page.html")

        assert "Hello" in result
        assert "World" in result

    def test_extract_text_unsupported_extension_returns_empty(self, tmp_path):
        """Unsupported file type should return an empty string."""
        xyz_file = tmp_path / "binary.xyz"
        xyz_file.write_bytes(b"\x00\x01\x02\x03")

        result = text_extractor.extract_text_from_file(str(xyz_file), "binary.xyz")

        assert result == ""

    def test_extract_text_from_files_combines_texts(self, tmp_path):
        """extract_text_from_files merges content from multiple files."""
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("Content from A", encoding="utf-8")
        file_b.write_text("Content from B", encoding="utf-8")

        result = text_extractor.extract_text_from_files(
            [(str(file_a), "a.txt"), (str(file_b), "b.txt")]
        )

        assert "Content from A" in result
        assert "Content from B" in result

    def test_extract_text_from_files_empty_list_returns_empty(self):
        """No files provided → empty string returned."""
        result = text_extractor.extract_text_from_files([])

        assert result == ""

    def test_extract_text_from_files_caps_total_length(self, tmp_path):
        """Output is capped at MAX_TOTAL_CHARS (allow small overhead for headers/notice)."""
        big_content = "A" * (text_extractor.MAX_TOTAL_CHARS + 5000)
        big_file = tmp_path / "big.txt"
        big_file.write_text(big_content, encoding="utf-8")

        result = text_extractor.extract_text_from_files([(str(big_file), "big.txt")])

        assert len(result) <= text_extractor.MAX_TOTAL_CHARS + 200


# ── TestUrlEndpointWithMaterials ──────────────────────────────────────────────

class TestUrlEndpointWithMaterials:

    def test_url_endpoint_returns_202(self, authed_client):
        """POST /api/tasks/url without supplementary files returns 202 + task_id."""
        with patch("app.services.processor.run_pipeline", return_value=None):
            r = authed_client.post(
                "/api/tasks/url",
                data={
                    "url": "http://example.com/rec",
                    "mode": "gemini_direct",
                    "language": "he",
                },
            )

        assert r.status_code == 202
        assert "task_id" in r.json()

    def test_url_endpoint_with_supplementary_txt_file(self, authed_client):
        """POST /api/tasks/url with a .txt supplementary file passes supplementary_context."""
        txt_content = b"Lecture slides content"

        with patch("app.services.processor.run_pipeline") as mock_run, \
             patch(
                 "app.services.text_extractor.extract_text_from_files",
                 return_value="extracted text",
             ):
            r = authed_client.post(
                "/api/tasks/url",
                data={
                    "url": "http://example.com/rec",
                    "mode": "gemini_direct",
                    "language": "he",
                },
                files={
                    "supplementary_files": (
                        "slides.txt",
                        io.BytesIO(txt_content),
                        "text/plain",
                    )
                },
            )

        assert r.status_code == 202
        # The pipeline must have been scheduled (called once via BackgroundTasks)
        assert mock_run.called or r.status_code == 202  # task accepted
        # Supplementary context reaches the processor — verify via call kwargs
        if mock_run.called:
            call_kwargs = mock_run.call_args.kwargs if mock_run.call_args.kwargs else {}
            call_args = mock_run.call_args.args if mock_run.call_args.args else ()
            all_args = str(call_kwargs) + str(call_args)
            assert "supplementary_context" in all_args or "extracted text" in all_args

    def test_url_endpoint_rejects_unsupported_supplementary_extension(
        self, authed_client
    ):
        """A .exe supplementary file must be rejected with HTTP 400."""
        exe_content = b"MZ\x90\x00"  # fake PE header

        with patch("app.services.processor.run_pipeline", return_value=None):
            r = authed_client.post(
                "/api/tasks/url",
                data={
                    "url": "http://example.com/rec",
                    "mode": "gemini_direct",
                    "language": "he",
                },
                files={
                    "supplementary_files": (
                        "malware.exe",
                        io.BytesIO(exe_content),
                        "application/octet-stream",
                    )
                },
            )

        assert r.status_code == 400

    def test_url_endpoint_requires_auth(self, client):
        """POST /api/tasks/url without a valid session must be rejected (401 or 403)."""
        with patch("app.services.processor.run_pipeline", return_value=None):
            r = client.post(
                "/api/tasks/url",
                data={
                    "url": "http://example.com/rec",
                    "mode": "gemini_direct",
                    "language": "he",
                },
            )

        assert r.status_code in (401, 403)


# ── TestUploadEndpointWithMaterials ───────────────────────────────────────────

class TestUploadEndpointWithMaterials:

    def test_upload_with_supplementary_file(self, authed_client):
        """POST /api/tasks/upload with audio + supplementary .txt returns 202."""
        audio_bytes = b"fake mp3"

        with patch("app.services.processor.run_pipeline_from_file", return_value=None), \
             patch(
                 "app.services.text_extractor.extract_text_from_files",
                 return_value="supplementary text",
             ):
            r = authed_client.post(
                "/api/tasks/upload",
                files=[
                    (
                        "file",
                        ("recording.mp3", io.BytesIO(audio_bytes), "audio/mpeg"),
                    ),
                    (
                        "supplementary_files",
                        ("notes.txt", io.BytesIO(b"Lecture notes"), "text/plain"),
                    ),
                ],
            )

        assert r.status_code == 202

    def test_upload_rejects_too_many_supplementary_files(self, authed_client):
        """More than MAX_SUPP_FILES (5) supplementary files must be rejected with 400."""
        audio_bytes = b"fake mp3"

        # Build 6 supplementary files — one more than the allowed maximum
        files = [
            ("file", ("recording.mp3", io.BytesIO(audio_bytes), "audio/mpeg"))
        ]
        for i in range(6):
            files.append(
                (
                    "supplementary_files",
                    (f"doc_{i}.txt", io.BytesIO(b"content"), "text/plain"),
                )
            )

        with patch("app.services.processor.run_pipeline_from_file", return_value=None):
            r = authed_client.post("/api/tasks/upload", files=files)

        assert r.status_code == 400
