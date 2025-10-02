from types import SimpleNamespace

import pytest

from avalan.memory.source import MemorySource, MemorySourceDocument


class DummyResponse:
    def __init__(
        self, *, headers: dict[str, str] | None = None, content: bytes = b""
    ) -> None:
        self.headers = headers or {}
        self.content = content
        self.raise_calls = 0

    def raise_for_status(self) -> None:
        self.raise_calls += 1


class DummyClient:
    def __init__(self, response: DummyResponse) -> None:
        self._response = response
        self.requested: str | None = None

    async def get(self, url: str) -> DummyResponse:
        self.requested = url
        return self._response

    async def aclose(self) -> None:
        raise AssertionError("DummyClient should not be closed")


@pytest.fixture
def require_asyncio_backend(anyio_backend: str) -> None:  # noqa: PT004
    if anyio_backend != "asyncio":
        pytest.skip("requires asyncio backend")


@pytest.mark.anyio
@pytest.mark.usefixtures("require_asyncio_backend")
async def test_fetch_uses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    response = DummyResponse(headers={}, content=b"payload")
    client = DummyClient(response=response)
    source = MemorySource(client=client)

    expected_doc = MemorySourceDocument(
        url="https://example.com",
        content_type="application/octet-stream",
        title="title",
        description="desc",
        markdown="payload",
    )

    async def fake_convert(
        url: str, content_type: str, data: bytes
    ) -> MemorySourceDocument:
        assert url == "https://example.com"
        assert content_type == ""
        assert data == b"payload"
        return expected_doc

    monkeypatch.setattr(source, "_convert_bytes", fake_convert)

    result = await source.fetch("https://example.com")

    assert result is expected_doc
    assert client.requested == "https://example.com"
    assert response.raise_calls == 1


@pytest.mark.anyio
@pytest.mark.usefixtures("require_asyncio_backend")
async def test_aclose_only_closes_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = MemorySource()

    closed = False

    async def fake_aclose() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(source._client, "aclose", fake_aclose)

    async with source:
        pass

    assert closed is True

    dummy = DummyClient(response=DummyResponse())
    shared = MemorySource(client=dummy)
    await shared.aclose()


@pytest.mark.anyio
@pytest.mark.usefixtures("require_asyncio_backend")
async def test_convert_bytes_prefers_html_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = MemorySource(max_description_chars=4)

    async def fake_run_sync(func, stream):
        return SimpleNamespace(
            markdown="# Heading\n\nParagraph",
            title=None,
            text_content="Fallback",
        )

    monkeypatch.setattr(
        "avalan.memory.source.to_thread.run_sync", fake_run_sync
    )
    monkeypatch.setattr(source, "_is_html", lambda url, ctype: True)
    monkeypatch.setattr(
        source,
        "_html_metadata",
        lambda data: ("Html Title", "  spaced   text  "),
    )
    monkeypatch.setattr(source, "_is_pdf", lambda url, ctype, data: False)

    document = await source.from_bytes(
        "https://example.com/index.html", "", b"<html></html>"
    )

    assert document.title == "Html Title"
    assert document.description == "spac"
    assert document.markdown == "# Heading\n\nParagraph"
    assert document.content_type == "application/octet-stream"


@pytest.mark.anyio
@pytest.mark.usefixtures("require_asyncio_backend")
async def test_convert_bytes_pdf_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = MemorySource()

    async def fake_run_sync(func, stream):
        return SimpleNamespace(
            markdown="# Abstract\nThis is the abstract.\n\n" + "A" * 120,
            title=None,
            text_content=None,
        )

    monkeypatch.setattr(
        "avalan.memory.source.to_thread.run_sync", fake_run_sync
    )
    monkeypatch.setattr(source, "_is_html", lambda url, ctype: False)
    monkeypatch.setattr(source, "_is_pdf", lambda url, ctype, data: True)

    class DummyReader:
        def __init__(self, stream) -> None:  # noqa: ANN001
            pass

        @property
        def metadata(self) -> dict[str, str]:
            return {"/Title": "PDF Title"}

    monkeypatch.setattr(
        "avalan.memory.source.PdfReader", lambda stream: DummyReader(stream)
    )

    document = await source.from_bytes(
        "https://example.com/doc.pdf", "application/pdf", b"%PDF-1.4 data"
    )

    assert document.title == "PDF Title"
    assert document.description == "This is the abstract."


@pytest.mark.anyio
@pytest.mark.usefixtures("require_asyncio_backend")
async def test_convert_bytes_uses_url_as_title_when_missing_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = MemorySource()

    async def fake_run_sync(func, stream):
        return SimpleNamespace(markdown="No headings here", title=None, text_content=None)

    monkeypatch.setattr(
        "avalan.memory.source.to_thread.run_sync",
        fake_run_sync,
    )
    monkeypatch.setattr(source, "_is_html", lambda url, ctype: False)
    monkeypatch.setattr(source, "_is_pdf", lambda url, ctype, data: False)

    document = await source.from_bytes(
        "https://example.com/files/document.txt", "text/plain", b"raw data"
    )

    assert document.title == "document.txt"
    assert document.description is None


def test_markdown_title() -> None:
    assert MemorySource._markdown_title("# Heading\nContent") == "Heading"
    assert MemorySource._markdown_title("No heading") is None


def test_markdown_description_paths() -> None:
    default_source = MemorySource()
    abstract = default_source._markdown_description(
        "# Abstract\nAn abstract with details."
    )
    assert abstract == "An abstract with details."

    long_para = "A" * 120
    limited_source = MemorySource(max_description_chars=20)
    fallback = limited_source._markdown_description(
        f"Intro\n\n{long_para}\n\nOther"
    )
    assert fallback == "A" * 20

    assert default_source._markdown_description("Too short") is None


def test_clean_snippet() -> None:
    assert (
        MemorySource._clean_snippet("  spaced   text  ", None) == "spaced text"
    )
    assert MemorySource._clean_snippet("  spaced   text  ", 6) == "spaced"


@pytest.mark.parametrize(
    ("url", "content_type", "data", "expected"),
    [
        ("https://example.com", "application/pdf", b"", True),
        ("https://example.com/file.pdf", "text/plain", b"", True),
        ("https://example.com/file", "text/plain", b"%PDF-1", True),
        ("https://example.com/file", "text/plain", b"data", False),
    ],
)
def test_is_pdf(
    url: str, content_type: str, data: bytes, expected: bool
) -> None:
    assert MemorySource._is_pdf(url, content_type, data) is expected


@pytest.mark.parametrize(
    ("url", "content_type", "expected"),
    [
        ("https://example.com", "text/html", True),
        ("https://example.com", "application/xhtml+xml", True),
        ("https://example.com/index.HTML", "text/plain", True),
        ("https://example.com/file.txt", "text/plain", False),
    ],
)
def test_is_html(url: str, content_type: str, expected: bool) -> None:
    assert MemorySource._is_html(url, content_type) is expected


def test_html_metadata_sources() -> None:
    source = MemorySource()
    html = b"""
    <html>
        <head>
            <meta property=\"og:title\" content=\"OG Title\" />
            <meta property=\"og:description\" content=\"OG Description\" />
        </head>
    </html>
    """

    title, description = source._html_metadata(html)
    assert title == "OG Title"
    assert description == "OG Description"

    limited_source = MemorySource(max_description_chars=9)
    html_with_p = b"""
    <html>
        <head><title>Page Title</title></head>
        <body>
        <p>Paragraph content that is quite long to trigger cleaning.</p>
        </body>
    </html>
    """

    title, description = limited_source._html_metadata(html_with_p)
    assert title == "Page Title"
    assert description == "Paragraph"

    twitter_only = b"""
    <html>
        <head>
            <meta name=\"twitter:title\" content=\"Twitter Title\" />
        </head>
    </html>
    """

    title, description = source._html_metadata(twitter_only)
    assert title == "Twitter Title"
    assert description is None
