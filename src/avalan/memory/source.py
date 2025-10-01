from os.path import basename
from re import DOTALL, MULTILINE, search, split
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlparse
from anyio import to_thread
from bs4 import BeautifulSoup
from httpx import AsyncClient, Response
from markitdown import MarkItDown
from pypdf import PdfReader


@dataclass(frozen=True, kw_only=True, slots=True)
class MemorySourceDocument:
    url: str
    content_type: str
    title: str | None
    description: str | None
    markdown: str


class MemorySource:
    _client: AsyncClient
    _max_description_chars: int | None
    _md: MarkItDown

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        follow_redirects: bool = True,
        client: AsyncClient | None = None,
        max_description_chars: int | None = None,
    ) -> None:
        self._own_client: bool = client is None
        self._client: AsyncClient = client or AsyncClient(
            follow_redirects=follow_redirects, timeout=timeout
        )
        self._md: MarkItDown = MarkItDown()
        self._max_description_chars = max_description_chars

    async def __aenter__(self) -> "MemorySource":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()

    async def fetch(self, url: str) -> MemorySourceDocument:
        resp: Response = await self._client.get(url)
        resp.raise_for_status()
        ctype: str = resp.headers.get("content-type", "") or ""
        data: bytes = resp.content
        return await self._convert_bytes(url, ctype, data)

    async def from_bytes(
        self, url: str, content_type: str, data: bytes
    ) -> MemorySourceDocument:
        return await self._convert_bytes(url, content_type, data)

    async def _convert_bytes(
        self, url: str, content_type: str, data: bytes
    ) -> MemorySourceDocument:
        result = await to_thread.run_sync(
            self._md.convert_stream,
            BytesIO(data)
        )
        markdown: str = (
            getattr(result, "markdown", None)
            or getattr(result, "text_content", None)
            or ""
        )
        title: str | None = getattr(result, "title", None)
        description: str | None = None

        if self._is_html(url, content_type):
            h_title, h_desc = self._html_metadata(data)
            title = title or h_title
            description = description or h_desc

        if self._is_pdf(url, content_type, data):
            metadata = PdfReader(BytesIO(data)).metadata
            metadata_title = (
                metadata["/Title"]
                if metadata and "/Title" in metadata else None
            )
            title = metadata_title or title or self._markdown_title(markdown)
            description = description or self._markdown_description(markdown)

        if not title:
            parsed = urlparse(url)
            title = basename(parsed.path) or parsed.netloc

        if description:
            description = self._clean_snippet(
                description,
                self._max_description_chars
            )

        return MemorySourceDocument(
            url=url,
            content_type=content_type or "application/octet-stream",
            title=title,
            description=description,
            markdown=markdown,
        )

    @staticmethod
    def _is_pdf(url: str, content_type: str, data: bytes) -> bool:
        ctype: str = (content_type or "").lower()
        if "pdf" in ctype:
            return True
        if url.lower().endswith(".pdf"):
            return True
        return data[:5] == b"%PDF-"

    @staticmethod
    def _is_html(url: str, content_type: str) -> bool:
        ctype: str = (content_type or "").lower()
        return (
            ("text/html" in ctype)
            or ("application/xhtml" in ctype)
            or url.lower().endswith((".html", ".htm"))
        )

    def _html_metadata(
        self,
        html_bytes: bytes
    ) -> tuple[str | None, str | None]:
        soup: BeautifulSoup = BeautifulSoup(html_bytes, "html.parser")
        title: str | None = None

        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        if not title:
            og = (
                soup.find("meta", attrs={"property": "og:title"}) or
                soup.find("meta", attrs={"name": "og:title"})
            )
            if og and og.get("content"):
                title = str(og["content"]).strip()

        if not title:
            tw = soup.find("meta", attrs={"name": "twitter:title"})
            if tw and tw.get("content"):
                title = str(tw["content"]).strip()

        desc: str | None = None
        for attrs in (
            {"name": "description"},
            {"property": "og:description"},
            {"name": "og:description"},
            {"name": "twitter:description"},
        ):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content"):
                desc = str(tag["content"]).strip()
                break

        if not desc:
            p = soup.find("p")
            if p:
                desc = self._clean_snippet(
                    p.get_text(" ", strip=True),
                    self._max_description_chars
                )

        return (title, desc)

    @staticmethod
    def _markdown_title(md: str) -> str | None:
        m = search(r"^#\s+(.+)$", md, flags=MULTILINE)
        return m.group(1).strip() if m else None

    def _markdown_description(self, md: str) -> str | None:
        m = search(
            r"(?im)^\s*(?:#{1,6}\s*)?abstract\s*(?:[:\-—]\s+|\n+)(.+?)(?=\n{2,}|\n\s*#{1,6}\s|\Z)",
            md,
            flags=DOTALL,
        )

        if m:
            return self._clean_snippet(
                m.group(1),
                self._max_description_chars
            )

        paras: list[str] = [
            p.strip() for p in split(r"\n{2,}", md) if len(p.strip()) > 80
        ]
        return self._clean_snippet(
            paras[0],
            self._max_description_chars
        ) if paras else None

    @staticmethod
    def _clean_snippet(text: str, limit: int | None) -> str:
        cleaned = " ".join(text.split())
        return cleaned if limit is None else cleaned[:limit]
