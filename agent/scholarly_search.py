import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "title",
        "authors",
        "year",
        "venue",
        "url",
        "externalIds",
        "isOpenAccess",
        "openAccessPdf",
    ]
)


class ScholarlySearchError(RuntimeError):
    pass


def _paper_from_semantic_scholar(item: dict[str, Any]) -> dict[str, Any]:
    authors = item.get("authors") or []
    author_names = [
        author.get("name", "").strip()
        for author in authors
        if isinstance(author, dict) and author.get("name")
    ]
    ext_ids_raw = item.get("externalIds")
    external_ids: dict[str, Any] = ext_ids_raw if isinstance(ext_ids_raw, dict) else {}
    doi = external_ids.get("DOI") if isinstance(external_ids.get("DOI"), str) else None

    open_access_pdf = item.get("openAccessPdf")
    open_access_url = (
        open_access_pdf.get("url")
        if isinstance(open_access_pdf, dict) and isinstance(open_access_pdf.get("url"), str)
        else None
    )
    url = item.get("url") if isinstance(item.get("url"), str) else None
    if not url and open_access_url:
        url = open_access_url
    if not url and doi:
        url = f"https://doi.org/{doi}"

    return {
        "title": item.get("title") if isinstance(item.get("title"), str) else "",
        "authors": author_names,
        "year": item.get("year") if isinstance(item.get("year"), int) else None,
        "venue": item.get("venue") if isinstance(item.get("venue"), str) else "",
        "doi": doi,
        "url": url or "",
        "open_access": bool(item.get("isOpenAccess")),
    }


def search_semantic_scholar(
    query: str,
    limit: int = 5,
    timeout_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    query_normalized = query.strip()
    if not query_normalized:
        return []

    safe_limit = max(1, min(limit, 20))
    params: dict[str, Any] = {
        "query": query_normalized,
        "limit": safe_limit,
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(SEMANTIC_SCHOLAR_API, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException as exc:
        raise ScholarlySearchError("Semantic Scholar request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        raise ScholarlySearchError(
            f"Semantic Scholar request failed with status {status_code}."
        ) from exc
    except Exception as exc:
        raise ScholarlySearchError(f"Semantic Scholar request failed: {exc}") from exc

    papers_raw = payload.get("data")
    if not isinstance(papers_raw, list):
        logger.warning("Unexpected Semantic Scholar payload: missing 'data' list.")
        return []

    parsed = [_paper_from_semantic_scholar(item) for item in papers_raw if isinstance(item, dict)]
    return [paper for paper in parsed if paper.get("title")]


def format_search_papers_results(papers: list[dict[str, Any]]) -> str:
    if not papers:
        return "No academic papers found for this query."

    lines: list[str] = []
    for index, paper in enumerate(papers, start=1):
        title = paper.get("title") or "Untitled"
        year = paper.get("year")
        year_text = str(year) if isinstance(year, int) else "n/a"
        authors = paper.get("authors") or []
        if isinstance(authors, list) and authors:
            author_text = ", ".join(str(name) for name in authors[:4])
            if len(authors) > 4:
                author_text += ", et al."
        else:
            author_text = "n/a"
        venue = paper.get("venue") or "n/a"
        doi = paper.get("doi") or "n/a"
        url = paper.get("url") or "n/a"
        open_access = "yes" if paper.get("open_access") else "no"

        lines.append(f"{index}. {title} ({year_text})")
        lines.append(f"   Authors: {author_text}")
        lines.append(f"   Venue: {venue}")
        lines.append(f"   DOI: {doi}")
        lines.append(f"   URL: {url}")
        lines.append(f"   Open Access: {open_access}")

    return "\n".join(lines)
