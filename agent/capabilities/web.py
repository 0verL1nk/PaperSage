from typing import Any

from ..tools.paper_search import search_papers
from ..tools.web_search import search_web


def build_web_tools(_deps: Any) -> list[Any]:
    return [search_web, search_papers]
