import pytest
from agent.tools.registry import ToolRegistry

class DummyTool:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description

def test_tool_registry_search():
    registry = ToolRegistry()
    registry.register("search_document", DummyTool("search_document", "Search uploaded paper content"))
    registry.register("search_web", DummyTool("search_web", "Search the internet for current events"))
    registry.register("ask_human", DummyTool("ask_human", "Ask human for help"))
    
    # Keyword search match
    results = registry.search("paper", top_k=2)
    assert len(results) >= 1
    assert results[0].name == "search_document"
    
    # Name search match
    results = registry.search("web", top_k=2)
    assert len(results) >= 1
    assert results[0].name == "search_web"
    
    # Exact name match points test
    results = registry.search("ask_human", top_k=1)
    assert len(results) == 1
    assert results[0].name == "ask_human"
