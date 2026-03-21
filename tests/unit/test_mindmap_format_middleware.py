from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.middlewares.mindmap_format import MindmapFormatMiddleware


def test_mindmap_format_middleware_retries_invalid_mermaid_output() -> None:
    middleware = MindmapFormatMiddleware()
    requests: list[ModelRequest[None]] = []

    def _handler(request: ModelRequest[None]) -> ModelResponse[None]:
        requests.append(request)
        if len(requests) == 1:
            return ModelResponse(
                result=[
                    AIMessage(
                        content="## 思维导图\n```mermaid\nmindmap\n  root((Seed-TTS))\n```"
                    )
                ]
            )
        return ModelResponse(
            result=[
                AIMessage(
                    content='<mindmap>{"name":"Seed-TTS","children":[]}</mindmap>'
                )
            ]
        )

    response = middleware.wrap_model_call(
        ModelRequest(
            model="llm",  # type: ignore[arg-type]
            messages=[HumanMessage(content="请给我一份 Seed-TTS 的思维导图")],
            system_message=SystemMessage(content="sys"),
        ),
        _handler,
    )

    assert len(requests) == 2
    assert response.result[0].content == '<mindmap>{"name":"Seed-TTS","children":[]}</mindmap>'
    retry_messages = requests[1].messages
    assert isinstance(retry_messages[-1], HumanMessage)
    assert "必须只输出一个 `<mindmap>...</mindmap>` 包裹的 JSON 对象" in retry_messages[-1].content
    assert "禁止输出 Mermaid" in retry_messages[-1].content


def test_mindmap_format_middleware_passes_valid_tagged_json_without_retry() -> None:
    middleware = MindmapFormatMiddleware()
    calls = {"count": 0}

    def _handler(request: ModelRequest[None]) -> ModelResponse[None]:
        calls["count"] += 1
        return ModelResponse(
            result=[AIMessage(content='<mindmap>{"name":"主题","children":[]}</mindmap>')]
        )

    response = middleware.wrap_model_call(
        ModelRequest(
            model="llm",  # type: ignore[arg-type]
            messages=[HumanMessage(content="请生成思维导图")],
            system_message=SystemMessage(content="sys"),
        ),
        _handler,
    )

    assert calls["count"] == 1
    assert response.result[0].content == '<mindmap>{"name":"主题","children":[]}</mindmap>'


def test_mindmap_format_middleware_returns_failure_message_after_retry_exhausted() -> None:
    middleware = MindmapFormatMiddleware()
    calls = {"count": 0}

    def _handler(request: ModelRequest[None]) -> ModelResponse[None]:
        calls["count"] += 1
        return ModelResponse(result=[AIMessage(content="```mermaid\nmindmap\n  root((X))\n```")])

    response = middleware.wrap_model_call(
        ModelRequest(
            model="llm",  # type: ignore[arg-type]
            messages=[HumanMessage(content="请生成思维导图")],
            system_message=SystemMessage(content="sys"),
        ),
        _handler,
    )

    assert calls["count"] == 2
    assert "思维导图输出格式校验失败" in str(response.result[0].content)


def test_mindmap_format_middleware_does_not_retry_regular_text_output() -> None:
    middleware = MindmapFormatMiddleware()
    calls = {"count": 0}

    def _handler(request: ModelRequest[None]) -> ModelResponse[None]:
        calls["count"] += 1
        return ModelResponse(result=[AIMessage(content="这是普通总结，不是思维导图。")])

    response = middleware.wrap_model_call(
        ModelRequest(
            model="llm",  # type: ignore[arg-type]
            messages=[HumanMessage(content="请总结一下论文")],
            system_message=SystemMessage(content="sys"),
        ),
        _handler,
    )

    assert calls["count"] == 1
    assert response.result[0].content == "这是普通总结，不是思维导图。"
