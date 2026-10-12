"""Tavily 搜索客户端，不包含 Agent 或前端逻辑。"""

import requests

from app.config import config

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def search(query: str) -> list[dict]:
    if not config.tavily_api_key:
        raise WebSearchError("missing_api_key", "联网搜索未配置，请设置 TAVILY_API_KEY。")

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {config.tavily_api_key}"},
            json={"query": query, "max_results": config.tavily_max_results},
            timeout=config.tavily_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise WebSearchError("timeout", "联网搜索请求超时，请稍后重试。") from exc
    except requests.RequestException as exc:
        raise WebSearchError("request_error", "联网搜索服务暂不可用，请稍后重试。") from exc
    except ValueError as exc:
        raise WebSearchError("invalid_response", "联网搜索结果格式异常，请稍后重试。") from exc

    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        raise WebSearchError("invalid_response", "联网搜索结果格式异常，请稍后重试。")

    return [
        {
            "title": str(item.get("title") or "未命名网页"),
            "url": str(item.get("url") or ""),
            "content": str(item.get("content") or ""),
            "score": float(item.get("score") or 0),
        }
        for item in raw_results[: config.tavily_max_results]
        if isinstance(item, dict) and item.get("url")
    ]
