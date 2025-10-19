from typing import List, Dict

import requests  # type: ignore


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Naive web search using DuckDuckGo's HTML results.

    This avoids API keys for the baseline. For production, replace with a proper API.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BaselineAgent/0.1; +https://example.com)"
    }
    params = {"q": query}
    try:
        resp = requests.get("https://duckduckgo.com/html/", params=params, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    # Very rough extraction from HTML results
    results: List[Dict[str, str]] = []
    for part in resp.text.split('<a rel="nofollow" class="result__a" href="')[1:]:
        url = part.split('"', 1)[0]
        rest = part.split('>', 1)[1]
        title = rest.split('<', 1)[0]
        results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break
    return results


