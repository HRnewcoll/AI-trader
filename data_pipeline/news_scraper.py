"""
News scraper — NO API KEY required by default.
Uses RSS feeds from major financial sites (Reuters, FXStreet, ForexLive, etc.).
Optional: NewsAPI if key provided.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import List

import feedparser
import requests

logger = logging.getLogger(__name__)

DEFAULT_RSS_FEEDS = [
    ("reuters",    "https://feeds.reuters.com/reuters/businessNews"),
    ("investing",  "https://www.investing.com/rss/news.rss"),
    ("forexlive",  "https://www.forexlive.com/feed/news"),
    ("fxstreet",   "https://www.fxstreet.com/rss"),
    ("dailyfx",    "https://www.dailyfx.com/feeds/all"),
    ("wsj_markets","https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("ft_markets", "https://www.ft.com/rss/home/uk"),
    ("cnbc_intl",  "https://www.cnbc.com/id/100727362/device/rss/rss.html"),
    ("cnn_money",  "https://rss.cnn.com/rss/money_news_international.rss"),
    ("seekingalpha","https://seekingalpha.com/feed.xml"),
    ("marketwatch","https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("yahoo_fin",  "https://finance.yahoo.com/news/rssindex"),
]

FOREX_KEYWORDS = [
    "forex", "fx", "currency", "exchange rate", "dollar", "euro",
    "pound", "yen", "fed", "ecb", "boe", "interest rate", "inflation",
    "cpi", "nfp", "employment", "gdp", "central bank", "rate hike",
    "rate cut", "hawkish", "dovish", "usd", "eur", "gbp", "jpy",
    "aud", "cad", "nzd", "chf",
]


def _is_relevant(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower()
    return any(kw in text for kw in FOREX_KEYWORDS)


def _entry_id(entry: dict) -> str:
    return hashlib.md5(entry.get("link", entry.get("title", "")).encode()).hexdigest()


def fetch_rss_feeds(
    feeds: list[tuple[str, str]] | None = None,
    max_age_hours: int = 24,
    filter_forex: bool = True,
) -> list[dict]:
    """
    Fetch news articles from RSS feeds.
    No API key needed.
    Returns list of dicts: {id, source, title, summary, link, published, timestamp}
    """
    if feeds is None:
        feeds = DEFAULT_RSS_FEEDS

    articles = []
    seen_ids: set[str] = set()
    cutoff = time.time() - max_age_hours * 3600

    for source_name, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                article_id = _entry_id(entry)
                if article_id in seen_ids:
                    continue
                seen_ids.add(article_id)

                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link = entry.get("link", "")

                # Parse published time
                pub_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub_struct:
                    ts = time.mktime(pub_struct)
                else:
                    ts = time.time()

                if ts < cutoff:
                    continue

                if filter_forex and not _is_relevant(title, summary):
                    continue

                articles.append({
                    "id": article_id,
                    "source": source_name,
                    "title": title,
                    "summary": summary[:500],
                    "link": link,
                    "published": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "timestamp": ts,
                })
        except Exception as e:
            logger.warning("RSS fetch error [%s]: %s", source_name, e)

    articles.sort(key=lambda x: x["timestamp"], reverse=True)
    logger.info("Fetched %d forex-relevant news articles from RSS", len(articles))
    return articles


def fetch_newsapi(
    query: str = "forex OR currency exchange",
    api_key: str = "",
    max_results: int = 100,
) -> list[dict]:
    """
    Optional: NewsAPI (requires free API key from newsapi.org).
    Falls back to RSS if no key provided.
    """
    if not api_key:
        logger.info("No NewsAPI key — using RSS feeds instead")
        return fetch_rss_feeds()

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(max_results, 100),
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for art in data.get("articles", []):
            articles.append({
                "id": _entry_id({"link": art.get("url", "")}),
                "source": art.get("source", {}).get("name", "newsapi"),
                "title": art.get("title", ""),
                "summary": art.get("description", "")[:500],
                "link": art.get("url", ""),
                "published": art.get("publishedAt", ""),
                "timestamp": time.time(),
            })
        logger.info("NewsAPI: fetched %d articles", len(articles))
        return articles
    except Exception as e:
        logger.error("NewsAPI error: %s — falling back to RSS", e)
        return fetch_rss_feeds()
