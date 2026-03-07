"""
Twitter/X scraper — NO API KEY required by default.
Uses Nitter (open-source Twitter front-end) instances for scraping.
Optional: Official Twitter/X API v2 if bearer token provided.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import List
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Public Nitter instances (no API key needed) — try each until one works
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.catsarch.com",
    "https://nitter.kavin.rocks",
]

FOREX_QUERIES = [
    "$EURUSD", "$GBPUSD", "$USDJPY",
    "forex trading", "forextrading", "EUR/USD",
    "GBP/USD", "USD/JPY", "central bank",
    "rate hike", "rate cut", "Fed", "ECB", "BOE",
]


def _scrape_nitter_search(
    query: str,
    nitter_base: str,
    max_tweets: int = 20,
) -> list[dict]:
    """Scrape a Nitter instance for a search query."""
    url = f"{nitter_base}/search"
    params = {"q": query, "f": "tweets"}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.debug("Nitter [%s] error for query '%s': %s", nitter_base, query, e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    tweet_items = soup.select("div.timeline-item")[:max_tweets]

    tweets = []
    for item in tweet_items:
        try:
            content_el = item.select_one("div.tweet-content")
            text = content_el.get_text(strip=True) if content_el else ""

            time_el = item.select_one("span.tweet-date a")
            ts_str = time_el.get("title", "") if time_el else ""
            try:
                ts = datetime.strptime(ts_str, "%b %d, %Y · %I:%M %p UTC")
                ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(tz=timezone.utc)

            stats = item.select("span.tweet-stat")
            replies = comments = likes = retweets = 0
            for stat in stats:
                txt = stat.get_text(strip=True).lower().replace(",", "")
                try:
                    val = int(re.search(r"\d+", txt).group()) if re.search(r"\d+", txt) else 0
                    if "retweet" in txt:
                        retweets = val
                    elif "like" in txt or "heart" in txt:
                        likes = val
                    elif "comment" in txt or "repl" in txt:
                        comments = val
                except Exception:
                    pass

            username_el = item.select_one("a.username")
            username = username_el.get_text(strip=True) if username_el else "unknown"

            tweets.append({
                "id": re.sub(r"\W", "", username + str(ts.timestamp())),
                "text": text,
                "username": username,
                "timestamp": ts.timestamp(),
                "published": ts.isoformat(),
                "likes": likes,
                "retweets": retweets,
                "comments": comments,
                "query": query,
                "source": "nitter",
            })
        except Exception:
            continue

    return tweets


def _try_nitter_instances(query: str, max_tweets: int = 20) -> list[dict]:
    """Try each Nitter instance until one works."""
    for instance in NITTER_INSTANCES:
        tweets = _scrape_nitter_search(query, instance, max_tweets)
        if tweets:
            logger.debug("Nitter success via %s", instance)
            return tweets
    logger.warning("All Nitter instances failed for query: %s", query)
    return []


def _fetch_twitter_api_v2(
    query: str,
    bearer_token: str,
    max_results: int = 20,
) -> list[dict]:
    """Official Twitter API v2 (requires bearer token)."""
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": f"{query} lang:en -is:retweet",
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,public_metrics,author_id",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        tweets = []
        for tw in data.get("data", []):
            metrics = tw.get("public_metrics", {})
            ts_str = tw.get("created_at", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(tz=timezone.utc)

            tweets.append({
                "id": tw.get("id", ""),
                "text": tw.get("text", ""),
                "username": tw.get("author_id", ""),
                "timestamp": ts.timestamp(),
                "published": ts.isoformat(),
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "comments": metrics.get("reply_count", 0),
                "query": query,
                "source": "twitter_api",
            })
        return tweets
    except Exception as e:
        logger.warning("Twitter API error: %s — falling back to Nitter", e)
        return _try_nitter_instances(query, max_results)


def fetch_tweets(
    queries: list[str] | None = None,
    bearer_token: str = "",
    max_per_query: int = 20,
) -> list[dict]:
    """
    Main entry. Auto-selects:
    - Twitter API v2 if bearer_token provided
    - Nitter scraping otherwise (no API key needed)
    """
    if queries is None:
        queries = FOREX_QUERIES

    all_tweets = []
    seen_ids: set[str] = set()

    for query in queries:
        if bearer_token:
            tweets = _fetch_twitter_api_v2(query, bearer_token, max_per_query)
        else:
            tweets = _try_nitter_instances(query, max_per_query)
            time.sleep(1.0)  # respectful delay

        for tw in tweets:
            if tw["id"] not in seen_ids:
                seen_ids.add(tw["id"])
                all_tweets.append(tw)

    logger.info("Twitter/X: fetched %d tweets (key=%s)", len(all_tweets), bool(bearer_token))
    return all_tweets
