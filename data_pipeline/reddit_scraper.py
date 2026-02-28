"""
Reddit scraper — NO API KEY required by default.
Scrapes Reddit HTML directly (no PRAW OAuth needed).
Optional: PRAW with API keys for higher rate limits.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

FOREX_SUBREDDITS = ["forex", "algotrading", "wallstreetbets", "investing", "FX_Traders"]


def _scrape_subreddit_json(
    subreddit: str,
    sort: str = "hot",
    limit: int = 25,
) -> list[dict]:
    """
    Use Reddit's public JSON API (no key required).
    Reddit exposes /r/{sub}/{sort}.json without auth for public subs.
    """
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params = {"limit": limit, "raw_json": 1}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 429:
            logger.warning("Reddit rate limited for r/%s — sleeping 30s", subreddit)
            time.sleep(30)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Reddit JSON scrape error [r/%s]: %s", subreddit, e)
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        posts.append({
            "id": post.get("id", ""),
            "subreddit": subreddit,
            "title": post.get("title", ""),
            "text": post.get("selftext", "")[:500],
            "score": post.get("score", 0),
            "comments": post.get("num_comments", 0),
            "url": f"https://reddit.com{post.get('permalink', '')}",
            "timestamp": post.get("created_utc", time.time()),
            "published": datetime.fromtimestamp(
                post.get("created_utc", time.time()), tz=timezone.utc
            ).isoformat(),
        })

    return posts


def _scrape_subreddit_praw(
    subreddit: str,
    client_id: str,
    client_secret: str,
    user_agent: str,
    limit: int = 25,
) -> list[dict]:
    """PRAW scrape — used when API keys are available (higher limits)."""
    try:
        import praw
    except ImportError:
        logger.warning("praw not installed — using JSON scrape")
        return _scrape_subreddit_json(subreddit, limit=limit)

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        sub = reddit.subreddit(subreddit)
        posts = []
        for post in sub.hot(limit=limit):
            posts.append({
                "id": post.id,
                "subreddit": subreddit,
                "title": post.title,
                "text": post.selftext[:500],
                "score": post.score,
                "comments": post.num_comments,
                "url": f"https://reddit.com{post.permalink}",
                "timestamp": post.created_utc,
                "published": datetime.fromtimestamp(
                    post.created_utc, tz=timezone.utc
                ).isoformat(),
            })
        return posts
    except Exception as e:
        logger.warning("PRAW error: %s — falling back to JSON scrape", e)
        return _scrape_subreddit_json(subreddit, limit=limit)


def fetch_reddit_posts(
    subreddits: list[str] | None = None,
    limit_per_sub: int = 25,
    client_id: str = "",
    client_secret: str = "",
    user_agent: str = "ForexAI/1.0",
) -> list[dict]:
    """
    Main entry. Auto-selects:
    - PRAW if API keys provided
    - Reddit public JSON API otherwise (no keys needed)
    """
    if subreddits is None:
        subreddits = FOREX_SUBREDDITS

    use_praw = bool(client_id and client_secret)
    all_posts = []

    for sub in subreddits:
        if use_praw:
            posts = _scrape_subreddit_praw(sub, client_id, client_secret, user_agent, limit_per_sub)
        else:
            posts = _scrape_subreddit_json(sub, limit=limit_per_sub)
            # Small delay to be respectful to Reddit's public API
            time.sleep(1.5)

        all_posts.extend(posts)

    logger.info("Reddit: fetched %d posts from %d subreddits", len(all_posts), len(subreddits))
    return all_posts
