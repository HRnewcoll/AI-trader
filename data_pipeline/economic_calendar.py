"""
Economic calendar scraper — NO API KEY required.
Scrapes Forex Factory and Investing.com for event data + surprise indices.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

PAIR_TO_CURRENCY = {
    "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"], "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"], "NZDUSD": ["NZD", "USD"],
    "USDCHF": ["USD", "CHF"],
}

HIGH_IMPACT_KEYWORDS = [
    "interest rate", "nfp", "non-farm", "cpi", "inflation", "gdp",
    "employment", "fomc", "ecb", "boe", "rba", "boc", "snb",
    "fed", "chair", "governor", "rate decision", "pmi", "ism",
]


def _is_high_impact(event_name: str) -> bool:
    name_lower = event_name.lower()
    return any(kw in name_lower for kw in HIGH_IMPACT_KEYWORDS)


def scrape_forexfactory(days_ahead: int = 3) -> pd.DataFrame:
    """Scrape Forex Factory calendar without API key."""
    url = "https://www.forexfactory.com/calendar"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("ForexFactory scrape failed: %s", e)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tr.calendar__row")

    events = []
    current_date = datetime.utcnow().date()

    for row in rows:
        try:
            impact_td = row.select_one("td.calendar__impact")
            if not impact_td:
                continue

            # Impact level from icon class
            impact_icon = impact_td.select_one("span")
            impact = "low"
            if impact_icon:
                cls = impact_icon.get("class", [])
                if any("high" in c for c in cls):
                    impact = "high"
                elif any("medium" in c for c in cls):
                    impact = "medium"

            currency_td = row.select_one("td.calendar__currency")
            currency = currency_td.get_text(strip=True) if currency_td else ""

            event_td = row.select_one("td.calendar__event")
            event_name = event_td.get_text(strip=True) if event_td else ""

            actual_td = row.select_one("td.calendar__actual")
            forecast_td = row.select_one("td.calendar__forecast")

            actual = actual_td.get_text(strip=True) if actual_td else ""
            forecast = forecast_td.get_text(strip=True) if forecast_td else ""

            events.append({
                "date": str(current_date),
                "currency": currency,
                "event": event_name,
                "impact": impact,
                "actual": actual,
                "forecast": forecast,
                "high_impact": _is_high_impact(event_name),
            })
        except Exception:
            continue

    df = pd.DataFrame(events)
    logger.info("ForexFactory: %d events scraped", len(df))
    return df


def compute_surprise_index(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute economic surprise index:
    surprise = actual - forecast (normalised by historical std).
    """
    if events_df.empty:
        return events_df

    def _to_float(val: str) -> float | None:
        val = re.sub(r"[%KMBkm,]", "", str(val).strip())
        try:
            return float(val)
        except ValueError:
            return None

    events_df = events_df.copy()
    events_df["actual_f"] = events_df["actual"].apply(_to_float)
    events_df["forecast_f"] = events_df["forecast"].apply(_to_float)

    mask = events_df["actual_f"].notna() & events_df["forecast_f"].notna()
    events_df.loc[mask, "surprise"] = (
        events_df.loc[mask, "actual_f"] - events_df.loc[mask, "forecast_f"]
    )

    # Normalise per event type
    for event_name, grp in events_df[mask].groupby("event"):
        std = grp["surprise"].std()
        if std and std > 0:
            events_df.loc[grp.index, "surprise_norm"] = grp["surprise"] / std

    return events_df


def get_upcoming_events(pairs: list[str], cfg: dict | None = None) -> pd.DataFrame:
    """Main entry: scrape calendar and filter for relevant pairs."""
    df = scrape_forexfactory()
    if df.empty:
        return df

    currencies = set()
    for p in pairs:
        base, quote = p[:3], p[3:]
        currencies.add(base)
        currencies.add(quote)

    df = df[df["currency"].isin(currencies)]
    df = compute_surprise_index(df)
    return df.reset_index(drop=True)
