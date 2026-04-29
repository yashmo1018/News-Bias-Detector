import feedparser
import newspaper
import json
import os
import time
from datetime import datetime, timezone, timedelta

RSS_FEEDS = {
    "Times of India":   "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "ABC Australia":        "https://www.abc.net.au/news/feed/51120/rss.xml",
    "India Today":      "https://www.indiatoday.in/rss/1206571",
    "The Guardian":     "https://www.theguardian.com/world/rss",
    "The Hindu":        "https://www.thehindu.com/news/national/feeder/default.rss",
    "BBC News":         "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera":       "https://www.aljazeera.com/xml/rss/all.xml",
    "NDTV":             "https://feeds.feedburner.com/ndtvnews-top-stories",
    "Indian Express":   "https://indianexpress.com/feed/",
}

CACHE_FILE    = "articles.json"
MAX_PER_SOURCE = 20       # keep it small during dev — saves API credits
ROLLING_DAYS  = 7        # drop articles older than this many days


def now_utc() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def is_within_window(scraped_at: str, days: int = ROLLING_DAYS) -> bool:
    """Return True if the article's scraped_at timestamp is within the rolling window."""
    try:
        dt = datetime.fromisoformat(scraped_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return dt >= cutoff
    except (ValueError, TypeError):
        return True  # keep articles with unparseable dates


def prune_old_articles(articles: list[dict], days: int = ROLLING_DAYS) -> tuple[list[dict], int]:
    """Remove articles older than `days`. Returns (pruned_list, removed_count)."""
    before = len(articles)
    kept = [a for a in articles if is_within_window(a.get("scraped_at", now_utc()), days)]
    return kept, before - len(kept)


def scrape_article(url: str) -> str:
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        return ""


def fetch_articles(force_refresh: bool = False) -> list[dict]:
    """
    Scrape all RSS feeds and merge with existing articles.json.

    - New articles get a `scraped_at` timestamp (UTC ISO 8601).
    - Articles older than ROLLING_DAYS are dropped during merge.
    - Deduplication is by URL.

    Args:
        force_refresh: If False, returns cache as-is without scraping.
                       If True, always scrapes and merges.
    """
    if not force_refresh and os.path.exists(CACHE_FILE):
        print("Loading from cache (no scrape)...")
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)

    # ── Load existing articles ────────────────────────────────────────────────
    existing: list[dict] = []
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            existing = json.load(f)

    existing_urls = {a["url"] for a in existing}

    # ── Scrape new articles ───────────────────────────────────────────────────
    new_articles: list[dict] = []
    for source, feed_url in RSS_FEEDS.items():
        print(f"Fetching {source}...")
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:MAX_PER_SOURCE]:
            link = entry.get("link", "")
            if not link or link in existing_urls:
                continue  # skip duplicates immediately

            text = scrape_article(link)
            if len(text) < 100:   # skip stubs / paywalled articles
                continue

            new_articles.append({
                "source":     source,
                "title":      entry.get("title", ""),
                "url":        link,
                "text":       text[:3000],
                "scraped_at": now_utc(),
            })
            existing_urls.add(link)
            time.sleep(0.5)  # polite delay

    # ── Backfill scraped_at for legacy articles that don't have it ────────────
    for a in existing:
        if "scraped_at" not in a:
            a["scraped_at"] = now_utc()   # treat as "just seen" — won't be pruned yet

    # ── Merge + prune ─────────────────────────────────────────────────────────
    merged = existing + new_articles
    merged, pruned_count = prune_old_articles(merged)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Scraped {len(new_articles)} new articles.")
    if pruned_count:
        print(f"Pruned {pruned_count} articles older than {ROLLING_DAYS} days.")
    print(f"Total in articles.json: {len(merged)}")
    return merged


if __name__ == "__main__":
    data = fetch_articles(force_refresh=True)
    if data:
        print(data[0])  # sanity check
