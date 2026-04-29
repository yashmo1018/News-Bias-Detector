"""
Layer 2 — Analyzer (Groq edition)
Sends each article to Llama-3.3-70b via Groq for bias/sentiment/framing analysis.
Outputs: analyzed.json  (articles.json fields + analysis fields merged)

Setup:
    pip install groq
    $env:GROQ_API_KEY = "gsk_..."   # get free key at console.groq.com

Usage:
    python analyzer.py                  # analyze all unanalyzed articles
    python analyzer.py --reanalyze      # force re-analyze everything
"""

import json
import os
import time
import argparse
from pathlib import Path
from groq import Groq, AuthenticationError, RateLimitError, APIError

# ── Config ────────────────────────────────────────────────────────────────────
ARTICLES_FILE  = "articles.json"
ANALYZED_FILE  = "analyzed.json"
MAX_TEXT_CHARS = 3000
MODEL          = "llama-3.3-70b-versatile"
RETRY_LIMIT    = 3
RETRY_DELAY    = 10    # seconds between retries

SYSTEM_PROMPT = """You are a professional media-bias analyst.
Your job is to read a news article and return a structured JSON object — nothing else.
No markdown fences, no explanation, no commentary outside the JSON.

The JSON must strictly follow this schema:
{
  "bias_label":      string,   // one of: "Left", "Left-Center", "Center", "Right-Center", "Right", "Unknown"
  "sentiment_score": float,    // -1.0 (very negative) to +1.0 (very positive)
  "framing":         string,   // one of: "Alarmist", "Neutral", "Optimistic", "Analytical", "Sensationalist", "Sympathetic", "Critical"
  "key_entities": [            // up to 5 most prominent people / orgs / policies
    {
      "name":      string,
      "portrayal": string      // one of: "Positive", "Neutral", "Negative"
    }
  ],
  "reasoning": string          // <= 60 words explaining your labels
}"""

USER_TEMPLATE = """Source outlet: {source}
Article title: {title}

Article text (may be truncated):
{text}

Return only the JSON object."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> list:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not text:
        return ""
    return text[:max_chars] + ("…" if len(text) > max_chars else "")

def parse_analysis(raw: str) -> dict | None:
    """Parse JSON — Groq's json_object mode makes this almost always clean."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    ⚠  JSON parse error: {e}")
        return None

# ── Core analysis call ────────────────────────────────────────────────────────

def analyze_article(client: Groq, article: dict) -> dict | None:
    prompt = USER_TEMPLATE.format(
        source=article.get("source", "Unknown"),
        title=article.get("title", ""),
        text=truncate(article.get("text", "")),
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=512,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = parse_analysis(raw)
            if result:
                return result
            print(f"    ✗ Bad JSON on attempt {attempt}, retrying…")

        except AuthenticationError:
            raise SystemExit(
                "\n✗ Authentication failed — GROQ_API_KEY is invalid.\n"
                "  Get your free key at: https://console.groq.com\n"
                "  Then run: $env:GROQ_API_KEY = 'gsk_...'"
            )
        except RateLimitError:
            wait = RETRY_DELAY * attempt
            print(f"    ⏳ Rate limited — waiting {wait}s (attempt {attempt})")
            time.sleep(wait)
        except APIError as e:
            print(f"    ✗ API error: {e} (attempt {attempt})")
            time.sleep(RETRY_DELAY)

    return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reanalyze", action="store_true",
                        help="Re-analyze all articles even if already cached")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "✗ GROQ_API_KEY not set.\n"
            "  Get your free key at: https://console.groq.com\n"
            "  Then run: $env:GROQ_API_KEY = 'gsk_...'"
        )

    client = Groq(api_key=api_key)

    articles = load_json(ARTICLES_FILE)
    if not articles:
        print(f"✗ No articles found in {ARTICLES_FILE}. Run scraper.py first.")
        return

    analyzed  = load_json(ANALYZED_FILE)
    done_urls: set[str] = set()
    if not args.reanalyze:
        done_urls = {a["url"] for a in analyzed if "bias_label" in a}

    to_process = [a for a in articles if a["url"] not in done_urls]
    print(f"📰 Articles total: {len(articles)}  |  "
          f"Already analyzed: {len(done_urls)}  |  "
          f"To process: {len(to_process)}")

    if not to_process:
        print("✅ Nothing new to analyze. Use --reanalyze to force refresh.")
        return

    result_map: dict[str, dict] = {a["url"]: a for a in analyzed}

    for i, article in enumerate(to_process, 1):
        title_preview = article.get("title", "")[:60]
        print(f"[{i}/{len(to_process)}] {article.get('source','?')} — {title_preview}…")

        analysis = analyze_article(client, article)
        if analysis:
            merged = {**article, **analysis}
            result_map[article["url"]] = merged
            score = analysis.get("sentiment_score", 0)
            print(f"    ✓ {analysis.get('bias_label')} | "
                  f"sentiment={score:+.2f} | "
                  f"framing={analysis.get('framing')}")
        else:
            result_map[article["url"]] = {**article, "analysis_error": True}
            print("    ✗ Analysis failed — stored with error flag")

        save_json(ANALYZED_FILE, list(result_map.values()))
        time.sleep(1)

    final = load_json(ANALYZED_FILE)
    ok    = sum(1 for a in final if "bias_label" in a)
    err   = sum(1 for a in final if a.get("analysis_error"))
    print(f"\n✅ Done. {ok} analyzed, {err} errors → {ANALYZED_FILE}")

if __name__ == "__main__":
    main()
