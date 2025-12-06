import requests
import time
import csv
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {"User-Agent": "NewsTrendsBot/1.0 (+https://example.com)"}
DATA_DIR = "../data"
OUTFILE = os.path.join(DATA_DIR, "raw_news_data.csv")

def fetch_hn_items(target=2000, sleep=0.05):
    ids_url = "https://hacker-news.firebaseio.com/v0/newstories.json"
    r = requests.get(ids_url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    ids = r.json()
    rows = []
    for i, id_ in enumerate(tqdm(ids, desc="HN ids")):
        if len(rows) >= target:
            break
        try:
            itm = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{id_}.json", headers=HEADERS, timeout=10).json()
            if not itm:
                continue
            title = itm.get("title", "")
            url = itm.get("url", "")
            score = itm.get("score", 0)
            descendants = itm.get("descendants", 0)
            t = itm.get("time", None)  # unix timestamp
            rows.append({
                "source": "hackernews",
                "title": title,
                "body": "",
                "url": url,
                "score": score,
                "num_comments": descendants,
                "timestamp": t
            })
        except Exception:
            pass
        time.sleep(sleep)
    return rows

def fetch_reddit(limit=1000, sub="news", max_pages=20, sleep=0.5):
    rows = []
    base = f"https://www.reddit.com/r/{sub}/hot.json"
    params = {"limit": 100}
    after = None
    for _ in range(max_pages):
        if after:
            params["after"] = after
        try:
            r = requests.get(base, headers=HEADERS, params=params, timeout=15)
            if r.status_code != 200:
                break
            j = r.json()
            children = j.get("data", {}).get("children", [])
            if not children:
                break
            for child in children:
                d = child.get("data", {})
                rows.append({
                    "source": "reddit",
                    "title": d.get("title", ""),
                    "body": d.get("selftext", ""),
                    "url": "https://reddit.com" + d.get("permalink", ""),
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "timestamp": d.get("created_utc", None)
                })
            after = j.get("data", {}).get("after", None)
            if not after:
                break
        except Exception:
            break
        time.sleep(sleep)
    return rows

def scrape_bbc(max_pages=5, sleep=0.5):
    rows = []
    base = "https://www.bbc.com/news"
    try:
        r = requests.get(base, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        anchors = soup.select("a.gs-c-promo-heading")
        for a in anchors:
            title = a.get_text(strip=True)
            href = a.get("href", "")
            if href and href.startswith("/"):
                url = "https://www.bbc.com" + href
            else:
                url = href
            rows.append({
                "source": "bbc",
                "title": title,
                "body": "",
                "url": url,
                "score": None,
                "num_comments": None,
                "timestamp": None
            })
            if len(rows) >= max_pages * 50:
                break
    except Exception:
        pass
    return rows

def collect_all(target_total=3500):
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    print("Fetching Hacker News items...")
    hn_rows = fetch_hn_items(target=2200)
    print("HN fetched:", len(hn_rows))
    rows.extend(hn_rows)

    print("Fetching Reddit hot posts (r/news)...")
    reddit_rows = fetch_reddit(limit=1000, max_pages=30)
    print("Reddit fetched:", len(reddit_rows))
    rows.extend(reddit_rows)

    print("Scraping BBC...")
    bbc_rows = scrape_bbc(max_pages=5)
    print("BBC fetched:", len(bbc_rows))
    rows.extend(bbc_rows)

    seen = set()
    final = []
    for r in rows:
        key = (r.get("source",""), (r.get("title") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        final.append(r)
        if len(final) >= target_total:
            break

    print("Total collected (after dedupe):", len(final))

    fieldnames = ["source","title","body","url","score","num_comments","timestamp"]
    with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in final:
            writer.writerow(item)
    print("Saved to", OUTFILE)
    return final

if __name__ == "__main__":
    collect_all(target_total=3500)
