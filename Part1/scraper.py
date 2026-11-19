import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TD3-student-scraper"}
OUT = Path(__file__).parent / "data" / "raw_articles.csv"

HESPRESS = [
    ("hespress", "economy", "https://www.hespress.com/economie/page/{}", 30),
    ("hespress", "society", "https://www.hespress.com/societe/page/{}", 10),
    ("hespress", "sport", "https://www.hespress.com/sport/page/{}", 10),
    ("hespress", "culture", "https://www.hespress.com/art-et-culture/page/{}", 10),
]
ALJAZEERA = [
    ("aljazeera", "economy", "https://www.aljazeera.net/ebusiness/", "ebusiness"),
    ("aljazeera", "sport", "https://www.aljazeera.net/sport/", "sport"),
    ("aljazeera", "culture", "https://www.aljazeera.net/culture/", "culture"),
    ("aljazeera", "society", "https://www.aljazeera.net/lifestyle/", "lifestyle"),
]

session = requests.Session()
session.headers.update(HEADERS)


def get(url, retries=2):
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=12)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return None


def hespress_links(template, pages):
    links = []
    for p in range(1, pages + 1):
        html = get(template.format(p))
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a[href]"):
            href = a["href"]
            if re.match(r"https://www\.hespress\.com/[^/]+-\d+\.html$", href):
                links.append(href)
        time.sleep(0.3)
    return list(dict.fromkeys(links))


def aljazeera_links(url, slug):
    html = get(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    pat = re.compile(rf"^/{slug}/\d{{4}}/\d{{1,2}}/\d{{1,2}}/")
    links = [urljoin(url, a["href"]) for a in soup.select("a[href]") if pat.match(a["href"])]
    return list(dict.fromkeys(links))


def parse_article(url, source):
    html = get(url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.article-content" if source == "hespress" else "div.wysiwyg")
    if body is None:
        return None
    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) > 30]
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    text = "\n".join(paragraphs)
    if len(text) < 200:
        return None
    return {"title": title, "text": text}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for source, section, tpl, pages in HESPRESS:
        links = hespress_links(tpl, pages)
        print(f"{source}/{section}: {len(links)} links", flush=True)
        jobs += [(source, section, u) for u in links]
    for source, section, url, slug in ALJAZEERA:
        links = aljazeera_links(url, slug)
        print(f"{source}/{section}: {len(links)} links", flush=True)
        jobs += [(source, section, u) for u in links]

    seen, rows = set(), []
    jobs = [j for j in jobs if not (j[2] in seen or seen.add(j[2]))]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(parse_article, url, src): (src, sec, url) for src, sec, url in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            src, sec, url = futures[fut]
            art = fut.result()
            if art:
                rows.append({"source": src, "section": sec, "url": url, **art})
            if i % 50 == 0:
                print(f"  parsed {i}/{len(jobs)} -> kept {len(rows)}", flush=True)

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "section", "url", "title", "text"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} articles -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
