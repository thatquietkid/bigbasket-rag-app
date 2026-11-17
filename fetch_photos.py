"""
fetch_photos.py — download one real photo per product type from Pexels.

Run ONCE on your computer (it needs internet), then commit the photos:

    PEXELS_API_KEY=your_key python3 fetch_photos.py
    git add static/photos && git commit -m "Add product photos" && git push

WHY Pexels: its licence allows free use of the photos in apps, with no
permission needed (credit is appreciated, so we save it). Photos from
store websites or Google Images belong to other companies and can't be reused.

WHAT it does:
  - For each product type in product_art.PHOTO_FOR (cereal, dal, shampoo...),
    searches Pexels, takes the best square-ish result, saves it as
    static/photos/<slug>.jpg (~600px, small file).
  - Writes static/photos/credits.json with photographer + link for each.
  - Skips photos that already exist, so you can re-run it safely.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

from product_art import PHOTO_FOR

API = "https://api.pexels.com/v1/search"
OUT = os.path.join("static", "photos")


def search(query, key):
    url = API + "?" + urllib.parse.urlencode({"query": query, "per_page": 5, "orientation": "square"})
    req = urllib.request.Request(url, headers={"Authorization": key, "User-Agent": "BasketAssistant/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r).get("photos", [])


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "BasketAssistant/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(path, "wb") as f:
        f.write(r.read())


def main():
    key = os.getenv("PEXELS_API_KEY", "").strip()
    if not key:
        sys.exit("Set PEXELS_API_KEY first, e.g.  PEXELS_API_KEY=xxxx python3 fetch_photos.py")
    os.makedirs(OUT, exist_ok=True)
    credits_path = os.path.join(OUT, "credits.json")
    credits = json.load(open(credits_path)) if os.path.exists(credits_path) else {}

    items = sorted({slug: q for slug, q in PHOTO_FOR.values()}.items())
    for n, (slug, query) in enumerate(items, 1):
        path = os.path.join(OUT, f"{slug}.jpg")
        if os.path.exists(path):
            print(f"[{n}/{len(items)}] {slug}: already there")
            continue
        try:
            photos = search(query, key)
            if not photos:
                print(f"[{n}/{len(items)}] {slug}: no result for '{query}'")
                continue
            p = photos[0]
            # "medium" (~350px tall) is plenty for a card and keeps the repo small
            download(p["src"].get("medium") or p["src"]["large"], path)
            credits[slug] = {"photographer": p.get("photographer"), "pexels_url": p.get("url"),
                             "query": query}
            print(f"[{n}/{len(items)}] {slug}: saved (by {p.get('photographer')})")
            time.sleep(0.4)   # be gentle with the free API
        except Exception as e:
            print(f"[{n}/{len(items)}] {slug}: failed ({e})")
    json.dump(credits, open(credits_path, "w"), indent=2)
    print(f"\nDone. Photos are in {OUT}/ — now commit and push them.")


if __name__ == "__main__":
    main()
