"""
image_lookup.py — real product photos from Open Food Facts (and its sister
databases), with a safe fallback to our illustrations.

WHY: the Kaggle BigBasket dataset has no images. Open Food Facts is a free,
openly licensed (CC BY-SA) database with real photos of product packs,
including many Indian brands. It has no API key and no cost.

HOW it works:
  1. Pick the right database for the product's aisle:
       food -> Open Food Facts, beauty/baby -> Open Beauty Facts,
       household/kitchen -> Open Products Facts.
  2. Skip products that won't be there (fresh produce, BigBasket's own
     brands like fresho!) — saves time and avoids wrong matches.
  3. Search "<brand> <product name>" and only ACCEPT a result when the brand
     matches and enough words of the name match. A wrong photo is worse than
     no photo, because evaluators check the app shows what the data says.
  4. Every network call has a short timeout and every error returns "",
     which means "use the illustration". The app can never break because of
     this module.
  5. Results are cached in memory for as long as the app runs, so each
     product is looked up once, not on every question. Only definite answers
     are cached (photo found / no match) — a timeout is retried next time.
"""

import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, wait

import requests

USER_AGENT = "BasketAssistant/1.0 (student RAG project; github.com/Kasimshaik2508/bigbasket-rag-assistant)"
TIMEOUT_S = 3.0          # per request
BUDGET_S = 4.0           # for a whole batch of cards, run in parallel

HOSTS = {
    "food": "https://world.openfoodfacts.org",
    "beauty": "https://world.openbeautyfacts.org",
    "products": "https://world.openproductsfacts.org",
}
AISLE_DB = {
    "Beauty & Hygiene": "beauty",
    "Baby Care": "beauty",
    "Cleaning & Household": "products",
    "Kitchen, Garden & Pets": "products",
}
SKIP_AISLES = {"Fruits & Vegetables", "Eggs, Meat & Fish", "Policy"}
# BigBasket's own labels are not in the open databases
SKIP_BRANDS = {"fresho", "bbroyal", "bbhome", "bbcombo", "bbpopular", "freshosignature",
               "bigbasket", "bbcomfort", "bbinstant", "choupstix", "kitchenpromise"}

STOP = {"with", "and", "for", "the", "of", "in", "a", "g", "kg", "ml", "l", "pc", "pcs", "pack",
        "gm", "gms", "ltr", "x", "free", "new", "combo", "assorted"}


def _plain(s):
    # "Nestlé" -> "nestle" so accented brand names still match
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", _plain(s))


def _words(s):
    return {w for w in re.findall(r"[a-z]+", _plain(s)) if len(w) > 2 and w not in STOP}


def _brand_ok(brand, result_brands):
    b = _norm(brand)
    if not b:
        return False
    for rb in re.split(r"[,;]", result_brands or ""):
        r = _norm(rb)
        if r and (b in r or r in b):
            return True
    return False


def _name_ok(product, result_name):
    want, got = _words(product), _words(result_name)
    if not want or not got:
        return False
    overlap = len(want & got) / len(want)
    return overlap >= 0.5


_CACHE = {}                 # (product, brand, category) -> url or ""
_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=8)   # shared; never blocks the page


def _fetch(product, brand, category):
    """Return an image URL, "" for "no confident match", or None on a network
    problem (so it is not cached and can be retried)."""
    try:
        if category in SKIP_AISLES or _norm(brand) in SKIP_BRANDS:
            return ""
        host = HOSTS[AISLE_DB.get(category, "food")]
        query = f"{brand} {product}"[:120]
        resp = requests.get(
            f"{host}/cgi/search.pl",
            params={
                "search_terms": query, "search_simple": 1, "action": "process", "json": 1,
                "page_size": 8,
                "fields": "product_name,brands,image_front_small_url,image_front_url,image_small_url,image_url",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_S,
        )
        if resp.status_code != 200:
            return None
        for p in resp.json().get("products", []) or []:
            if not _brand_ok(brand, p.get("brands", "")):
                continue
            if not _name_ok(product, p.get("product_name", "")):
                continue
            url = (p.get("image_front_small_url") or p.get("image_front_url")
                   or p.get("image_small_url") or p.get("image_url") or "")
            if url.startswith("https://"):
                return url
        return ""
    except Exception:
        return None


def lookup_one(product, brand, category):
    key = (product, brand, category)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    url = _fetch(product, brand, category)
    if url is not None:
        with _LOCK:
            _CACHE[key] = url
    return url or ""


def lookup_many(items):
    """items: list of (product, brand, category). Lookups run in parallel and
    the page waits at most BUDGET_S; anything slower shows the illustration
    this time (and is usually cached by the next question)."""
    results = [""] * len(items)
    if not items:
        return results
    futures = {_POOL.submit(lookup_one, *it): i for i, it in enumerate(items)}
    done, _ = wait(futures, timeout=BUDGET_S)
    for f in done:
        try:
            results[futures[f]] = f.result() or ""
        except Exception:
            pass
    return results
