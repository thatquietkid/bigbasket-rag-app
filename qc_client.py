"""
qc_client.py — Thin client for the QuickCommerce API (api.quickcommerceapi.com).

Live product search and delivery ETA for Indian/UAE quick commerce platforms.
Every call costs 1 credit per platform, so successful responses are cached in
memory for CACHE_TTL seconds to keep repeats free.

Usage:
    client = QuickCommerceClient()                  # reads QuickCommerceAPI env var
    products = client.search("amul butter", "BigBasket", 12.9021, 77.6639)
    eta = client.eta("BigBasket", 12.9021, 77.6639)
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

BASE_URL = "https://api.quickcommerceapi.com"
API_KEY_ENV = "QuickCommerceAPI"
CACHE_TTL = 300.0

# UAE platforms are priced in AED; everything else is INR.
AED_PLATFORMS = {"NoonMinutes", "TalabatMart"}

Transport = Callable[[str, Dict[str, str], float], Tuple[int, str]]


class QuickCommerceError(Exception):
    """Raised when the QuickCommerce API rejects a request."""

    def __init__(self, message: str, status_code: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _urllib_get(url: str, headers: Dict[str, str], timeout: float) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def _as_float(value: Any) -> Optional[float]:
    """The API returns prices as strings ("325"); normalise them to floats."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _eta_minutes(eta: Any) -> Optional[int]:
    if not isinstance(eta, str):
        return None
    match = re.search(r"(\d+)", eta)
    return int(match.group(1)) if match else None


class QuickCommerceClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        timeout: float = 25.0,
        http_get: Optional[Transport] = None,
        cache_ttl: float = CACHE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.api_key = api_key or os.getenv(API_KEY_ENV, "")
        if not self.api_key:
            raise ValueError(
                f"No QuickCommerce API key found. Set the {API_KEY_ENV} environment variable."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_get = http_get or _urllib_get
        self._cache_ttl = cache_ttl
        self._clock = clock
        self._cache: Dict[Any, Tuple[float, Dict[str, Any]]] = {}
        self.last_credits_remaining: Optional[int] = None

    # -- plumbing ----------------------------------------------------------
    def _get(self, path: str, params: Dict[str, Any], unwrap: bool = True) -> Dict[str, Any]:
        clean = {k: v for k, v in params.items() if v is not None}
        cache_key = (path, tuple(sorted((k, str(v)) for k, v in clean.items())))

        hit = self._cache.get(cache_key)
        if hit is not None and hit[0] > self._clock():
            return hit[1]

        url = f"{self.base_url}{path}?{urllib.parse.urlencode(clean)}"
        headers = {"X-API-Key": self.api_key, "Accept": "application/json"}
        status, text = self._http_get(url, headers, self.timeout)

        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {}

        if status >= 400:
            message = (
                payload.get("detail")
                or payload.get("error")
                or payload.get("message")
                or text[:300]
                or f"HTTP {status}"
            )
            raise QuickCommerceError(str(message), status_code=status, code=payload.get("code"))

        remaining = payload.get("credits_remaining")
        if isinstance(remaining, int):
            self.last_credits_remaining = remaining

        data = payload.get("data") if unwrap else payload
        if data is None:
            raise QuickCommerceError(
                f"Unexpected QuickCommerce response for {path}", status_code=status
            )

        if self._cache_ttl > 0:
            self._cache[cache_key] = (self._clock() + self._cache_ttl, data)
        return data

    # -- endpoints ---------------------------------------------------------
    def search(
        self,
        query: str,
        platform: str,
        lat: float,
        lon: float,
        pincode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Live product search on one platform. Costs 1 credit (cached afterwards)."""
        data = self._get(
            "/v1/search",
            {"q": query, "platform": platform, "lat": lat, "lon": lon, "pincode": pincode},
        )
        currency = "AED" if platform in AED_PLATFORMS else "INR"
        products = []
        for p in data.get("products") or []:
            meta = p.get("platform") or {}
            mrp = _as_float(p.get("mrp"))
            price = _as_float(p.get("offer_price"))
            products.append(
                {
                    "id": str(p.get("id")) if p.get("id") is not None else None,
                    "name": p.get("name"),
                    "brand": p.get("brand"),
                    "available": bool(p.get("available")),
                    "price": price if price is not None else mrp,
                    "mrp": mrp,
                    "currency": p.get("currency") or currency,
                    "quantity": p.get("quantity"),
                    "inventory": p.get("inventory"),
                    "deeplink": p.get("deeplink"),
                    "platform": meta.get("name") or platform,
                    "sla": meta.get("sla"),
                    "open": meta.get("open"),
                }
            )
        return products

    def eta(
        self,
        platform: str,
        lat: float,
        lon: float,
        pincode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delivery ETA + store availability. Costs 1 credit (cached afterwards)."""
        data = self._get("/v1/eta", {"platform": platform, "lat": lat, "lon": lon, "pincode": pincode})
        return {
            "platform": data.get("platform") or platform,
            "eta": data.get("eta"),
            "eta_minutes": _eta_minutes(data.get("eta")),
            "open": data.get("open"),
            "store_id": data.get("store_id"),
            "store_ids": data.get("store_ids") or [],
            "url": data.get("url"),
        }

    def credits(self) -> Dict[str, Any]:
        """Free endpoint: remaining credits and pack breakdown."""
        payload = self._get("/v1/credits", {}, unwrap=False)
        return payload.get("summary", {})
