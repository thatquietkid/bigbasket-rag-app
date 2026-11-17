"""Tests for qc_client.QuickCommerceClient (written before the client existed)."""

import json

import pytest

from qc_client import QuickCommerceClient, QuickCommerceError


def search_body(products):
    return {
        "status": "success",
        "request_id": "req-1",
        "credits_remaining": 99,
        "data": {
            "query": "amul butter",
            "platform": "BigBasket",
            "total_results": len(products),
            "products": products,
        },
    }


def eta_body(eta="14 mins", store_id="45342"):
    return {
        "status": "success",
        "request_id": "req-2",
        "credits_remaining": 98,
        "data": {
            "platform": "BlinkIt",
            "eta": eta,
            "open": True,
            "store_id": store_id,
            "store_ids": [store_id, "36475"],
            "url": "https://blinkit.com",
        },
    }


def make_client(responses, seen=None):
    """Build a client whose transport replays canned (status, body) responses."""
    queue = list(responses)

    def http_get(url, headers, timeout):
        if seen is not None:
            seen.append({"url": url, "headers": dict(headers)})
        status, body = queue.pop(0)
        return status, json.dumps(body)

    return QuickCommerceClient(api_key="test-key", http_get=http_get)


def test_search_normalizes_string_prices_to_floats():
    # The live API returns mrp/offer_price as strings like "325".
    client = make_client(
        [
            (
                200,
                search_body(
                    [
                        {
                            "id": "104864",
                            "name": "Pasteurised Butter",
                            "brand": "Amul",
                            "available": True,
                            "mrp": "325",
                            "offer_price": "325",
                            "quantity": "500 g",
                            "rating": None,
                            "inventory": 1,
                            "deeplink": "https://www.bigbasket.com/pd/104864",
                            "platform": {"name": "BigBasket", "sla": "5 mins", "open": True},
                        }
                    ]
                ),
            )
        ]
    )

    products = client.search("amul butter", platform="BigBasket", lat=12.9021, lon=77.6639)

    assert len(products) == 1
    assert products[0]["price"] == 325.0
    assert products[0]["mrp"] == 325.0
    assert products[0]["currency"] == "INR"
    assert products[0]["sla"] == "5 mins"
    assert products[0]["platform"] == "BigBasket"
    assert products[0]["available"] is True


def test_search_sends_api_key_header_and_query_params():
    seen = []
    client = make_client([(200, search_body([]))], seen=seen)

    client.search("amul butter", platform="BigBasket", lat=12.9021, lon=77.6639, pincode="560102")

    assert seen[0]["headers"]["X-API-Key"] == "test-key"
    url = seen[0]["url"]
    assert url.startswith("https://api.quickcommerceapi.com/v1/search?")
    assert "q=amul+butter" in url
    assert "platform=BigBasket" in url
    assert "lat=12.9021" in url
    assert "lon=77.6639" in url
    assert "pincode=560102" in url


def test_repeated_identical_search_costs_only_one_credit():
    seen = []
    client = make_client([(200, search_body([]))], seen=seen)

    client.search("milk", platform="BigBasket", lat=12.9, lon=77.6)
    client.search("milk", platform="BigBasket", lat=12.9, lon=77.6)

    assert len(seen) == 1


def test_eta_parses_minutes_and_store():
    client = make_client([(200, eta_body("14 mins"))])

    eta = client.eta(platform="BlinkIt", lat=12.9021, lon=77.6639)

    assert eta["eta"] == "14 mins"
    assert eta["eta_minutes"] == 14
    assert eta["open"] is True
    assert eta["store_id"] == "45342"
    assert eta["platform"] == "BlinkIt"


def test_eta_keeps_null_when_eta_is_not_parseable():
    client = make_client([(200, eta_body("Temporarily unavailable"))])

    eta = client.eta(platform="BlinkIt", lat=12.9, lon=77.6)

    assert eta["eta_minutes"] is None
    assert eta["eta"] == "Temporarily unavailable"


def test_no_credits_raises_error_with_status_code():
    client = make_client([(402, {"detail": "No credits remaining"})])

    with pytest.raises(QuickCommerceError) as excinfo:
        client.search("milk", platform="BigBasket", lat=12.9, lon=77.6)

    assert excinfo.value.status_code == 402
    assert "credits" in str(excinfo.value).lower()


def test_api_key_is_read_from_the_QuickCommerceAPI_env_var(monkeypatch):
    monkeypatch.setenv("QuickCommerceAPI", "env-key")
    assert QuickCommerceClient().api_key == "env-key"


def test_missing_api_key_raises_value_error_naming_the_env_var(monkeypatch):
    monkeypatch.delenv("QuickCommerceAPI", raising=False)

    with pytest.raises(ValueError, match="QuickCommerceAPI"):
        QuickCommerceClient()


def test_credits_are_exposed_for_monitoring():
    client = make_client(
        [
            (
                200,
                {
                    "summary": {"total_available": 99, "total_used": 1, "active_packs": 1},
                    "active": [],
                    "inactive": [],
                },
            )
        ]
    )

    credits = client.credits()

    assert credits["total_available"] == 99
