"""Tests for live QuickCommerce search + delivery ETA wired into RAGPipeline.answer."""

import pytest

from qc_client import QuickCommerceClient, QuickCommerceError
from rag import LLMResponse, RAGPipeline, live_request_kwargs, live_search_query


class StubRetriever:
    """Returns one catalogue chunk so the pipeline takes the answering path."""

    ids = ["prod-1"]
    metas = [
        {
            "source": "BigBasket Catalogue",
            "source_file": "BigBasket Products.csv",
            "product": "Amul Pasteurised Butter",
            "brand": "Amul",
            "category": "Bakery, Cakes & Dairy",
            "price": 325.0,
            "page": "row-1",
            "chunk": 0,
        }
    ]

    def retrieve(self, query, top_k=5, score_threshold=0.0, mode="hybrid", rerank=True):
        meta = dict(self.metas[0])
        return [
            {
                "id": self.ids[0],
                "content": "Name: Amul Pasteurised Butter\nBrand: Amul\nPrice: Rs 325",
                "text": "Name: Amul Pasteurised Butter\nBrand: Amul\nPrice: Rs 325",
                "metadata": meta,
                "similarity_score": 0.81,
                "score": 0.81,
                "similarity": 0.81,
            }
        ]


class StubLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        return LLMResponse(content="Amul Pasteurised Butter costs Rs 325 [1].")


class StubLiveClient:
    def __init__(self, eta="14 mins", fail=None):
        self.search_calls = []
        self.eta_calls = []
        self._eta = eta
        self._fail = fail

    def search(self, query, platform, lat, lon, pincode=None):
        self.search_calls.append(
            {"query": query, "platform": platform, "lat": lat, "lon": lon, "pincode": pincode}
        )
        if self._fail:
            raise self._fail
        return [
            {
                "id": "104864",
                "name": "Amul Pasteurised Butter",
                "brand": "Amul",
                "available": True,
                "price": 325.0,
                "mrp": 325.0,
                "currency": "INR",
                "quantity": "500 g",
                "deeplink": "https://www.bigbasket.com/pd/104864",
                "platform": "BigBasket",
                "sla": "5 mins",
            }
        ]

    def eta(self, platform, lat, lon, pincode=None):
        self.eta_calls.append({"platform": platform, "lat": lat, "lon": lon})
        if self._fail:
            raise self._fail
        return {
            "platform": platform,
            "eta": self._eta,
            "eta_minutes": 14,
            "open": True,
            "store_id": "45342",
            "store_ids": ["45342"],
            "url": "https://blinkit.com",
        }


def make_pipeline(live_client=None):
    return RAGPipeline(retriever=StubRetriever(), llm=StubLLM(), live_client=live_client)


def test_answer_puts_live_eta_and_prices_in_the_prompt():
    live = StubLiveClient(eta="14 mins")
    pipeline = make_pipeline(live)

    result = pipeline.answer(
        "What is the price of Amul butter?",
        live_search=True,
        live_eta=True,
        platform="BlinkIt",
        lat=12.9021,
        lon=77.6639,
    )

    assert result["live"]["eta"]["eta_minutes"] == 14
    assert result["live"]["products"][0]["price"] == 325.0
    prompt = pipeline.llm.prompts[-1]
    assert "LIVE" in prompt
    assert "14 mins" in prompt
    assert "104864" in prompt or "Amul Pasteurised Butter" in prompt


def test_live_search_uses_the_cleaned_question_and_location():
    live = StubLiveClient()
    pipeline = make_pipeline(live)

    pipeline.answer(
        "What is the price of Amul butter?",
        live_search=True,
        platform="BigBasket",
        lat=12.9021,
        lon=77.6639,
        pincode="560102",
    )

    assert live.search_calls[0]["query"] == "amul butter"
    assert live.search_calls[0]["platform"] == "BigBasket"
    assert live.search_calls[0]["lat"] == 12.9021
    assert live.search_calls[0]["pincode"] == "560102"


def test_answer_still_returns_when_live_api_fails():
    live = StubLiveClient(fail=QuickCommerceError("No credits remaining", status_code=402))
    pipeline = make_pipeline(live)

    result = pipeline.answer(
        "What is the price of Amul butter?",
        live_search=True,
        live_eta=True,
        platform="BigBasket",
        lat=12.9021,
        lon=77.6639,
    )

    assert "325" in result["answer"]
    assert result["live"] is None
    assert "402" in result["live_error"]


def test_live_lookup_is_skipped_when_no_location_is_given():
    live = StubLiveClient()
    pipeline = make_pipeline(live)

    result = pipeline.answer(
        "What is the price of Amul butter?",
        live_search=True,
        platform="BigBasket",
    )

    assert live.search_calls == []
    assert result["live"] is None
    assert result["live_error"] == "location_required"


def test_pipeline_creates_live_client_automatically_from_the_env_key(monkeypatch):
    monkeypatch.setenv("QuickCommerceAPI", "env-key")

    pipeline = RAGPipeline(retriever=StubRetriever(), llm=StubLLM())

    assert isinstance(pipeline.live_client, QuickCommerceClient)
    assert pipeline.live_client.api_key == "env-key"


def test_explicitly_passed_live_client_beats_the_env_key(monkeypatch):
    monkeypatch.setenv("QuickCommerceAPI", "env-key")
    live = StubLiveClient()

    pipeline = make_pipeline(live)

    assert pipeline.live_client is live


def test_pipeline_falls_back_cleanly_when_the_key_is_absent(monkeypatch):
    monkeypatch.delenv("QuickCommerceAPI", raising=False)
    pipeline = RAGPipeline(retriever=StubRetriever(), llm=StubLLM())

    assert pipeline.live_client is None

    result = pipeline.answer(
        "What is the price of Amul butter?",
        live_search=True,
        live_eta=True,
        platform="BigBasket",
        lat=12.9021,
        lon=77.6639,
    )

    assert "325" in result["answer"]
    assert result["live"] is None
    assert result["live_error"] == "live_client_not_configured"


def test_live_search_query_strips_question_scaffolding():
    assert live_search_query("What is the price of Amul butter?") == "amul butter"
    assert live_search_query("How much does Amul butter cost?") == "amul butter"


def test_live_search_query_drops_stopwords_and_keeps_brand_terms():
    assert live_search_query("Show me cheaper alternatives to Saffola Gold oil") == (
        "cheaper alternatives saffola gold oil"
    )


def test_live_search_query_drops_possessives():
    assert (
        live_search_query("What is BigBasket's return policy for spoiled food items?")
        == "bigbasket return policy spoiled food items"
    )


def test_live_search_query_drops_leading_pronouns():
    assert (
        live_search_query("It's a 500 g pack of Amul butter, how much is it?")
        == "500 g pack amul butter"
    )


def test_live_search_query_cuts_a_trailing_question_clause():
    assert (
        live_search_query("Which face wash suits oily skin, and what does it cost?")
        == "face wash suits oily skin"
    )


def test_live_search_query_is_capped_to_a_few_words():
    query = live_search_query(
        "Can you please suggest some healthy high protein breakfast cereals with low sugar for diabetics"
    )

    assert len(query.split()) <= 6
    assert "cereals" in query or "breakfast" in query


def test_live_search_query_never_returns_empty():
    assert live_search_query("?") != ""
    assert live_search_query("What is it?") != ""


def test_live_request_kwargs_are_all_off_when_the_toggle_is_off():
    kwargs = live_request_kwargs(
        enabled=False, platform="BlinkIt", lat=12.9, lon=77.6, pincode="560102"
    )

    assert kwargs == {
        "live_search": False,
        "live_eta": False,
        "platform": "BlinkIt",
        "lat": 12.9,
        "lon": 77.6,
        "pincode": "560102",
    }


def test_live_request_kwargs_enable_both_search_and_eta():
    kwargs = live_request_kwargs(
        enabled=True, platform="BigBasket", lat=12.9021, lon=77.6639, pincode=None
    )

    assert kwargs["live_search"] is True
    assert kwargs["live_eta"] is True
    assert kwargs["platform"] == "BigBasket"


def test_live_request_is_not_served_from_a_cached_catalogue_only_answer():
    live = StubLiveClient()
    pipeline = make_pipeline(live)

    pipeline.answer("What is the price of Amul butter?")
    result = pipeline.answer(
        "What is the price of Amul butter?",
        live_search=True,
        platform="BigBasket",
        lat=12.9021,
        lon=77.6639,
    )

    assert len(live.search_calls) == 1
    assert result["live"] is not None
    assert result["cached"] is False


def test_live_lookup_is_skipped_when_not_requested():
    live = StubLiveClient()
    pipeline = make_pipeline(live)

    result = pipeline.answer("What is the price of Amul butter?", lat=12.9, lon=77.6)

    assert live.search_calls == []
    assert live.eta_calls == []
    assert result["live"] is None
    assert result["live_error"] is None
