"""
rag.py — RAG Pipeline for the BigBasket Product Assistant.

Implements your rag_enhanced pattern with:
  - retriever.retrieve(query, top_k, score_threshold)
  - llm.invoke(prompt) returning response.content
  - Standardized dictionary output: answer, sources, confidence_score, context
  - Groq primary (chatgpt-oss-20b / llama-3.1-8b) with Gemini fallback wrapper
  - Hybrid retrieval (dense vector + BM25 keyword merged via RRF)
  - Cross-Encoder re-ranking (ms-marco-MiniLM-L-6-v2)
  - Exact Hash Cache (SHA-256)
  - Input/output safety & relevance guardrails
  - Query decomposition for compound questions
"""

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from qc_client import QuickCommerceClient, QuickCommerceError

load_dotenv()

# Offline mode stops Hugging Face from checking the internet, which is handy
# on a laptop where the models are already downloaded. On a fresh server
# (Streamlit Cloud) the models must be downloaded once, so offline mode is
# only switched on when BB_OFFLINE=1 is set.
if os.getenv("BB_OFFLINE") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv("BB_DATA_PATH", "data/BigBasket Products.csv")
POLICY_PATH = os.getenv("BB_POLICY_PATH", "data/bigbasket_policies.txt")
CHROMA_DIR = "chroma_db"
COLLECTION = "bigbasket_products"
LOG_DIR = os.getenv("BB_LOG_DIR", "logs")

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_ROWS = 5000
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K = 5
MIN_SCORE = 0.20
RRF_K = 60
SIM_THRESHOLD = 0.30

GROQ_MODELS = [
    os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

GEMINI_MODELS = [
    m for m in [os.getenv("GEMINI_MODEL"), "gemini-2.0-flash", "gemini-2.5-flash", "gemini-flash-latest"] if m
]

FALLBACK_MSG = (
    "I couldn't find enough information about that in the BigBasket product "
    "data I have, so I don't want to guess. Try asking about a specific "
    "product, brand, category, price range or BigBasket policy."
)

GUARDRAIL_MSG = (
    "Your query could not be processed as it requests out-of-scope actions, "
    "unsupported topics, or potential prompt injection. Please ask questions "
    "related to BigBasket products, prices, ingredients, comparisons, or policies."
)


def format_live_block(live: Dict[str, Any]) -> str:
    """Render live QuickCommerce data as a context block for the answer prompt."""
    lines = [
        "LIVE DATA (fetched just now from the platform's own app — "
        "prefer these prices, stock and ETAs over the catalogue snapshot):"
    ]

    eta = live.get("eta")
    if eta:
        state = "open" if eta.get("open") else "closed"
        lines.append(
            f"Delivery ETA — {eta.get('platform')}: {eta.get('eta')} "
            f"(store {eta.get('store_id')}, {state})"
        )

    products = live.get("products") or []
    if products:
        lines.append(f"Live results for '{live.get('query')}' on {live.get('platform')}:")
        for n, p in enumerate(products[:5], 1):
            lines.append(
                f"{n}. id={p.get('id')} | {p.get('brand') or ''} {p.get('name')} | "
                f"{p.get('quantity')} | {p.get('currency')} {p.get('price')} "
                f"(MRP {p.get('mrp')}) | in stock: {p.get('available')} | "
                f"ETA {p.get('sla')} | {p.get('deeplink')}"
            )
    return "\n".join(lines)


def default_live_client() -> Optional[QuickCommerceClient]:
    """Build a live client from the QuickCommerceAPI env var, or None if it is absent."""
    try:
        return QuickCommerceClient()
    except ValueError:
        return None


# Question scaffolding stripped from live search queries; longest first.
LIVE_QUERY_PREFIXES = (
    "what is the price of",
    "what's the price of",
    "what is the cost of",
    "how much does it cost to buy",
    "how much does",
    "how much do",
    "how much is",
    "what are the",
    "what is the",
    "what is",
    "what's",
    "which is the",
    "which",
    "can you",
    "could you",
    "would you",
    "tell me",
    "show me",
    "give me",
    "find me",
    "do you have",
    "is there",
    "are there",
    "i want",
    "i need",
    "please",
    "suggest",
    "list",
    "compare",
)

LIVE_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "for", "on", "in", "to", "at",
    "from", "with", "me", "my", "do", "does", "and", "or", "cost", "costs", "there",
    "it", "its", "this", "that", "these", "those",
}


def live_search_query(question: str, max_words: int = 6) -> str:
    """Turn a conversational question into a short, search-ready API query.

    The QuickCommerce API matches keywords, so "What is the price of Amul butter?"
    becomes "amul butter" instead of returning noisy results for the whole sentence.
    """
    text = (question or "").strip().lower()
    text = re.sub(r"['\u2019]s\b", "", text)  # possessive: BigBasket's -> bigbasket
    text = re.sub(r"[^\w\s\u20b9%.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().strip(".")

    changed = True
    while changed:
        changed = False
        for prefix in LIVE_QUERY_PREFIXES:
            if text == prefix:
                text = ""
                changed = True
                break
            if text.startswith(prefix + " "):
                text = text[len(prefix) + 1 :].strip()
                changed = True
                break

    # "face wash suits oily skin, and what does it cost" -> "face wash suits oily skin"
    text = re.split(r"\s+(?:and\s+)?(?:what|how|which|who|why|when)\b", text)[0].strip()

    words = [w for w in text.split() if w.strip(".") and w not in LIVE_QUERY_STOPWORDS]
    if not words:
        words = [w for w in re.findall(r"[\w\u20b9%.-]+", (question or "").lower()) if w not in LIVE_QUERY_STOPWORDS]
    if not words:
        words = [w for w in re.findall(r"[\w\u20b9%.-]+", (question or "").lower())]
    return " ".join(words[:max_words]) or (question or "").strip()


def live_request_kwargs(
    enabled: bool,
    platform: str,
    lat: Optional[float],
    lon: Optional[float],
    pincode: Optional[str] = None,
) -> Dict[str, Any]:
    """Map UI state onto RAGPipeline.answer() live parameters."""
    return {
        "live_search": bool(enabled),
        "live_eta": bool(enabled),
        "platform": platform,
        "lat": lat,
        "lon": lon,
        "pincode": pincode,
    }

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# AUDIT LOGGING HELPER
# ---------------------------------------------------------------------------
def log_audit_trail(audit_entry: Dict[str, Any], log_dir: str = LOG_DIR) -> str:
    """Log structured query telemetry and chunks to logs/audit_YYYYMMDD.jsonl."""
    try:
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        filepath = os.path.join(log_dir, f"audit_{today}.jsonl")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        return filepath
    except Exception as e:
        print(f"[Audit Log Error] Failed to write log: {e}")
        return ""
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) (instructions|directions|prompts)",
    r"system prompt",
    r"you are now",
    r"forget (your|all) (instructions|rules)",
    r"disregard",
    r"bypass",
    r"dan mode",
]

OFF_TOPIC_PATTERNS = [
    r"\b(write|generate)\s+(code|python|script|sql|javascript|java|c\+\+)\b",
    r"\b(diagnose|prescribe|medical advice|treat disease)\b",
    r"\b(legal advice|lawsuit|sue)\b",
    r"\b(write an essay|write a poem|write a song)\b",
]

PROFANITY_PATTERNS = [
    r"\b(fuck|shit|bitch|bastard|asshole)\b",
]


from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# AUDIT LOGGING HELPER
# ---------------------------------------------------------------------------
def log_audit_trail(audit_entry: Dict[str, Any], log_dir: str = LOG_DIR) -> str:
    """Log structured query telemetry and chunks to logs/audit_YYYYMMDD.jsonl."""
    try:
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        filepath = os.path.join(log_dir, f"audit_{today}.jsonl")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        return filepath
    except Exception as e:
        print(f"[Audit Log Error] Failed to write log: {e}")
        return ""
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) (instructions|directions|prompts)",
    r"system prompt",
    r"you are now",
    r"forget (your|all) (instructions|rules)",
    r"disregard",
    r"bypass",
    r"dan mode",
]

OFF_TOPIC_PATTERNS = [
    r"\b(write|generate)\s+(code|python|script|sql|javascript|java|c\+\+)\b",
    r"\b(diagnose|prescribe|medical advice|treat disease)\b",
    r"\b(legal advice|lawsuit|sue)\b",
    r"\b(write an essay|write a poem|write a song)\b",
]

PROFANITY_PATTERNS = [
    r"\b(fuck|shit|bitch|bastard|asshole)\b",
]


def check_input_guardrail(query: str) -> Tuple[bool, Optional[str]]:
    q_lower = query.lower().strip()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "Prompt injection pattern detected."
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "Off-topic query detected."
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "Inappropriate language detected."
    return True, None


def contextualize_query(query: str, chat_history: List[Dict[str, str]], llm: Any) -> str:
    """Rewrite a follow-up query to include conversational context if needed."""
    if not chat_history:
        return query

    lower_q = query.lower()
    has_followup_clues = any(w in lower_q for w in [
        "cheaper", "alternative", "another", "instead", "it", "them", "these", "those",
        "better", "more", "less", "one of those", "under", "show me", "how about"
    ]) or len(query.split()) <= 5

    if not has_followup_clues:
        return query

    recent_history = []
    for msg in chat_history[-4:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")[:200]
        recent_history.append(f"{role}: {content}")

    history_str = "\n".join(recent_history)

    prompt = f"""Given the following conversation history and a follow-up user question, rephrase the follow-up question into a standalone, complete search query.
Do NOT answer the question, only output the rephrased query.

Chat History:
{history_str}

Follow-up Question: {query}
Standalone Question:"""

    try:
        rephrased = llm.invoke(prompt).content.strip()
        rephrased = re.sub(r'^["\']|["\']$', '', rephrased).strip()
        if rephrased and len(rephrased) > len(query):
            return rephrased
    except Exception:
        pass

    return query
    q_lower = query.lower().strip()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "Prompt injection pattern detected."
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "Off-topic query detected."
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "Inappropriate language detected."
    return True, None


# ---------------------------------------------------------------------------
# FOLLOW-UP ANCHORING
# ---------------------------------------------------------------------------
FOLLOW_UP_RE = re.compile(
    r"\b(cheaper|cheapest|alternative|alternatives|another|instead|these|those|them|it|its|same|"
    r"ones|the first|the second|that one|this one)\b",
    re.IGNORECASE,
)


def is_follow_up(query: str) -> bool:
    """True when the question only makes sense with the previous turn."""
    q = (query or "").strip().lower()
    if FOLLOW_UP_RE.search(q):
        return True
    return bool(
        re.match(r"^(and|also|what about|how about|any|do they|does it|is it|are they|which one)\b", q)
    )


def follow_up_anchor(chat_history: List[Dict[str, Any]], max_products: int = 2) -> str:
    """Previous topic + products, so a follow-up retrieves the same catalogue context."""
    if not chat_history:
        return ""

    parts: List[str] = []
    last_user = next((m for m in reversed(chat_history) if m.get("role") == "user"), None)
    if last_user:
        topic = live_search_query(last_user.get("content", ""), max_words=6)
        if topic:
            parts.append(topic)

    last_assistant = next((m for m in reversed(chat_history) if m.get("role") == "assistant"), None)
    products: List[str] = []
    if last_assistant:
        for chunk in last_assistant.get("chunks") or []:
            name = (chunk.get("metadata") or {}).get("product")
            if name and name not in products:
                products.append(name)
    parts.extend(products[:max_products])

    return " ".join(parts).strip()


def previous_turn_price(chat_history: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    """Price of the first product discussed in the previous answer, if any."""
    if not chat_history:
        return None
    last_assistant = next((m for m in reversed(chat_history) if m.get("role") == "assistant"), None)
    if not last_assistant:
        return None
    for chunk in last_assistant.get("chunks") or []:
        try:
            price = float((chunk.get("metadata") or {}).get("price"))
        except (TypeError, ValueError):
            continue
        if price > 0:
            return price
    return None


def wants_cheaper(query: str) -> bool:
    return bool(
        re.search(
            r"\b(cheaper|cheapest|lower price|less expensive|affordable|budget)\b",
            query or "",
            re.IGNORECASE,
        )
    )


# ---------------------------------------------------------------------------
# DATA LOADING & CHUNKING
# ---------------------------------------------------------------------------
def load_and_clean(path: str = DATA_PATH, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop(columns=[c for c in ["index", "unnamed:_0"] if c in df.columns])
    df = df.dropna(subset=["product", "description"])

    for col in ["product", "brand", "category", "sub_category", "type", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    df = df[df["description"].str.len() >= 30]
    df = df.drop_duplicates(subset=["product", "brand"])
    df = df.drop_duplicates(subset=["description"])
    df["brand"] = df["brand"].replace("", "Unknown")

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    return df.reset_index(drop=True)


def _fmt_price(x: Any) -> str:
    try:
        return f"₹{float(x):,.0f}"
    except (TypeError, ValueError):
        return "not listed"


def row_to_header(row: pd.Series) -> str:
    discount = ""
    try:
        sp, mp = float(row["sale_price"]), float(row["market_price"])
        if mp > sp > 0:
            discount = f", {round((mp - sp) / mp * 100)}% off"
    except (TypeError, ValueError, KeyError):
        pass

    rating = row.get("rating")
    rating_txt = f"{rating}/5" if pd.notna(rating) else "no rating"

    return (
        f"Product: {row['product']}\n"
        f"Brand: {row['brand']}\n"
        f"Category: {row.get('category', '')} > {row.get('sub_category', '')}\n"
        f"Type: {row.get('type', '')}\n"
        f"Price: {_fmt_price(row.get('sale_price'))} "
        f"(MRP {_fmt_price(row.get('market_price'))}{discount})\n"
        f"Rating: {rating_txt}\n"
    )


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start, end)
        if nxt != -1:
            start = nxt + 1
    return chunks


def build_documents(df: pd.DataFrame, policy_path: str = POLICY_PATH) -> List[Dict[str, Any]]:
    docs = []
    for i, row in df.iterrows():
        header = row_to_header(row)
        pieces = split_text(row["description"])
        for j, piece in enumerate(pieces):
            content = header + f"Description: {piece}"
            docs.append({
                "id": f"prod-{i}-{j}",
                "content": content,
                "text": content,
                "metadata": {
                    "source": "BigBasket Catalogue",
                    "source_file": "BigBasket Products.csv",
                    "product": row["product"],
                    "brand": row["brand"],
                    "category": str(row.get("category", "")),
                    "price": float(row["sale_price"]) if pd.notna(row.get("sale_price")) else -1.0,
                    "page": f"row-{i}",
                    "chunk": j,
                },
            })

    if policy_path and os.path.exists(policy_path):
        with open(policy_path, encoding="utf-8") as f:
            paras = [p.strip() for p in f.read().split("\n\n") if p.strip()]
        for k, para in enumerate(paras):
            for j, piece in enumerate(split_text(para)):
                content = f"BigBasket policy / FAQ:\n{piece}"
                docs.append({
                    "id": f"policy-{k}-{j}",
                    "content": content,
                    "text": content,
                    "metadata": {
                        "source": "BigBasket Policy Document",
                        "source_file": "bigbasket_policies.txt",
                        "product": "BigBasket policy",
                        "brand": "BigBasket",
                        "category": "Policy",
                        "price": -1.0,
                        "page": f"section-{k}",
                        "chunk": j,
                    },
                })
    return docs


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# LLM WRAPPER: Compatible with llm.invoke(prompt) -> response.content
# ---------------------------------------------------------------------------
class LLMResponse:
    def __init__(self, content: str):
        self.content = content


class MultiProviderLLM:
    """Wrapper that supports llm.invoke(prompt) with Groq primary & Gemini fallback."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_model: str = GROQ_MODELS[0],
        temperature: float = 0.2,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.groq_model = groq_model
        self.temperature = temperature

    def invoke(self, prompt: Union[str, List[Any]]) -> LLMResponse:
        prompt_text = prompt[0] if isinstance(prompt, list) else str(prompt)

        # 1. Try Groq (LangChain ChatGroq or Native Groq)
        if self.groq_api_key:
            try:
                from langchain_groq import ChatGroq

                for model in [self.groq_model] + [m for m in GROQ_MODELS if m != self.groq_model]:
                    try:
                        llm = ChatGroq(
                            model=model,
                            groq_api_key=self.groq_api_key,
                            temperature=self.temperature,
                        )
                        res = llm.invoke(prompt_text)
                        return LLMResponse(content=res.content)
                    except Exception as e:
                        err_str = str(e)
                        if any(x in err_str for x in ["404", "model_not_found", "rate_limit", "429"]):
                            continue
                        raise
            except Exception:
                if not self.gemini_api_key:
                    raise

        # 2. Try Gemini Fallback
        if self.gemini_api_key:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)
            last_err = None
            for model in GEMINI_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt_text,
                        config=types.GenerateContentConfig(temperature=self.temperature),
                    )
                    return LLMResponse(content=resp.text)
                except Exception as e:
                    if "NOT_FOUND" in str(e) or "404" in str(e):
                        last_err = e
                        continue
                    raise
            raise RuntimeError(f"None of the Gemini models {GEMINI_MODELS} are available: {last_err}")

        raise ValueError("No valid API key provided for Groq or Gemini.")


# ---------------------------------------------------------------------------
# RETRIEVER: Compatible with retriever.retrieve(query, top_k, score_threshold)
# ---------------------------------------------------------------------------
class BigBasketRetriever:
    def __init__(self, data_path: str = DATA_PATH):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedder = SentenceTransformer(EMBED_MODEL)
        try:
            self.reranker = CrossEncoder(RERANK_MODEL)
        except Exception:
            self.reranker = None

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0:
            docs = build_documents(load_and_clean(data_path))
            self._index(docs)

        stored = self.collection.get(include=["documents", "metadatas"])
        self.ids = stored["ids"]
        self.texts = stored["documents"]
        self.metas = stored["metadatas"]
        self.id_to_idx = {d: i for i, d in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    def _index(self, docs: List[Dict[str, Any]], batch: int = 256):
        texts = [d["content"] for d in docs]
        embs = self.embedder.encode(
            texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True
        )
        for s in range(0, len(docs), 2000):
            part = docs[s : s + 2000]
            self.collection.add(
                ids=[d["id"] for d in part],
                documents=[d["content"] for d in part],
                metadatas=[d["metadata"] for d in part],
                embeddings=embs[s : s + 2000].tolist(),
            )

    def _vector_search(self, query: str, n: int) -> Tuple[List[str], List[float]]:
        q = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q, n_results=n)
        ids = res["ids"][0]
        sims = [1 - d for d in res["distances"][0]]
        return ids, sims

    def _keyword_search(self, query: str, n: int) -> List[str]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.ids[i] for i in top if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        score_threshold: float = MIN_SCORE,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        pool = top_k * 4
        v_ids, v_sims = self._vector_search(query, pool)
        sim_of = dict(zip(v_ids, v_sims))

        if mode == "vector":
            ranked = v_ids
        elif mode == "keyword":
            ranked = self._keyword_search(query, pool)
        else:  # hybrid
            k_ids = self._keyword_search(query, pool)
            fused = {}
            for rank, d in enumerate(v_ids):
                fused[d] = fused.get(d, 0) + 1 / (RRF_K + rank + 1)
            for rank, d in enumerate(k_ids):
                fused[d] = fused.get(d, 0) + 1 / (RRF_K + rank + 1)
            ranked = sorted(fused, key=fused.get, reverse=True)

        candidates = ranked[:pool]

        # Cross-Encoder Re-ranking
        if rerank and self.reranker is not None and candidates:
            try:
                cand_pairs = [[query, self.texts[self.id_to_idx[d]]] for d in candidates[:12]]
                scores = self.reranker.predict(cand_pairs)
                sorted_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
                ranked = [candidates[i] for i in sorted_indices]
            except Exception:
                pass

        results = []
        for d in ranked:
            sim = sim_of.get(d, 0.0)
            if sim < score_threshold:
                continue
            i = self.id_to_idx[d]
            results.append({
                "id": d,
                "content": self.texts[i],
                "text": self.texts[i],
                "metadata": self.metas[i],
                "similarity_score": round(float(sim), 4),
                "score": round(float(sim), 4),
                "similarity": round(float(sim), 4),
            })
            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------------------------
# EXACT PATTERN: rag_enhanced(query, retriever, llm, top_k, min_score, return_context)
# ---------------------------------------------------------------------------
def rag_enhanced(
    query: str,
    retriever: BigBasketRetriever,
    llm: MultiProviderLLM,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
    return_context: bool = False,
    decompose: bool = True,
    chat_history: Optional[List[Dict[str, str]]] = None,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    RAG Pipeline matching your pattern:
    - Returns answer, sources, confidence score, and optionally full context.
    - Contextualizes conversational follow-ups.
    - Augmented with safety guardrails and multi-part query decomposition.
    """
    # 0. Conversation Contextualization (resolves "Give me a cheaper alternative")
    search_query = query
    if chat_history:
        search_query = contextualize_query(query, chat_history, llm)
        # Follow-ups also inherit the previous topic and products verbatim, so the
        # retrieval query cannot drift when the rewrite drops the entity.
        if is_follow_up(query):
            anchor = follow_up_anchor(chat_history)
            if anchor:
                search_query = f"{search_query} {anchor}".strip()

    # 1. Guardrail Check
    is_safe, _ = check_input_guardrail(query)
    if not is_safe:
        return {
            "answer": GUARDRAIL_MSG,
            "sources": [],
            "confidence_score": 0.0,
            "context": "" if return_context else None,
            "guardrail": True,
            "fallback": True,
        }

    # 2. Query Decomposition
    sub_queries = [search_query]
    if decompose:
        lower_q = search_query.lower()
        if any(w in lower_q for w in [" and ", " also ", "compare", " vs ", "versus"]):
            try:
                dec_prompt = (
                    f"Break the user question into 1 to 3 distinct atomic sub-questions for product search. "
                    f"Return ONLY a JSON list of strings.\nQuestion: {search_query}\nJSON:"
                )
                dec_resp = llm.invoke(dec_prompt).content
                match = re.search(r"\[.*\]", dec_resp, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                        sub_queries = parsed[:3]
            except Exception:
                sub_queries = [search_query]

    # 3. Retrieve documents
    all_results = []
    seen_ids = set()
    for sq in sub_queries:
        for doc in retriever.retrieve(sq, top_k=top_k, score_threshold=min_score):
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_results.append(doc)

    results = all_results[:top_k]

    # 3b. "Cheaper" follow-ups only keep products below the price just discussed.
    if chat_history and wants_cheaper(query):
        reference = previous_turn_price(chat_history)
        if reference:
            cheaper = [
                d for d in all_results if 0 < (d["metadata"].get("price") or 0) < reference
            ]
            if cheaper:
                results = cheaper[:top_k]

    # 4. Fallback if no relevant documents found
    if not results:
        return {
            "answer": FALLBACK_MSG,
            "sources": [],
            "confidence_score": 0.0,
            "context": "" if return_context else None,
            "fallback": True,
            "guardrail": False,
        }

    # 5. Prepare context and sources (exact pattern)
    context = "\n\n".join(doc["content"] for doc in results)
    if extra_context:
        context = f"{context}\n\n{extra_context}"

    sources = [
        {
            "source": doc["metadata"].get(
                "source_file", doc["metadata"].get("source", "unknown")
            ),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["score"],
            "preview": doc["content"][:300] + "...",
            "product": doc["metadata"].get("product", ""),
            "brand": doc["metadata"].get("brand", ""),
            "price": doc["metadata"].get("price", -1.0),
        }
        for doc in results
    ]

    confidence = max(doc["score"] for doc in results)

    # 6. Generate answer using the LLM (exact pattern)
    history_block = ""
    if chat_history:
        lines = []
        for msg in chat_history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {(msg.get('content') or '')[:300]}")
        history_block = "\n\nRecent conversation:\n" + "\n".join(lines) + "\n"

    prompt = f"""Use the following context to answer the question concisely.
Cite product numbers like [1], [2] when mentioning items.
If the question refers to earlier items ("these", "those", "it", "cheaper"), answer about the
products named in the recent conversation and do not switch to different products.
{history_block}
Context:
{context}

Question: {query}
Answer:"""

    response = llm.invoke(prompt)

    output = {
        "answer": response.content,
        "sources": sources,
        "confidence_score": confidence,
        "context": context if return_context else None,
        "fallback": FALLBACK_MSG[:30] in response.content,
        "guardrail": False,
        "decomposed": len(sub_queries) > 1,
        "sub_queries": sub_queries,
        "chunks": results,  # compatibility helper for UI
        "best_sim": confidence,  # compatibility helper for UI
    }

    if return_context:
        output["context"] = context

    return output


# ---------------------------------------------------------------------------
# PIPELINE CLASS WRAPPER (for caching & Streamlit integration)
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(
        self,
        data_path: str = DATA_PATH,
        retriever: Optional[BigBasketRetriever] = None,
        llm: Optional[MultiProviderLLM] = None,
        live_client: Optional[QuickCommerceClient] = None,
    ):
        self.retriever = retriever if retriever is not None else BigBasketRetriever(data_path=data_path)
        self.llm = llm
        # Picked up automatically from QuickCommerceAPI; None keeps the app catalogue-only.
        self.live_client = live_client if live_client is not None else default_live_client()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def ids(self):
        return self.retriever.ids

    @property
    def metas(self):
        return self.retriever.metas

    def cache_stats(self) -> Dict[str, int]:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }

    def clear_cache(self):
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def answer(
        self,
        question: str,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        k: int = TOP_K,
        mode: str = "hybrid",
        rerank: bool = True,
        decompose: bool = True,
        chat_history: Optional[List[Dict[str, str]]] = None,
        live_search: bool = False,
        live_eta: bool = False,
        platform: str = "BigBasket",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        pincode: Optional[str] = None,
        live_client: Optional[QuickCommerceClient] = None,
    ) -> Dict[str, Any]:
        start_t = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        cache_k = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()
        # Live answers depend on location and platform, so they never use the
        # question-only cache (the client has its own short-lived HTTP cache).
        use_cache = not chat_history and not (live_search or live_eta)
        if use_cache and cache_k in self._cache:
            self._cache_hits += 1
            res = dict(self._cache[cache_k])
            res["cached"] = True
            res["latency_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
            
            # Log cache hit
            audit_entry = {
                "timestamp": timestamp,
                "query": question,
                "cached": True,
                "latency_ms": res["latency_ms"],
                "flow": "cache_hit",
                "sub_queries": [question],
                "confidence_score": res.get("confidence_score", 0.0),
                "chunks_sent_count": len(res.get("chunks", [])),
                "chunks": [
                    {
                        "id": c.get("id"),
                        "product": c.get("metadata", {}).get("product"),
                        "brand": c.get("metadata", {}).get("brand"),
                        "similarity_score": c.get("similarity_score"),
                    }
                    for c in res.get("chunks", [])
                ],
            }
            log_audit_trail(audit_entry)
            res["audit_entry"] = audit_entry
            return res

        self._cache_misses += 1

        llm = self.llm or MultiProviderLLM(groq_api_key=groq_api_key, gemini_api_key=gemini_api_key)

        # Live QuickCommerce lookup (search and/or delivery ETA). Billed per platform,
        # so failures degrade to catalogue-only answers instead of breaking the request.
        live: Optional[Dict[str, Any]] = None
        live_error: Optional[str] = None
        client = live_client or self.live_client
        if live_search or live_eta:
            if lat is None or lon is None:
                live_error = "location_required"
            elif client is None:
                live_error = "live_client_not_configured"
            else:
                try:
                    search_query = live_search_query(question)
                    live = {"query": search_query, "platform": platform}
                    if live_search:
                        live["products"] = client.search(
                            search_query, platform=platform, lat=lat, lon=lon, pincode=pincode
                        )
                    if live_eta:
                        live["eta"] = client.eta(
                            platform=platform, lat=lat, lon=lon, pincode=pincode
                        )
                except QuickCommerceError as e:
                    live = None
                    live_error = f"live_api_error: {e.status_code} {e}"
                except Exception as e:
                    live = None
                    live_error = f"live_api_error: {e}"

        res = rag_enhanced(
            query=question,
            retriever=self.retriever,
            llm=llm,
            top_k=k,
            min_score=MIN_SCORE,
            return_context=True,
            decompose=decompose,
            chat_history=chat_history,
            extra_context=format_live_block(live) if live else "",
        )

        res["live"] = live
        res["live_error"] = live_error
        res["cached"] = False
        res["latency_ms"] = round((time.perf_counter() - start_t) * 1000, 2)

        # Determine flow path for auditing
        if res.get("guardrail"):
            flow_name = "guardrail_blocked"
        elif res.get("fallback"):
            flow_name = "low_confidence_fallback"
        elif res.get("decomposed"):
            flow_name = "decomposed_hybrid_rerank"
        else:
            flow_name = "standard_hybrid_rerank"

        res["flow"] = flow_name

        audit_entry = {
            "timestamp": timestamp,
            "query": question,
            "cached": False,
            "latency_ms": res["latency_ms"],
            "flow": flow_name,
            "sub_queries": res.get("sub_queries", [question]),
            "confidence_score": res.get("confidence_score", 0.0),
            "chunks_sent_count": len(res.get("chunks", [])),
            "chunks": [
                {
                    "id": c.get("id"),
                    "product": c.get("metadata", {}).get("product"),
                    "brand": c.get("metadata", {}).get("brand"),
                    "category": c.get("metadata", {}).get("category"),
                    "price": c.get("metadata", {}).get("price"),
                    "similarity_score": c.get("similarity_score"),
                    "score": c.get("score"),
                }
                for c in res.get("chunks", [])
            ],
            "answer_preview": res.get("answer", "")[:300],
        }
        log_file = log_audit_trail(audit_entry)
        res["audit_file"] = log_file
        res["audit_entry"] = audit_entry

        if use_cache:
            self._cache[cache_k] = res
        return res
