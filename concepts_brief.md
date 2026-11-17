# BigBasket RAG Assistant — Concepts Brief & Technical Architecture

## 1. Executive Summary & Problem Framing
- **Target Company & Industry**: BigBasket (Tata Enterprise) — Online Food, Grocery & FMCG Quick/Scheduled Commerce.
- **Target Persona**: Time-pressed grocery shoppers with dietary constraints (vegan, gluten-free, diabetic-friendly), budget filters, brand fidelity, or queries regarding return/cancellation policies.
- **Core Problem Statement**: Keyword-based search engines in e-commerce fail at multi-attribute natural language questions (e.g., *"Suggest 3 high-protein breakfast items under ₹250 each"* or *"Does this coconut oil contain preservatives and can I return it if opened?"*). A pure LLM hallucinates prices and stock, whereas a standard keyword search lacks semantic comprehension.
- **Solution**: A low-latency, 100% cost-free local RAG system running on CPU embeddings (`all-MiniLM-L6-v2`), BM25 keyword matching with Reciprocal Rank Fusion (RRF), in-memory semantic caching, and Groq's high-speed inference engine (`chatgpt-oss-20b` or Groq's open model endpoints) wrapped in a Streamlit UI.

---

## 2. Technical Decisions & Component Architecture

```
User Query
    │
    ▼
[Semantic / Exact Cache (SQLite / Dict + Cosine)] ──(Hit)──► Return Cached Answer (< 50ms)
    │ (Miss)
    ▼
[Query Analysis & Metadata Pre-Filtering] (Price bounds, category, brand keywords)
    │
    ├───► [BM25 Inverted Index] (Exact terms: brand names, weights, SKUs) ──► Top-15
    │                                                                             │
    └───► [Dense Vector Retrieval (ChromaDB + all-MiniLM-L6-v2)] ────────► Top-15
                                                                                  │
                                                                                  ▼
                                                            [Reciprocal Rank Fusion (RRF, k=60)]
                                                                                  │
                                                                                  ▼
                                                              [Lightweight Cross-Encoder Re-ranker]
                                                                (cross-encoder/ms-marco-MiniLM-L-6-v2)
                                                                                  │
                                                                                  ▼
                                                                        Top-K Context Chunks (k=4)
                                                                                  │
                                                                                  ▼
                                                      [Groq API: chatgpt-oss-20b / open weights]
                                                      (Strict ground-truth prompt + Citation indices)
                                                                                  │
                                                                                  ▼
                                                                           Streamlit Output
```

### A. Data Cleaning Strategy
1. **Deduplication**: Strict deduplication on `(product, brand)` tuples and `description` MD5 hashes.
2. **Noise Removal**: Clean HTML entities, scraped newline clutter, and non-ASCII artifacts.
3. **Information Density Filter**: Drop items with description length < 40 characters (e.g., placeholder rows like "imported item" or "good quality").
4. **Synthetic Policy Enrichment**: Inject structured text documents for BigBasket policies (return window, refund timelines, per-category return eligibility like perishables vs. staples) to fulfill policy Q&A requirements.

### B. Document Construction & Chunking Strategy
- **Catalog Items (Granular Products)**:
  - **Structure**: 1 Product = 1 Canonical Document.
  - **Header Injection**:
    ```text
    Product: {product_name}
    Brand: {brand}
    Category: {category} > {sub_category} > {type}
    Price: ₹{sale_price} (MRP: ₹{market_price})
    Rating: {rating}★
    Description: {description}
    ```
  - **Chunk Boundary**: Most catalog entries are 150–250 tokens; keep them as atomic documents without mid-sentence slicing to avoid separating price/brand from specs. For descriptions > 800 characters, use recursive character splitting with a 100-character overlap, prepending the Product Header to every continuation chunk.
- **Policy Documents**:
  - Split by section headers (e.g., "Returns & Refunds", "Delivery Slots", "Cancellation") with a 500-token window and 50-token overlap.

### C. Embedding & Vector Database
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
  - *Why*: Free, runs locally on laptop CPU, executes a query embedding in ~15–25ms, zero network dependency.
- **Vector Database**: `ChromaDB` (Persistent, cosine similarity space).
  - *Fallback Guard*: If max cosine similarity < 0.32, trigger fallback message immediately without dispatching an LLM API call.

### D. Hybrid Retrieval & Re-Ranking
- **Hybrid Retrieval**:
  - Dense Vector (ChromaDB) captures semantic intent ("healthy snack without gluten").
  - Sparse Keyword (BM25Okapi) captures exact entities ("Saffola", "Fortune", "Aashirvaad", "500g").
  - Merge scores via Reciprocal Rank Fusion: RRF_Score(d) = sum(1 / (60 + rank_i(d))).
- **Local Re-ranking (Speed-Optimized)**:
  - Apply `cross-encoder/ms-marco-MiniLM-L-6-v2` only to top-10 fused candidates, trimming to final top-4 for prompt injection. Can be toggled on/off to benchmark latency.

### E. Free & Ultra-Fast LLM Layer (Groq)
- **Engine**: Groq Cloud Free Tier via `groq` Python client.
- **Model Choice**: `openai/chatgpt-oss-20b` (or Groq's high-throughput open models like `llama-3.1-8b-instant`).
- **Prompt Engineering**:
  - Zero-hallucination instruction: System prompt forbids using internal parametric memory for specific prices or specs.
  - Bracket citations: Every claim must cite `[Product N]`.
  - Strict fallback phrasing when retrieved context lacks evidence.

### F. Fast Local Caching Layer
- **Two-Tier Cache**:
  1. **Exact Match Cache**: Hash table (`hashlib.sha256(query.strip().lower())` -> answer + chunks). Latency: < 1ms.
  2. **Semantic Cache**: Store past query embeddings in memory; if cosine similarity between current query and cached query > 0.94, serve cached response directly without calling Groq.

---

## 3. Presentation & Evaluation Alignment (5-Minute Slot Plan)

| Time | Agenda Item | Key Talking Points |
| :--- | :--- | :--- |
| **0:00 - 1:00** | Problem Statement & Architecture | BigBasket product discovery problem; semantic search vs. keyword limitations; explain the RAG pipeline flow. |
| **1:00 - 3:45** | Live Chatbot Demo (4 Queries) | 1. **Factual lookup**: Exact price & packaging for a branded staple.<br>2. **Comparison**: Compare two competing olive oils or oats brands.<br>3. **Dietary Constraint / Multi-part**: Gluten-free snack under ₹150.<br>4. **Out-of-domain / Fallback**: "Can I book a flight to Mumbai on BigBasket?" (Demonstrating graceful refusal). |
| **3:45 - 5:00** | Q&A & Error Analysis | Discussion of RRF merging weights, cold-start latency, and handling truncated descriptions in scraped datasets. |
