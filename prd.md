# PRD & Implementation Specification: BigBasket RAG Assistant

## 1. Executive Summary & Problem Definition
- **Industry & Company**: E-Commerce & Quick/Scheduled Grocery Retail — BigBasket (Tata Enterprise).
- **Target User**: Grocery consumers seeking specific dietary solutions (e.g. low-GI, gluten-free, diabetic-friendly), budget-constrained bundles, price comparisons across brands, and policy verification (return windows, cancellation, refund timelines).
- **Core Problem**: Traditional e-commerce search relies strictly on keyword matching (elastic search/BM25) and fails on semantic, multi-intent, or comparative queries (e.g., *"Find 2 gluten-free snacks under ₹200 and tell me if I can return them if damaged"*). Pure LLMs hallucinate stock, prices, and policies.
- **RAG Solution**: A hybrid retrieval pipeline (Dense Vector + BM25 with Reciprocal Rank Fusion) augmented with Query Decomposition for multi-part questions, Cross-Encoder Re-ranking, Dual-Tier Caching, Safety/Relevance Guardrails, and low-latency inference via Groq (`chatgpt-oss-20b` with automated fallback to `llama-3.1-8b-instant` and Gemini).

---

## 2. Technical Architecture & End-to-End Flow

```
                      User Question
                           │
                           ▼
              ┌─────────────────────────┐
              │   Input Guardrails      │ ──► [Reject prompt injections / off-topic]
              └────────────┬────────────┘
                           │ (Passed)
                           ▼
              ┌─────────────────────────┐
              │    Exact Hash Cache     │ ──► (Hit: SHA-256 normalized match)
              │    (In-memory / instant)│     Return cached response (< 2ms)
              └────────────┬────────────┘
                           │ (Cache Miss)
                           ▼
              ┌─────────────────────────┐
              │ Query Decomposition     │ ──► Splits multi-part questions if detected
              │ (Heuristic / LLM split) │     into atomic sub-queries [q1, q2, ...]
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────────────────┐
              │ For each sub-query:                 │
              │   1. BM25 Sparse Search (Top-15)    │
              │   2. ChromaDB Dense Search (Top-15) │
              │   3. Reciprocal Rank Fusion (RRF)   │
              │   4. Cross-Encoder Re-ranking       │
              │      (ms-marco-MiniLM-L-6-v2)       │
              └────────────┬────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Deduplicated Top Chunks │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Relevance Guardrail     │ ──► (Best similarity < 0.30? Trigger Fallback)
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Multi-Provider LLM      │
              │ 1. Groq: chatgpt-oss-20b│ ──► (404/Quota failover)
              │ 2. Groq: llama-3.1-8b   │ ──► (Cross-provider failover)
              │ 3. Gemini-2.5/1.5-flash │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Output Faithfulness    │
              │  Guardrail & Citation   │
              └────────────┬────────────┘
                           │
                           ▼
              [Store in Exact Cache] ──► Render in Streamlit UI
```

---

## 3. Detailed Component Specifications

### 3.1 Data Preparation & Cleaning (5,000 Sample)
- **Source**: Kaggle BigBasket Product Catalog (27k rows) sampled to 5,000 records with fixed random seed (42).
- **Cleaning Filters**:
  1. Remove rows with null `product` or null `description`.
  2. Filter out placeholder/low-information descriptions (length < 30 characters).
  3. Strip redundant whitespace, trailing escape characters, and HTML formatting.
  4. Deduplicate on `(product, brand)` and `description` text.
  5. Include BigBasket Policy Document (`data/bigbasket_policies.txt`) covering returns, cancellations, perishable goods rules, and refunds.

### 3.2 Chunking Strategy
- **Products**: 1 Document per product. Structured schema:
  ```text
  [PRODUCT]
  Name: {product}
  Brand: {brand}
  Category: {category} > {sub_category} > {type}
  Sale Price: ₹{sale_price} | Market Price: ₹{market_price}
  Rating: {rating}★
  Description: {description}
  ```
  Descriptions exceeding 800 characters are chunked with a 100-character overlap, repeating the product metadata header on each chunk.
- **Policies**: Chunked by semantic sections (500 characters, 100 character overlap).

### 3.3 Embeddings & Vector Store
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, running locally on CPU).
- **Vector Database**: `ChromaDB` (Persistent cosine distance metric).

### 3.4 Hybrid Retrieval & Re-Ranking
- **Sparse**: `rank_bm25.BM25Okapi` initialized on tokenized corpus.
- **Dense**: ChromaDB cosine search.
- **Fusion**: Reciprocal Rank Fusion (RRF with $k=60$):
  $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{60 + rank_m(d)}$$
- **Re-Ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks top candidates down to the top-4 most relevant context chunks.

### 3.5 Query Decomposition
- When a query contains multi-part intents (e.g. comparing two disparate products or asking for both a product recommendation and a policy verification):
  - Fast query decomposition breaks the question into atomic sub-queries.
  - Hybrid retrieval executes for each sub-query.
  - The retrieved context lists are merged, deduplicated, and passed to the LLM for unified synthesis.

### 3.6 Fast Caching Layer
- **Exact Hash Cache**: In-memory hash map (`hashlib.sha256(normalized_query)` -> `{answer, sources, latency}`). Returns in $< 1$ ms with zero risk of semantic false positives across similar SKUs.
- **Demo Pre-warming**: Pre-populates the cache with representative test questions to guarantee instant live demo responses.

### 3.7 Guardrails & Safety
1. **Input Guardrail**: Pattern matching rejecting prompt injections ("ignore previous instructions") and unsupported out-of-domain requests.
2. **Relevance Threshold Guardrail**: If the top retrieved chunk similarity is below `0.30`, bypass LLM synthesis and return the standard fallback message.
3. **Output Faithfulness Guardrail**: Enforces inline citations `[Product N]` or `[Policy Section]` and verifies that answers are strictly grounded in retrieved context.

### 3.8 LLM Orchestration & Failover (Groq + Gemini)
- **Primary Model**: `chatgpt-oss-20b` (via Groq API).
- **Secondary Failover**: `llama-3.1-8b-instant` (via Groq API).
- **Tertiary Cross-Provider Failover**: Google Gemini (`gemini-2.5-flash` / `gemini-1.5-flash` via `google-genai`).

---

## 4. Evaluation Suite & Test Cases

The evaluation script (`evaluate.py`) benchmarks:
1. **Factual Lookup**: Exact price, brand, and specifications.
2. **Comparison Query**: Head-to-head comparison between two competing brands.
3. **Multi-Part / Decomposed Query**: Product recommendation paired with return policy verification.
4. **Out-of-Dataset / Unsupported Query**: Explicit out-of-domain question testing refusal without hallucination.
