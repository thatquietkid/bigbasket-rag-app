"""Follow-up turns must stay on the products the conversation established."""

from rag import LLMResponse, RAGPipeline


def doc(pid, product, price, brand="TestBrand", sim=0.8):
    text = f"Product: {product}\nBrand: {brand}\nPrice: ₹{price}\nDescription: a face wash"
    return {
        "id": pid,
        "content": text,
        "text": text,
        "metadata": {
            "source": "BigBasket Catalogue",
            "source_file": "BigBasket Products.csv",
            "product": product,
            "brand": brand,
            "category": "Beauty & Hygiene",
            "price": float(price),
            "page": "row-1",
            "chunk": 0,
        },
        "similarity_score": sim,
        "score": sim,
        "similarity": sim,
    }


HISTORY = [
    {"role": "user", "content": "Which face wash suits oily skin, and what does it cost?"},
    {
        "role": "assistant",
        "content": "Cetaphil Oily Skin Cleanser 236 ml costs ₹1,034 [1].",
        "chunks": [
            {
                "metadata": {
                    "product": "Cetaphil Oily Skin Cleanser",
                    "brand": "Cetaphil",
                    "price": 1034.0,
                }
            }
        ],
    },
]


class StubRetriever:
    ids = ["1"]
    metas = [{}]

    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def retrieve(self, query, top_k=5, score_threshold=0.0, mode="hybrid", rerank=True):
        self.queries.append(query)
        return [dict(d) for d in self.docs]


class StubLLM:
    """Stand-in that rewrites follow-ups without repeating the product name."""

    def __init__(self, rewrite="cheaper face wash options for oily skin"):
        self.rewrite = rewrite
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        if "rephrase the follow-up" in prompt:
            return LLMResponse(content=self.rewrite)
        return LLMResponse(content="Cetaphil Oily Skin Cleanser costs ₹1,034 [1].")


def make_pipeline(docs):
    retriever = StubRetriever(docs)
    llm = StubLLM()
    return RAGPipeline(retriever=retriever, llm=llm), retriever, llm


FACE_WASHES = [
    doc("1", "Cetaphil Oily Skin Cleanser", 1034.0, brand="Cetaphil"),
    doc("2", "Volcanic Green Clay Face Wash", 406.0, brand="RawNature"),
    doc("3", "Tea Tree Oil Face Wash", 400.0, brand="Nyassa"),
]


def test_follow_up_retrieval_keeps_the_previous_product_context():
    pipeline, retriever, _ = make_pipeline(FACE_WASHES)

    pipeline.answer("Any cheaper alternative?", chat_history=HISTORY)

    query = retriever.queries[0].lower()
    assert "cetaphil" in query
    assert "face wash" in query


def test_pronoun_follow_up_resolves_to_the_discussed_product():
    pipeline, retriever, _ = make_pipeline(FACE_WASHES)

    pipeline.answer("Do these two suit oily skin?", chat_history=HISTORY)

    assert "cetaphil" in retriever.queries[0].lower()


def test_answer_prompt_includes_the_previous_turn():
    pipeline, _, llm = make_pipeline(FACE_WASHES)

    pipeline.answer("Do these two suit oily skin?", chat_history=HISTORY)

    prompt = llm.prompts[-1]
    assert "Cetaphil Oily Skin Cleanser" in prompt
    assert "Which face wash suits oily skin" in prompt


def test_new_question_is_not_anchored_to_the_previous_topic():
    pipeline, retriever, _ = make_pipeline(FACE_WASHES)

    pipeline.answer("What is the price of Tata Salt?", chat_history=HISTORY)

    assert "cetaphil" not in retriever.queries[0].lower()


def test_cheaper_follow_up_excludes_more_expensive_products():
    pipeline, _, _ = make_pipeline(FACE_WASHES)

    result = pipeline.answer("Any cheaper alternative?", chat_history=HISTORY)

    prices = [c["metadata"]["price"] for c in result["chunks"]]
    assert prices and all(p < 1034.0 for p in prices)
    assert result["chunks"]
