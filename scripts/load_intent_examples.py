"""intent_examples 컬렉션 샘플 적재."""

from __future__ import annotations

import os

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from board_bot.llm.factory import create_embeddings

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
PROVIDER = os.getenv("LLM_PROVIDER", "google")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

SAMPLES = {
    "상품 누락": ["상자에 주문한 물건 하나가 빠졌어요", "세트 중 일부가 누락되었습니다"],
    "상품 오배송": ["다른 상품이 왔어요", "주문하지 않은 색상이 배송됐습니다"],
    "상품 미배송": ["배송완료라고 뜨는데 못 받았습니다", "아직 도착하지 않았습니다"],
    "품절": ["결제 후 품절 안내를 받았습니다", "재고가 없다고 연락받았습니다"],
    "상품 파손": ["유리병이 깨져서 도착했어요", "상품이 파손된 상태로 왔습니다"],
    "주문 취소": ["주문을 취소하고 싶습니다", "출고 전 취소 요청합니다"],
}


def _embed(texts: list[str]) -> list[list[float]]:
    if API_KEY:
        return create_embeddings(provider=PROVIDER, model=EMBED_MODEL, api_key=API_KEY).embed_documents(texts)
    return [[0.1] * 8 for _ in texts]


def main():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    probe = _embed(["dim_probe"])[0]
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="intent_label", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="example_text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(probe)),
    ]
    if utility.has_collection("intent_examples"):
        coll = Collection("intent_examples")
    else:
        coll = Collection("intent_examples", schema=CollectionSchema(fields=fields, description="intent examples"))
        coll.create_index("embedding", {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})

    labels, texts = [], []
    for label, examples in SAMPLES.items():
        for t in examples:
            labels.append(label)
            texts.append(t)
    coll.insert([labels, texts, _embed(texts)])
    coll.flush()
    print("intent examples loaded")


if __name__ == "__main__":
    main()
