"""외부 Milvus 컬렉션 스키마/인덱스 검증 스크립트.

운영 전제:
- Milvus는 외부 서비스(운영 클러스터)를 사용한다.
- 운영 계정은 컬렉션 생성/인덱스 변경 권한이 없을 수 있다.

동작:
1) 지정 컬렉션 존재 여부 확인
2) dense_vector/sparse_vector 필드 존재 확인
3) 미존재/부족 시 명확한 오류와 대안(v2 컬렉션 생성+재적재) 안내
4) --allow-create=true 인 경우에만 PoC용 컬렉션 생성
"""

from __future__ import annotations

import argparse

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from board_bot.services.config import AppConfig


def validate_or_create(collection_name: str, allow_create: bool) -> None:
    cfg = AppConfig(system_path="config/system.yaml", policy_path="config/policy.yaml", compat_path="config/app.yaml")
    host = cfg.get("milvus", "host", default="127.0.0.1")
    port = cfg.get("milvus", "port", default=19530)

    connections.connect(alias="default", host=str(host), port=str(port))

    if not utility.has_collection(collection_name):
        if not allow_create:
            raise RuntimeError(
                f"Milvus 컬렉션 '{collection_name}' 이(가) 없습니다. "
                "운영 환경에서는 스키마 변경 권한이 제한될 수 있습니다. "
                "대안: board_bot_rag_v2 신규 컬렉션 생성 후 데이터 재적재 절차를 수행하세요."
            )

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=3072),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        coll = Collection(collection_name, schema=CollectionSchema(fields=fields, description="board-bot hybrid v2"))
        coll.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "COSINE"})
        coll.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
        print(f"created collection: {collection_name}")
        return

    coll = Collection(collection_name)
    fields = {f.name for f in coll.schema.fields}
    required = {"dense_vector", "sparse_vector"}
    missing = sorted(required - fields)
    if missing:
        raise RuntimeError(
            f"컬렉션 '{collection_name}' 는 hybrid 필수 필드가 부족합니다: {missing}. "
            "운영 컬렉션 변경이 불가하면 v2 컬렉션을 별도 생성하고 재적재하세요."
        )

    idx = {i.field_name for i in coll.indexes}
    print(f"ok collection={collection_name} fields={sorted(required)} indexes={sorted(idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=None)
    parser.add_argument("--allow-create", action="store_true")
    args = parser.parse_args()

    cfg = AppConfig(system_path="config/system.yaml", policy_path="config/policy.yaml", compat_path="config/app.yaml")
    col = args.collection or cfg.get("milvus", "collection_name", default="semas_v2")
    validate_or_create(collection_name=col, allow_create=args.allow_create)
