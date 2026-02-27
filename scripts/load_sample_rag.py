from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

connections.connect(host="localhost", port="19530")
for name in ["rag_policy", "rag_script", "rag_manual"]:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=8),
    ]
    schema = CollectionSchema(fields=fields, description=f"{name} sample")
    c = Collection(name=name, schema=schema)
    c.insert([["샘플 정책 문서"], [[0.1] * 8]])
print("sample rag loaded")
