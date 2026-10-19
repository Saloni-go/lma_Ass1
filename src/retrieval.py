"""Pipeline 1: dense Bi-Encoder retrieval with a FAISS inner-product index."""
import hashlib
import json
import time

import faiss
import numpy as np

from .embeddings import embed_texts


def build_index(chunk_embeddings):
    """Exact (brute-force) inner-product index over normalised vectors = cosine search."""
    norms = np.linalg.norm(chunk_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "chunk embeddings must be L2-normalised"
    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    return index


def corpus_fingerprint(chunks, model_name):
    """Identifies the (chunk texts, embedding model) pair a cached index was built from."""
    h = hashlib.sha256(model_name.encode())
    for c in chunks:
        h.update(f"{c['chunk_id']}:{c['chunk_hash']}".encode())
    return h.hexdigest()


def load_or_build_index(chunks, bi_encoder, model_name, cache_dir):
    """Return (index, id_map, embeddings, built_now).

    id_map[faiss_row] -> chunk_id. Embeddings, index and id_map are cached in
    `cache_dir`; the cache is reused only if its fingerprint matches the
    current chunks and model, so re-chunking automatically triggers a rebuild.
    """
    emb_path, idx_path, meta_path = (cache_dir / "embeddings.npy", cache_dir / "faiss.index",
                                     cache_dir / "index_meta.json")
    fp = corpus_fingerprint(chunks, model_name)
    if meta_path.exists() and emb_path.exists() and idx_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            index = faiss.read_index(str(idx_path))
            embeddings = np.load(emb_path)
            assert index.ntotal == len(meta["id_map"]) == len(chunks)
            return index, meta["id_map"], embeddings, False

    embeddings = embed_texts(bi_encoder, [c["text"] for c in chunks])
    index = build_index(embeddings)
    id_map = [c["chunk_id"] for c in chunks]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    faiss.write_index(index, str(idx_path))
    meta_path.write_text(json.dumps({
        "fingerprint": fp, "model": model_name, "n_vectors": index.ntotal,
        "dim": index.d, "metric": "inner_product_on_l2_normalised (= cosine)",
        "id_map": id_map,
    }, indent=1))
    return index, id_map, embeddings, True


def baseline_retrieve(query, bi_encoder, index, chunks_by_id, id_map, top_k=10,
                      return_timing=False):
    """Retrieve the Top-k chunks for a query.

    Returns a list of candidate dicts (rank 1..k) holding chunk_id, pages,
    text and cosine similarity. These exact candidates are later handed to
    the Cross-Encoder, so Pipeline 2 never performs a second retrieval.
    """
    # FAISS pads missing results with row -1, which would silently map to the last chunk
    assert top_k <= index.ntotal, f"top_k={top_k} exceeds the {index.ntotal} indexed chunks"
    t0 = time.perf_counter()
    q = embed_texts(bi_encoder, [query])
    t1 = time.perf_counter()
    scores, rows = index.search(q, top_k)
    t2 = time.perf_counter()

    results = []
    for rank, (row, score) in enumerate(zip(rows[0], scores[0]), start=1):
        c = chunks_by_id[id_map[int(row)]]
        results.append({
            "chunk_id": c["chunk_id"],
            "baseline_rank": rank,
            "cosine_score": float(score),
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "pages": c["pages"],
            "text": c["text"],
        })
    if return_timing:
        return results, {"embed_s": t1 - t0, "search_s": t2 - t1}
    return results
