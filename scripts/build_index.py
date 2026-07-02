"""
Build the retrieval index over the SHL catalog.

DESIGN NOTE - why TF-IDF instead of a pretrained neural embedding model:
The original plan was sentence embeddings (fastembed + BAAI/bge-small-en-v1.5,
ONNX so no torch dependency). That code was written and worked -- until the
model download itself failed: fastembed pulls weights from huggingface.co
at first use, and that host was unreachable in this sandboxed environment
(network egress restricted to package registries only). That is not just a
"my sandbox" problem to shrug off -- the same failure mode can hit a
from-scratch Render build if HuggingFace has a bad day, or the box's
network policy is locked down, and it turns a "should take 30 seconds"
cold start into an outage. A retrieval method with zero external runtime
dependencies is a real robustness win, not just a workaround.

TF-IDF is fit entirely on our own 377-document corpus -- no download, no
external service, ever. For this domain specifically (job titles, tool
names, and technology keywords that appear near-verbatim in both the
query and the catalog descriptions -- "Java", "AWS", "Excel", "contact
centre") word-level TF-IDF captures most of the useful signal. What it
gives up is synonym/paraphrase matching a neural embedding would catch
("individual contributor" vs "IC"); the requirement-extraction stage in
the agent (not this script) is where that gap gets partially closed, by
normalizing user phrasing into catalog-shaped terms before it ever hits
this index. Swapping in a real embedding model later is a contained
change (this script and retrieval.py's `embed_query` are the only two
places that would need to change) if a deployment target has reliable
internet to a model hub.

Run: python3 scripts/build_index.py
Reads:  data/catalog.json
Writes: data/tfidf_vectorizer.pkl, data/tfidf_matrix.npz, data/entity_ids.json
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"


def main():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(catalog)} catalog items")

    entity_ids = [item["entity_id"] for item in catalog]
    texts = [item["search_text"] for item in catalog]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,   # dampen raw term-frequency spikes
        min_df=1,            # corpus is tiny (377 docs); don't discard rare-but-real terms
        norm="l2",           # so dot product == cosine similarity at query time
    )
    matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix: {matrix.shape[0]} items x {matrix.shape[1]} vocabulary terms")

    with open(DATA_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    sparse.save_npz(DATA_DIR / "tfidf_matrix.npz", matrix)
    (DATA_DIR / "entity_ids.json").write_text(json.dumps(entity_ids), encoding="utf-8")
    print("Saved vectorizer, matrix, and entity_id ordering to data/")

    # --- Sanity check: does this actually retrieve sensible results? ---
    print("\n--- Sanity check queries ---")
    by_id = {item["entity_id"]: item for item in catalog}
    test_queries = [
        "Java developer who works well with stakeholders",
        "entry level call center customer service agent",
        "senior leadership personality assessment executive",
        "excel and word skills for admin assistant",
        "hiring a Rust engineer for networking infrastructure",
    ]
    for q in test_queries:
        qvec = vectorizer.transform([q])
        sims = cosine_similarity(qvec, matrix)[0]
        top5 = np.argsort(-sims)[:5]
        print(f"\nQuery: {q!r}")
        for rank, idx in enumerate(top5, 1):
            eid = entity_ids[idx]
            item = by_id[eid]
            print(f"  {rank}. {item['name']}  (score={sims[idx]:.3f}, type={item['test_type']})")


if __name__ == "__main__":
    main()
