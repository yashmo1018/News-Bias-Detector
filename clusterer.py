"""
Layer 3 — Clusterer
Embeds article titles+text with all-MiniLM-L6-v2, clusters with DBSCAN,
writes clustered.json with group_id and cluster_label added to each article.

Usage:
    pip install sentence-transformers scikit-learn
    python clusterer.py
    python clusterer.py --eps 0.3       # tighter clusters
    python clusterer.py --min-samples 1 # no noise points
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

INPUT_FILE  = "analyzed.json"
OUTPUT_FILE = "clustered.json"

# DBSCAN defaults — tune these if clusters feel too tight/loose
DEFAULT_EPS         = 0.35   # cosine distance threshold (0=identical, 2=opposite)
DEFAULT_MIN_SAMPLES = 2      # min articles to form a cluster (1 = no noise)


def load_articles(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_texts(articles: list[dict]) -> list[str]:
    """Combine title + first 500 chars of body for embedding."""
    texts = []
    for a in articles:
        title = a.get("title", "")
        body  = a.get("text", "")[:500]
        texts.append(f"{title}. {body}")
    return texts


def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    print(f"[clusterer] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[clusterer] Embedding {len(texts)} articles …")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # L2-normalize so cosine distance = euclidean distance on unit sphere
    return normalize(embeddings, norm="l2")


def cluster(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    print(f"[clusterer] Running DBSCAN (eps={eps}, min_samples={min_samples}) …")
    dist_matrix = cosine_distances(embeddings).astype(np.float64)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    return labels


def make_cluster_label(articles_in_cluster: list[dict]) -> str:
    """
    Derive a human-readable label from the most common words in titles.
    Simple approach: pick the title whose words overlap most with others.
    """
    titles = [a.get("title", "") for a in articles_in_cluster]
    if len(titles) == 1:
        return titles[0][:60]

    # score each title by how many words appear in other titles
    def overlap_score(t: str, others: list[str]) -> int:
        words = set(t.lower().split())
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "of", "and",
                     "or", "is", "was", "are", "for", "with", "as", "by", "it"}
        words -= stopwords
        return sum(1 for o in others for w in o.lower().split() if w in words)

    best_title = max(titles, key=lambda t: overlap_score(t, [x for x in titles if x != t]))
    label = best_title[:60]
    return label + ("…" if len(best_title) > 60 else "")


def assign_clusters(articles: list[dict], labels: np.ndarray) -> list[dict]:
    unique_labels = sorted(set(labels))
    noise_counter = 0

    # First pass: collect articles per cluster for labelling
    cluster_map: dict[int, list[dict]] = {}
    for label in unique_labels:
        if label == -1:
            continue
        cluster_map[label] = [articles[i] for i, l in enumerate(labels) if l == label]

    cluster_labels: dict[int, str] = {
        label: make_cluster_label(arts)
        for label, arts in cluster_map.items()
    }

    # Second pass: annotate articles
    result = []
    for i, article in enumerate(articles):
        a = dict(article)
        raw_label = int(labels[i])
        if raw_label == -1:
            noise_counter += 1
            a["group_id"]      = f"noise_{noise_counter}"
            a["cluster_label"] = "Unclustered"
            a["is_noise"]      = True
        else:
            a["group_id"]      = f"cluster_{raw_label}"
            a["cluster_label"] = cluster_labels[raw_label]
            a["is_noise"]      = False
        result.append(a)

    return result


def print_summary(articles: list[dict]) -> None:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for a in articles:
        groups[a["group_id"]].append(a)

    noise   = [g for g in groups if g.startswith("noise_")]
    clusters = [g for g in groups if not g.startswith("noise_")]

    print(f"\n{'='*60}")
    print(f"  CLUSTERING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total articles : {len(articles)}")
    print(f"  Clusters found : {len(clusters)}")
    print(f"  Noise articles : {len(noise)}")
    print(f"{'='*60}")

    for gid in sorted(clusters):
        arts = groups[gid]
        label = arts[0]["cluster_label"]
        print(f"\n  [{gid}] {label}")
        for a in arts:
            print(f"    • [{a['source']}] {a['title'][:70]}")

    if noise:
        print(f"\n  [NOISE — unclustered]")
        for gid in sorted(noise):
            a = groups[gid][0]
            print(f"    • [{a['source']}] {a['title'][:70]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="NBP Layer 3 — Article Clusterer")
    parser.add_argument("--input",       default=INPUT_FILE,        help="Input JSON (analyzed.json)")
    parser.add_argument("--output",      default=OUTPUT_FILE,       help="Output JSON (clustered.json)")
    parser.add_argument("--eps",         type=float, default=DEFAULT_EPS,
                        help="DBSCAN eps: cosine distance threshold (default 0.35)")
    parser.add_argument("--min-samples", type=int,   default=DEFAULT_MIN_SAMPLES,
                        help="DBSCAN min_samples (default 2)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] {args.input} not found. Run analyzer.py first.")
        return

    articles   = load_articles(args.input)
    print(f"[clusterer] Loaded {len(articles)} articles from {args.input}")

    texts      = build_texts(articles)
    embeddings = embed(texts)
    labels     = cluster(embeddings, eps=args.eps, min_samples=args.min_samples)
    annotated  = assign_clusters(articles, labels)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print_summary(annotated)
    print(f"[clusterer] ✅ Saved → {args.output}")


if __name__ == "__main__":
    main()
