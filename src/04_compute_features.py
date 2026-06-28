from pathlib import Path
import pandas as pd
import networkx as nx

EDGE_PATH = Path("data/processed/rfa_support_edges.csv")
PAGERANK_PATH = Path("outputs/pagerank_scores.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    edges = pd.read_csv(EDGE_PATH, keep_default_na=False)
    pr = pd.read_csv(PAGERANK_PATH, keep_default_na=False)
    pagerank = dict(zip(pr["node"], pr["pagerank"]))

    G = nx.DiGraph()
    for row in edges.itertuples(index=False):
        G.add_edge(row.source, row.target, weight=row.support_count, support_count=row.support_count)

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    rows = []
    for row in edges.itertuples(index=False):
        source = row.source
        target = row.target

        source_pr = pagerank.get(source, 0.0)
        target_pr = pagerank.get(target, 0.0)

        source_in = in_degree.get(source, 0)
        source_out = out_degree.get(source, 0)
        target_in = in_degree.get(target, 0)
        target_out = out_degree.get(target, 0)

        rows.append({
            "source": source,
            "target": target,
            "support_count": row.support_count,
            "source_in_degree": source_in,
            "source_out_degree": source_out,
            "target_in_degree": target_in,
            "target_out_degree": target_out,
            "in_degree_diff": target_in - source_in,
            "out_degree_diff": target_out - source_out,
            "reciprocal": 1 if G.has_edge(target, source) else 0,
            "pagerank_diff": target_pr - source_pr,
            "uphill": 1 if target_pr > source_pr else 0,
        })

    feature_df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "edge_features.csv"
    feature_df.to_csv(out_path, index=False)

    print(f"Feature table saved to: {out_path}")
    print(feature_df.head())
    print("\nTarget distribution:")
    print(feature_df["uphill"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
