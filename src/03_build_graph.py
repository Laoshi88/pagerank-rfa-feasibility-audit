from pathlib import Path
import json
import pandas as pd
import networkx as nx

EDGE_PATH = Path("data/processed/rfa_support_edges.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not EDGE_PATH.exists():
        raise FileNotFoundError(f"Could not find {EDGE_PATH}. Run src/02_clean_data.py first.")

    edges = pd.read_csv(EDGE_PATH, keep_default_na=False)
    G = nx.DiGraph()

    for row in edges.itertuples(index=False):
        G.add_edge(
            row.source,
            row.target,
            weight=row.support_count,
            support_count=row.support_count,
            first_year=row.first_year,
            last_year=row.last_year,
            accepted_rate=row.accepted_rate,
        )

    pagerank = nx.pagerank(G, alpha=0.85)

    pr_df = (
        pd.DataFrame.from_dict(pagerank, orient="index", columns=["pagerank"])
        .reset_index()
        .rename(columns={"index": "node"})
        .sort_values("pagerank", ascending=False)
    )

    pr_df.to_csv(OUTPUT_DIR / "pagerank_scores.csv", index=False)
    pr_df.head(25).to_csv(OUTPUT_DIR / "top_25_pagerank_nodes.csv", index=False)

    summary = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "self_loops": nx.number_of_selfloops(G),
        "weakly_connected_components": nx.number_weakly_connected_components(G),
        "strongly_connected_components": nx.number_strongly_connected_components(G),
    }

    with open(OUTPUT_DIR / "graph_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Graph construction and PageRank complete.")
    print(summary)
    print("\nTop PageRank nodes:")
    print(pr_df.head(10))


if __name__ == "__main__":
    main()
