from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_PATH = Path("data/processed/edge_features.csv")
PAGERANK_PATH = Path("outputs/pagerank_scores.csv")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    features = pd.read_csv(FEATURE_PATH, keep_default_na=False)
    pagerank = pd.read_csv(PAGERANK_PATH, keep_default_na=False)

    plt.figure()
    plt.hist(features["source_in_degree"], bins=50)
    plt.xlabel("Source In-Degree")
    plt.ylabel("Number of Edges")
    plt.title("Distribution of Source In-Degree")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "source_in_degree_distribution.png", dpi=200)

    plt.figure()
    plt.hist(features["target_in_degree"], bins=50)
    plt.xlabel("Target In-Degree")
    plt.ylabel("Number of Edges")
    plt.title("Distribution of Target In-Degree")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_in_degree_distribution.png", dpi=200)

    plt.figure()
    plt.hist(pagerank["pagerank"], bins=50)
    plt.xlabel("PageRank Score")
    plt.ylabel("Number of Nodes")
    plt.title("PageRank Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "pagerank_distribution.png", dpi=200)

    plt.figure()
    plt.hist(features["pagerank_diff"], bins=50)
    plt.xlabel("Target PageRank - Source PageRank")
    plt.ylabel("Number of Edges")
    plt.title("Edge PageRank Difference Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "pagerank_edge_difference.png", dpi=200)

    uphill_counts = features["uphill"].value_counts().sort_index()
    plt.figure()
    plt.bar(["Non-uphill", "Uphill"], uphill_counts.values)
    plt.xlabel("Edge Type")
    plt.ylabel("Number of Edges")
    plt.title("Count of PageRank-Uphill vs Non-Uphill Edges")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "uphill_edge_counts.png", dpi=200)

    print("Figures saved to figures/.")


if __name__ == "__main__":
    main()
