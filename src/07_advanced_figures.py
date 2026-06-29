from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay


FEATURE_PATH = Path("data/processed/edge_features.csv")
SUPPORT_EDGES_PATH = Path("data/processed/rfa_support_edges.csv")
FIGURE_DIR = Path("figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def make_model_diagnostic_figures():
    df = pd.read_csv(FEATURE_PATH, keep_default_na=False)

    y = df["uphill"]

    drop_cols = [
        "source",
        "target",
        "uphill",
        "pagerank_diff",
    ]

    X = df.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ROC curve
    plt.figure()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("Random Forest ROC Curve")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve_random_forest.png", dpi=200)

    # Precision-recall curve
    plt.figure()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Random Forest Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "precision_recall_curve_random_forest.png", dpi=200)

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(importances.index, importances.values)
    plt.xlabel("Random Forest Feature Importance")
    plt.title("Feature Importance for Predicting PageRank-Uphill Edges")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "random_forest_feature_importance.png", dpi=200)


def make_temporal_uphill_figure():
    features = pd.read_csv(FEATURE_PATH, keep_default_na=False)
    support_edges = pd.read_csv(SUPPORT_EDGES_PATH, keep_default_na=False)

    merged = features.merge(
        support_edges[["source", "target", "first_year"]],
        on=["source", "target"],
        how="left",
    )

    yearly = (
        merged
        .dropna(subset=["first_year"])
        .groupby("first_year", as_index=False)
        .agg(
            uphill_share=("uphill", "mean"),
            edge_count=("uphill", "size")
        )
    )

    plt.figure()
    plt.plot(yearly["first_year"], yearly["uphill_share"], marker="o")
    plt.xlabel("First Year of Support Edge")
    plt.ylabel("Share of Edges That Are PageRank-Uphill")
    plt.title("PageRank-Uphill Share by Year")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "uphill_share_by_year.png", dpi=200)


def main():
    make_model_diagnostic_figures()
    make_temporal_uphill_figure()
    print("Advanced figures saved to figures/.")


if __name__ == "__main__":
    main()