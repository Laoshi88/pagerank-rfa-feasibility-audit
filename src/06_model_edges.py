from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

FEATURE_PATH = Path("data/processed/edge_features.csv")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
    else:
        roc_auc = None

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc,
    }, preds


def main():
    df = pd.read_csv(FEATURE_PATH, keep_default_na=False)
    y = df["uphill"]
    drop_cols = ["source", "target", "uphill", "pagerank_diff"]
    X = df.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "baseline_majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results = []
    best_name = None
    best_f1 = -1
    best_preds = None

    for name, model in models.items():
        metrics, preds = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_preds = preds

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_preds,
    display_labels=["Non-uphill (0)", "Uphill (1)"],
    values_format="d"
    )
    
    plt.title(f"Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=200)

    print("Model metrics:")
    print(results_df)
    print(f"Best model by F1: {best_name}")
    print(f"Saved metrics to {OUTPUT_DIR / 'model_metrics.csv'}")
    print(f"Saved confusion matrix to {FIGURE_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
