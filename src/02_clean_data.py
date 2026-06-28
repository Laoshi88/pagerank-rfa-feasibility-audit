from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/wiki-RfA.txt")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_rfa_file(path: Path) -> pd.DataFrame:
    """Parse the SNAP Wikipedia RfA raw text file into a structured DataFrame."""
    records = []
    current = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                if current:
                    records.append(current)
                    current = {}
                continue

            if ":" not in line:
                # Continuation lines are ignored in the simple starter parser.
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key in {"SRC", "TGT", "VOT", "RES", "YEA", "DAT", "TXT"}:
                current[key] = value

        if current:
            records.append(current)

    return pd.DataFrame(records)


def clean_votes(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "SRC": "source",
        "TGT": "target",
        "VOT": "vote",
        "RES": "result",
        "YEA": "year",
        "DAT": "date",
        "TXT": "text",
    }
    df = df.rename(columns=rename_map)

    expected_cols = ["source", "target", "vote", "result", "year", "date", "text"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[expected_cols].copy()
    df = df.dropna(subset=["source", "target", "vote"])

    # Standardize source and target user names as strings.
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    
    # Drop rows where source or target is an empty string after stripping.
    df = df[(df["source"] != "") & (df["target"] != "")].copy()

    df["vote"] = pd.to_numeric(df["vote"], errors="coerce")
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df.dropna(subset=["vote"])
    df["vote"] = df["vote"].astype(int)

    df = df[df["source"] != df["target"]].copy()

    return df


def create_support_edges(votes: pd.DataFrame) -> pd.DataFrame:
    support = votes[votes["vote"] == 1].copy()

    support_edges = (
        support
        .groupby(["source", "target"], as_index=False)
        .agg(
            support_count=("vote", "size"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            accepted_rate=("result", lambda x: (x == 1).mean()),
        )
    )

    return support_edges


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Could not find raw data file at {RAW_PATH}. "
            "Copy wiki-RfA.txt into data/raw/ before running this script."
        )

    print(f"Reading raw file from: {RAW_PATH}")
    raw_votes = parse_rfa_file(RAW_PATH)
    clean = clean_votes(raw_votes)
    support_edges = create_support_edges(clean)

    clean_out = PROCESSED_DIR / "rfa_votes_clean.csv"
    support_out = PROCESSED_DIR / "rfa_support_edges.csv"

    clean.to_csv(clean_out, index=False)
    support_edges.to_csv(support_out, index=False)

    print("Cleaning complete.")
    print(f"Clean vote table saved to: {clean_out}")
    print(f"Support edge list saved to: {support_out}")
    print(f"Clean votes shape: {clean.shape}")
    print(f"Support edges shape: {support_edges.shape}")
    print("\nVote counts:")
    print(clean["vote"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
