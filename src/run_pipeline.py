import subprocess
import sys

SCRIPTS = [
    "src/02_clean_data.py",
    "src/03_build_graph.py",
    "src/04_compute_features.py",
    "src/05_model_edges.py",
    "src/06_make_figures.py",
]


def main():
    for script in SCRIPTS:
        print("\n" + "=" * 80)
        print(f"Running {script}")
        print("=" * 80)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed.")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
