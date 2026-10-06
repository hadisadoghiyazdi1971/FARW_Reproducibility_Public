# Environment for the minimal public release

The completed manuscript experiment manifests recorded Python 3.14.3, NumPy 2.5.0, SciPy 1.18.0, and scikit-learn 1.9.0 on Windows. `requirements.txt` retains those recorded package versions. The retained figure scripts also require Matplotlib; its publication-run version was not established, so it is intentionally unpinned.

The public repository contains selection implementations and figure generation from compact CSVs. It does not contain the raw signals or full descriptor matrices needed to rerun the manuscript experiments. On a supported Python environment, install `requirements.txt` and run the two figure commands in `README.md`. The public index verifier needs NumPy and only the files included in `config/`.
