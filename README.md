## lilynorm

LilyPond normalization / tokenization pipeline used for the thesis dataset tooling.

### Prerequisites

* Python 3.10+
* [uv](https://github.com/astral-sh/uv) (optional, but recommended for quick virtual environments)

### Quick start with uv

```bash
# Install dependencies (editable so scripts can import lilynorm)
uv pip install -e .

# Run the dataset processor (adjust --input / output paths as needed)
uv run python -m scripts.process_dataset --input "data/raw/Dataset"
```

`uv run …` keeps the project isolated while still exposing the installed `lilynorm`
package to any script you execute.

### Running without uv

The scripts can also be executed directly because `scripts/process_dataset.py`
now falls back to adding the repository `src/` directory to `sys.path` when the
package is not installed:

```bash
python -m scripts.process_dataset --input "data/raw/Dataset"
```

If you prefer a traditional virtual environment, activate it and run
`pip install -e .` once so imports resolve everywhere.
