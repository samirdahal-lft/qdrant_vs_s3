# Qdrant vs S3 Vectors Benchmark Suite

A comprehensive Python benchmarking suite that compares **Qdrant** (self-hosted vector database) with **Amazon S3 Vectors** (AWS managed) across 17 different search scenarios and features.

## Overview

This project provides hands-on comparison of two modern vector search platforms through a series of isolated test scenarios covering:

- **Semantic Search** - Pure vector similarity without filters
- **Filtering** - Exact match, numeric ranges, Boolean logic (AND/OR/NOT), set membership
- **Retrieval** - ID-based lookups  
- **Mutations** - Metadata updates and deletion
- **Advanced Features** - Hybrid search, recommendations, full-text, geospatial, grouping, named vectors
- **Boundary Testing** - TopK limits and architectural constraints

Each test is self-contained, uses a clean abstract base class (ABC) pattern, and captures timing + state for fair comparison.

## Key Features

- **17 independent tests** (test_01 through test_18, no test_08)
-   **Auto-discovery & CSV reporting** – `run_all.py` discovers tests via glob, runs them, and writes timestamped CSV
-   **NumPy-style docstrings** – Comprehensive parameter/return/attribute documentation
-   **ABC-based architecture** – Consistent interface across Qdrant and S3 Vectors implementations
-  **Null Object pattern** – Unsupported features gracefully return `is_supported=False`


##  Quick Start

### 1. Prerequisites

- **Python 3.11+** (tested with 3.11.13)
- **Qdrant running locally** (default: `http://localhost:6333`)
- **AWS credentials** configured (for S3 Vectors access)
- **uv** or **pip** for dependency management

## 2. Run Qdrant locally
```bash
# Pull the latest Qdrant image
docker pull qdrant/qdrant

# Run Qdrant container
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
  ```

### 3. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/qdrant_vs_s3.git
cd qdrant_vs_s3

# Create & activate virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv sync
```

### 4. Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Then edit `.env` with your settings:

```bash
AWS_ACCESS_KEY_ID= <your_aws_access_key_id>
AWS_SECRET_ACCESS_KEY= <your_aws_secret_access_key>
AWS_SESSION_TOKEN = <your_aws_session_token>
```

### 5. Run All Tests

```bash
python run_all.py
```

Output:
```
Discovered 17 test(s).

[1/17] Running test_01_semantic_search ... ✓  (2.3s)
[2/17] Running test_02_filter_exact_match ... ✓  (1.8s)
...
[17/17] Running test_18_topk_limit ... ✓  (0.5s)

  DONE — 17/17 passed, 0 failed  (28.4s)
  CSV  → results/benchmark_20260226_120000.csv
```

### 5. Run Individual Tests

```bash
# Run a single test directly
python -m tests.test_01_semantic_search
```

##  CSV Export Format

The `run_all.py` script generates a **timestamped CSV** under `results/` with columns:

| Column | Type | Description |
|--------|------|-------------|
| `test_module` | str | Dotted module name (e.g., `tests.test_01_semantic_search`) |
| `test_label` | str | Human-readable test name |
| `status` | str | `PASS` or `FAIL` |
| `duration_s` | float | Wall-clock execution time |
| `output` | str | Captured stdout (console output) |
| `error` | str | Exception message (if `FAIL`) |
| `timestamp` | str | ISO 8601 execution timestamp |





##  Key Findings

| Feature | Qdrant | S3 Vectors |
|---------|--------|-----------|
| Semantic Search | yes | yes |
| Filtering |  Full |  Exact/Range only |
| Hybrid Search | Dense + Sparse (RRF) | No |
| Recommendations | yes, Vector-based | No |
| Full-Text Search | yes, With text index | No |
| Geospatial | yes, geo_radius filter | No |
| Grouping | yes, query_points_groups | No |
| Named Vectors | yes, Multi-vector per point | No |
| TopK Ceiling | None | 100 hard limit |
| Update Strategy | Partial (set_payload) | Full re-put |



