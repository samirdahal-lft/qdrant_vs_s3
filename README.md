# Qdrant vs S3 Vectors Benchmark Suite

A comprehensive Python benchmarking suite that compares **Qdrant** (self-hosted vector database) with **Amazon S3 Vectors** (AWS managed embedding search) across 17 different search scenarios and features.

## 📋 Overview

This project provides hands-on comparison of two modern vector search platforms through a series of isolated test scenarios covering:

- **Semantic Search** — Pure vector similarity without filters
- **Filtering** — Exact match, numeric ranges, Boolean logic (AND/OR/NOT), set membership
- **Retrieval** — ID-based lookups  
- **Mutations** — Metadata updates and deletion
- **Advanced Features** — Hybrid search, recommendations, full-text, geospatial, grouping, named vectors
- **Boundary Testing** — TopK limits and architectural constraints

Each test is self-contained, uses a clean abstract base class (ABC) pattern, and captures timing + state for fair comparison.

## ✨ Key Features

- ✅ **17 independent tests** (test_01 through test_18, no test_08)
- ✅ **Auto-discovery & CSV reporting** – `run_all.py` discovers tests via glob, runs them, and writes timestamped CSV
- ✅ **NumPy-style docstrings** – Comprehensive parameter/return/attribute documentation
- ✅ **ABC-based architecture** – Consistent interface across Qdrant and S3 Vectors implementations
- ✅ **Null Object pattern** – Unsupported features gracefully return `is_supported=False`
- ✅ **Cloud-agnostic** – Easily switch between self-hosted Qdrant and AWS S3 Vectors
- ✅ **Progress tracking** – Real-time progress bar during test execution

## 🏗️ Project Structure

```
qdrant_vs_s3/
├── core/
│   ├── __init__.py
│   ├── clients.py              # Qdrant & S3 Vectors SDK initialization
│   ├── config.py               # Shared configuration & constants
│   ├── dataset.py              # Movie dataset for benchmarking
│   └── embeddings.py           # Embedding generation & caching
├── tests/
│   ├── test_01_semantic_search.py
│   ├── test_02_filter_exact_match.py
│   ├── test_03_filter_numeric_range.py
│   ├── test_04_filter_combined_and.py
│   ├── test_05_filter_or.py
│   ├── test_06_filter_negation.py
│   ├── test_07_filter_set_membership.py
│   ├── test_09_get_by_id.py
│   ├── test_10_update_metadata.py
│   ├── test_11_delete.py
│   ├── test_12_hybrid_search.py
│   ├── test_13_recommendation.py
│   ├── test_14_fulltext_match.py
│   ├── test_15_geo_filter.py
│   ├── test_16_grouping.py
│   ├── test_17_named_vectors.py
│   └── test_18_topk_limit.py
├── results/                    # Generated CSV reports (auto-created)
│   └── benchmark_20260226_120000.csv
├── run_all.py                  # Master test runner with CSV export
├── setup.py                    # Package metadata
├── pyproject.toml              # uv dependency configuration
├── .env.example                # Environment template
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11+** (tested with 3.11.13)
- **Qdrant running locally** (default: `http://localhost:6333`)
- **AWS credentials** configured (for S3 Vectors access)
- **uv** or **pip** for dependency management

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/qdrant_vs_s3.git
cd qdrant_vs_s3

# Create & activate virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
# OR with pip
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Then edit `.env` with your settings:

```bash
QDRANT_URL=http://localhost:6333
AWS_REGION=us-east-1
S3V_BUCKET_NAME=movie-search-comparison
S3V_INDEX_NAME=movies
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 4. Run All Tests

```bash
# Execute all 17 tests and generate CSV report
python run_all.py
```

Output:
```
Discovered 17 test(s).

[1/17] Running test_01_semantic_search ... ✓  (2.3s)
[2/17] Running test_02_filter_exact_match ... ✓  (1.8s)
...
[17/17] Running test_18_topk_limit ... ✓  (0.5s)

============================================================
  DONE — 17/17 passed, 0 failed  (28.4s)
  CSV  → results/benchmark_20260226_120000.csv
============================================================
```

### 5. Run Individual Tests

```bash
# Run a single test directly
python -m tests.test_01_semantic_search

# or with Python:
from tests.test_01_semantic_search import run
run()
```

## 📊 CSV Export Format

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

**Example row:**
```csv
tests.test_01_semantic_search,Test 01 Semantic Search,PASS,2.341,"=== ... ===","",2026-02-26T12:00:00
```

## 🧪 Test Categories

### Semantic Search (01)

Pure vector similarity without filtering. Runs 3 queries and compares result quality.

- **01**: Semantic search (no filters)

### Filtering (02–07)

Metadata filters using Qdrant's `Filter` + `FieldCondition` and S3 Vectors' JSON operators.

- **02**: Exact match (`genre = "Sci-Fi"`)
- **03**: Numeric range (`year >= 2010`)
- **04**: Combined AND (`Sci-Fi AND year>=2010 AND rating>7.5`)
- **05**: OR logic (`genre IN [Drama, Comedy]`)
- **06**: Negation (`language != "English"`)
- **07**: Set membership (`genre IN [Action, Thriller]`)

### CRUD Operations (09–11)

Retrieval, updates, and deletion.

- **09**: Get by ID
- **10**: Update metadata (compare partial vs full re-put)
- **11**: Delete vectors (verify count drop)

### Advanced Features (12–18)

Features supported by only one platform or with special handling.

- **12**: Hybrid search (dense + BM25 sparse, RRF fusion) — **Qdrant only**
- **13**: Recommendation API (positive/negative examples) — **Qdrant only**
- **14**: Full-text match (keyword in description) — **Qdrant only**
- **15**: Geo filtering (radius search) — **Qdrant only**
- **16**: Grouping / Diversity (1 result per genre) — **Qdrant only**
- **17**: Named vectors (search by title vs description) — **Qdrant only**
- **18**: TopK boundary analysis (100-result ceiling in S3 Vectors)

## 🏛️ Architecture

### ABC Pattern

Each test inherits from an abstract base class (ABC) with a common interface:

```python
class VectorBenchmark(ABC):
    @abstractmethod
    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        pass
    
    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000

class QdrantEngine(VectorBenchmark):
    def search(self, vector, limit):
        # Qdrant-specific implementation
        pass

class S3VectorEngine(VectorBenchmark):
    def search(self, vector, limit):
        # S3 Vectors-specific implementation
        pass
```

### Dataclasses for Results

Immutable `@dataclass(frozen=True)` records capture results:

```python
@dataclass(frozen=True)
class SearchResult:
    title: str
    score: float
    platform: str
    latency_ms: float
    metadata: Dict[str, Any]
```

### Null Object Pattern

Unsupported features (S3 Vectors lacks hybrid search, recommendations, etc.) gracefully return:

```python
class S3HybridEngine(HybridSearchBenchmark):
    def search_hybrid(self, query_text: str, limit: int) -> List[HybridSearchResult]:
        return [HybridSearchResult("N/A", 0.0, "S3 Vectors", 0.0, is_supported=False)]
```

## 🔧 Dependencies

See `pyproject.toml` or `requirements.txt`. Key packages:

- **qdrant-client** – Qdrant Python SDK
- **boto3** – AWS SDK for S3 Vectors
- **sentence-transformers** – Embedding model
- **fastembed** – Sparse embeddings (BM25)
- **rich** – Console output formatting
- **python-dotenv** – Environment variable loading

## 📈 Key Findings

| Feature | Qdrant | S3 Vectors |
|---------|--------|-----------|
| Semantic Search | ✅ | ✅ |
| Filtering | ✅ Full | ⚠️ Exact/Range only |
| Hybrid Search | ✅ Dense + Sparse (RRF) | ❌ |
| Recommendations | ✅ Vector-based | ❌ |
| Full-Text Search | ✅ With text index | ❌ |
| Geospatial | ✅ geo_radius filter | ❌ |
| Grouping | ✅ query_points_groups | ❌ |
| Named Vectors | ✅ Multi-vector per point | ❌ |
| TopK Ceiling | None | 100 hard limit |
| Update Strategy | Partial (set_payload) | Full re-put |



