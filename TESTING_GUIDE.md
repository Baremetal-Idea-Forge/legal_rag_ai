# Testing Guide

Comprehensive testing strategy and how-to for the Legal RAG AI system.

## Test Structure

```
tests/
├── conftest.py                      # Pytest fixtures, shared setup
├── test_api.py                      # Router tests (FastAPI integration)
├── test_rag_service.py              # RAG orchestration tests
├── test_search_service.py           # Search logic tests
├── test_embedding_service.py        # Embedding service tests
├── test_ingestion_service.py        # PDF ingestion tests
├── test_*_repository.py             # Data access tests
├── test_*_client.py                 # External client tests
├── test_logging.py                  # Logging instrumentation tests
├── test_security.py                 # Auth & rate limiting tests
├── eval/
│   ├── golden_set.json              # Test queries with expected citations
│   ├── metrics.py                   # Evaluation metrics
│   ├── run_eval.py                  # Evaluation runner
│   └── test_metrics.py              # Metrics tests
└── data/
    └── sample-documents/            # Test PDFs
```

---

## Quick Start

### Run All Tests

```bash
pytest
```

**Output**:
```
tests/test_api.py::test_health PASSED                          [ 1%]
tests/test_api.py::test_search PASSED                          [ 2%]
...
1042 passed in 23.45s
```

### Run Specific Tests

```bash
# Single test file
pytest tests/test_api.py

# Single test function
pytest tests/test_api.py::test_search

# Tests matching a pattern
pytest -k "search"  # All tests with "search" in name

# Only failing tests from last run
pytest --lf

# Verbose mode
pytest -v
```

### Test with Coverage

```bash
# Generate coverage report
pytest --cov=backend --cov-report=html

# View report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

**Coverage targets**:
- Core logic (services, repositories): ≥ 90%
- Routes (HTTP layer): ≥ 80%
- Helpers, utilities: ≥ 70%

---

## Test Categories

### 1. Unit Tests

**What**: Test individual functions/classes in isolation.

**Tools**: `pytest`, fakes, mocks.

**Examples**:

```python
# test_search_service.py
def test_search_vector_mode(embedding_service, typesense_repo):
    """Test vector search returns results ranked by distance."""
    service = SearchService(
        embedding_service=embedding_service,
        typesense_repo=typesense_repo,
    )
    response = service.search(
        "Article 21",
        mode="vector",
        top_k=5,
    )
    assert len(response.hits) == 5
    assert response.hits[0].score > response.hits[1].score
```

**Run unit tests**:

```bash
pytest tests/ -k "not integration"
```

---

### 2. Integration Tests

**What**: Test interactions between components (services + repos + clients).

**Tools**: `pytest`, TestClient, real or mocked Typesense.

**Examples**:

```python
# test_api.py
def test_ingest_and_search(client, typesense_repo):
    """Upload PDF, then search it."""
    # 1. Upload PDF
    with open("tests/data/sample.pdf", "rb") as f:
        resp = client.post("/ingest", files={"file": f})
    assert resp.status_code == 201
    pdf_id = resp.json()["pdf_id"]

    # 2. Search the indexed content
    resp = client.post("/search", json={
        "query": "Article 21",
        "top_k": 5,
    })
    assert resp.status_code == 200
    assert len(resp.json()["hits"]) > 0
```

**Run integration tests**:

```bash
pytest tests/test_api.py -v
```

---

### 3. End-to-End Tests

**What**: Test complete workflows (upload → search → chat).

**Tools**: `pytest`, TestClient, real services.

**Example**:

```python
@pytest.mark.integration
async def test_full_rag_workflow():
    """Complete flow: ingest → search → chat."""
    # Upload
    ingest_resp = await ingest_pdf("constitution.pdf")
    pdf_id = ingest_resp.pdf_id

    # Search
    search_resp = await search_service.search("Article 21")
    assert search_resp.count > 0

    # Chat
    chat_resp = await rag_service.answer("What is Article 21?")
    assert chat_resp.answer
    assert len(chat_resp.citations) > 0
```

**Run E2E tests**:

```bash
pytest tests/ -m "integration" -v
```

---

### 4. Performance Tests

**What**: Measure latency, throughput, resource usage.

**Tools**: `pytest-benchmark`, `time`, profilers.

**Example**:

```python
def test_search_latency(benchmark, search_service):
    """Search should complete within 500ms."""
    def search():
        search_service.search("Article 21", top_k=5)

    result = benchmark(search)
    assert result.stats.mean < 0.5  # 500ms
```

**Run performance tests**:

```bash
pytest tests/test_performance.py -v
```

---

### 5. Evaluation Tests

**What**: Measure RAG quality against golden set (ground truth).

**Golden Set Format** (`tests/eval/golden_set.json`):

```json
[
  {
    "query": "What are the rights under Article 21?",
    "expected_citations": ["constitution.pdf"],
    "expected_pages": [15, 16],
    "expected_chunks": [42, 43]
  }
]
```

**Evaluation Metrics**:

| Metric | Formula | Target |
|--------|---------|--------|
| **Retrieval Recall@5** | (citations found in top-5) / (total expected) | ≥ 95% |
| **Citation Precision** | (correct citations) / (total citations) | ≥ 90% |
| **Answer Relevance** | (LLM answer covers expected info) | ≥ 85% |

**Run evaluation**:

```bash
python tests/eval/run_eval.py
```

**Output**:

```
=== Evaluation Results ===
Recall@5: 96.2%
Precision: 92.1%
Answer Relevance: 88.5%
```

---

## Writing Tests

### Setup (conftest.py)

Shared fixtures for all tests:

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
import main

@pytest.fixture
def client():
    """HTTP test client (no lifespan)."""
    return TestClient(main.app, raise_server_exceptions=False)

@pytest.fixture
def fake_embedding_service():
    """Mock embedding service."""
    class FakeEmbedding:
        def embed_query(self, query: str):
            return [0.1] * 1024  # Mock 1024-dim embedding
    return FakeEmbedding()

@pytest.fixture
def fake_typesense_repo():
    """Mock Typesense repo."""
    class FakeRepo:
        def search(self, query, **kwargs):
            return {
                "hits": [
                    {
                        "document": {
                            "id": "chunk-1",
                            "pdf_name": "test.pdf",
                            "content": "Article 21..."
                        }
                    }
                ]
            }
    return FakeRepo()
```

### Writing a Test

```python
import pytest
from models.schemas import SearchResponse

def test_search_returns_chunks(client, fake_typesense_repo):
    """POST /search returns matching chunks."""
    # Arrange: Set up dependencies
    main.app.dependency_overrides[get_typesense_repository] = \
        lambda: fake_typesense_repo

    # Act: Call the endpoint
    response = client.post("/search", json={
        "query": "Article 21",
        "top_k": 5,
    })

    # Assert: Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "hits" in data
    assert len(data["hits"]) > 0

    # Cleanup
    main.app.dependency_overrides.clear()
```

### Test Conventions

1. **Naming**: `test_<subject>_<behavior>` (e.g., `test_search_returns_chunks`)
2. **Structure**: Arrange → Act → Assert (AAA pattern)
3. **Focus**: Test one thing per test
4. **Isolation**: Use fixtures to reset state between tests
5. **Clarity**: Comments explain non-obvious logic

---

## Edge Cases to Test

### Upload & Ingestion

```python
@pytest.mark.parametrize("file_size", [
    1,                           # Tiny file
    MAX_UPLOAD_BYTES - 1,        # Just under limit
    MAX_UPLOAD_BYTES,            # Exactly at limit
    MAX_UPLOAD_BYTES + 1,        # Just over limit
])
def test_upload_size_boundaries(client, file_size):
    """Upload size validation."""
    pdf_bytes = b"fake pdf" * (file_size // 8)
    # ... test upload ...
```

### Concurrent Requests

```python
@pytest.mark.asyncio
async def test_concurrent_uploads():
    """Multiple uploads simultaneously."""
    tasks = [
        upload_pdf("doc1.pdf"),
        upload_pdf("doc2.pdf"),
        upload_pdf("doc3.pdf"),
    ]
    results = await asyncio.gather(*tasks)
    assert all(r.status == "indexed" for r in results)
```

### Error Scenarios

```python
def test_search_with_typesense_down():
    """Search fails gracefully when Typesense is down."""
    # Mock Typesense unavailable
    with patch('typesense.Client') as mock:
        mock.side_effect = ConnectionError("Connection refused")
        
        with pytest.raises(SearchError):
            search_service.search("Article 21")
```

### Large Datasets

```python
@pytest.fixture
def large_corpus():
    """Generate 10,000 chunks for stress testing."""
    chunks = [
        ChunkHit(
            id=f"chunk-{i}",
            content=f"Article {i}: ..." * 50,
            # ...
        )
        for i in range(10000)
    ]
    return chunks

def test_search_with_large_corpus(search_service, large_corpus):
    """Search should still be fast with many chunks."""
    start = time.time()
    result = search_service.search("Article 1", top_k=10)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete in < 1 sec
    assert result.count <= 10
```

---

## Mocking External Services

### Mock LLM Client

```python
@pytest.fixture
def mock_gemini_client():
    """Mock Gemini API responses."""
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value="Article 21 says...")
    return mock

def test_chat_uses_llm(mock_gemini_client):
    service = RagService(
        search_service=...,
        llm_client=mock_gemini_client,
        ...
    )
    response = await service.answer("What is Article 21?")
    
    # Verify LLM was called
    mock_gemini_client.chat.assert_called_once()
    assert "Article 21" in response.answer
```

### Mock Typesense

```python
@pytest.fixture
def mock_typesense():
    """Mock Typesense search responses."""
    mock = Mock()
    mock.search = Mock(return_value={
        "hits": [
            {
                "document": {
                    "id": "1",
                    "content": "Article 21...",
                    "pdf_name": "constitution.pdf"
                },
                "vector_distance": 0.1
            }
        ]
    })
    return mock

def test_search_queries_typesense(mock_typesense):
    repo = TypesenseRepository(client=mock_typesense)
    results = repo.search("Article 21", vector_query="...")
    
    mock_typesense.search.assert_called_once()
    assert len(results["hits"]) == 1
```

---

## Debugging Tests

### Run with Output

```bash
# Show print() statements
pytest -s tests/test_api.py

# Show setup/teardown info
pytest -v tests/test_api.py
```

### Interactive Debugger

```bash
# Drop into pdb on failure
pytest --pdb tests/test_api.py

# Drop into pdb immediately
pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb tests/test_api.py
```

### Add Breakpoints

```python
def test_search():
    # ... code ...
    import pdb; pdb.set_trace()  # Debugger pauses here
    # ... more code ...
```

### Capture Logs

```bash
# Show log output during tests
pytest --log-cli-level=DEBUG tests/test_api.py
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      typesense:
        image: typesense/typesense:latest
        options: >-
          --health-cmd "curl -f http://localhost:8108/health || exit 1"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 8108:8108

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev

      - name: Run tests
        run: |
          pytest --cov=backend --cov-report=term

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

## Performance Testing

### Benchmark Searches

```python
import pytest
import time

def test_search_performance():
    """Ensure search completes within latency budget."""
    service = SearchService(...)
    
    times = []
    for _ in range(100):
        start = time.time()
        service.search("Article 21", top_k=5)
        times.append(time.time() - start)
    
    avg = sum(times) / len(times)
    p95 = sorted(times)[95]
    
    print(f"\nSearch latency: avg={avg*1000:.1f}ms, p95={p95*1000:.1f}ms")
    assert p95 < 0.5  # p95 < 500ms
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory-profiler

# Profile a function
python -m memory_profiler backend/services/embedding_service.py
```

---

## Test Maintenance

### Keep Tests Fresh

- Update tests when APIs change
- Remove tests for deprecated features
- Add tests for new bugs (regression tests)
- Refactor test code for clarity

### Run Tests Locally Before Pushing

```bash
# Full suite
pytest

# Specific area
pytest tests/test_rag_service.py

# With coverage
pytest --cov=backend tests/
```

---

## Additional Resources

- [Pytest Docs](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/advanced/testing-dependencies/)
- [unittest.mock Docs](https://docs.python.org/3/library/unittest.mock.html)
- [Async Testing with pytest-asyncio](https://pytest-asyncio.readthedocs.io/)

