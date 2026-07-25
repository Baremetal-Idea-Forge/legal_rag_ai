# LLM Debugging - Execution Guide

## What Was Done

I've added comprehensive logging to your RAG pipeline to track down where the Article 21 inconsistency occurs. Here's what was instrumented:

### 📊 Logging Added To:

1. **[backend/services/rag_service.py](../backend/services/rag_service.py)**
   - Logs retrieved chunks (PDF name, chunk index, score)
   - Logs context size before sending to LLM
   - Logs message sizes sent to LLM
   - Logs LLM response length

2. **[backend/services/search_service.py](../backend/services/search_service.py)**
   - Logs query embedding dimensions
   - Logs all search results with scores
   - Logs vector distance and text match scores for top 5 hits

3. **[backend/services/prompts.py](../backend/services/prompts.py)**
   - Logs which chunks are selected
   - Logs why chunks are dropped (character limit)
   - Logs cumulative context size for each chunk

4. **[backend/clients/ollama_client.py](../backend/clients/ollama_client.py)**
   - Logs request parameters (model, temperature, message count/sizes)
   - Logs response length and first 100 chars
   - Logs streaming completion status

5. **[backend/clients/gemini_client.py](../backend/clients/gemini_client.py)**
   - Same logging as Ollama client for consistency

---

## 🧪 How to Run Tests

### Option 1: Run Test Harness (RECOMMENDED)

```bash
# From project root
cd /home/prateek/Prateek/Codes/legal_rag_ai

# Ensure backend server is running
# Then in another terminal:

python -m pytest tests/test_article21_consistency.py -v -s

# Or run standalone:
python tests/test_article21_consistency.py
```

**What it does:**
- Runs 20 identical queries in sequence
- Logs every step (retrieval, context selection, LLM request/response)
- Analyzes for variance in:
  - Retrieved chunks
  - Context size
  - Answer consistency
  - "Cannot find" refusal patterns
- Writes detailed logs to `article21_debug.log`

### Option 2: Test Retrieval Only (Isolate LLM)

The test harness includes a function to test if retrieval is consistent:

```python
# In test_article21_consistency.py
result = await run_retrieval_only_test(iterations=20)
```

This helps determine if variance is in retrieval or LLM.

### Option 3: Enable Debug Logging in Production

Set environment variable to enable DEBUG logs:

```bash
export LOG_LEVEL=DEBUG

# Then start your app
python -m backend.main
```

This will output debug logs to stderr:
```
[2026-07-25T10:00:01] [search_service] DEBUG: Hit 1: document.pdf (chunk 5, score=0.8543, ...)
[2026-07-25T10:00:01] [prompts] DEBUG: Selected chunk 0 (doc p.42, idx=5): 850 chars, cumulative=850/50000
```

---

## 🔍 What to Look For

### Good Signs (Consistent Behavior)
```
Retrieved 5 → selected 5 (all 20 runs identical)
Context size: 45,000 chars consistently
Same chunks: doc.pdf (p.42), doc.pdf (p.43), ...
Same LLM responses (if temperature=0)
```

### Bad Signs (Inconsistent Behavior)
```
Run 1: Retrieved 5 → selected 3
Run 2: Retrieved 5 → selected 5    ← Article 21 included only now!
Context size varies: 35k → 50k
Hit scores fluctuate: 0.85 → 0.72
LLM sometimes refuses despite context present
```

---

## 📋 Key Metrics to Check

| Metric | File | Line |
|--------|------|------|
| Retrieved hits count | search_service.py | `logger.info("Search...")` |
| Hit scores (vector distance) | search_service.py | `logger.debug("Hit %d:")` |
| Selected chunks (after cutoff) | rag_service.py | `logger.info("Retrieved %d → selected %d")` |
| Context size to LLM | rag_service.py | `logger.debug("Context size")` |
| Context cutoff threshold | prompts.py | `logger.debug("Context limit hit")` |
| LLM messages sent | rag_service.py | `logger.debug("Sending to LLM")` |
| LLM response | ollama_client.py | `logger.debug("Ollama response")` |

---

## 🚀 Debugging Workflow

### Step 1: Run the test
```bash
python tests/test_article21_consistency.py
```

### Step 2: Analyze the summary
The test outputs a FINAL VERDICT telling you where the issue is:

| Verdict | Next Action |
|---------|------------|
| "Retrieval is CONSISTENT" | → Go to Step 3 (LLM issue) |
| "Retrieval is INCONSISTENT" | → Go to Step 4 (Search/Embedding issue) |

### Step 3: If LLM is the culprit
Check:
1. Does the same context produce different answers?
2. Is temperature=0 deterministic?
3. Does the system prompt constrain the model correctly?

### Step 4: If Retrieval is the culprit
Check:
1. Do embedding vectors vary for same query?
2. Do Typesense scores fluctuate?
3. Is the character limit excluding Article 21?
4. Check `RAG_MAX_CONTEXT_CHARS` setting (backend/core/config.py)

---

## 📖 Log File Interpretation

After running the test, check `article21_debug.log`:

```
2026-07-25 10:00:15 [search_service] DEBUG: Query embedding: 768 dims, first 3: 0.123456, -0.234567, 0.345678
2026-07-25 10:00:15 [search_service] INFO: Search 'What does Article...' (hybrid) → 5 hits
2026-07-25 10:00:15 [search_service] DEBUG:   Hit 1: doc.pdf (chunk 42, score=0.8543, ...)
2026-07-25 10:00:15 [search_service] DEBUG:   Hit 2: doc.pdf (chunk 43, score=0.7821, ...)
2026-07-25 10:00:16 [prompts] DEBUG: Selected chunk 0 (doc p.42, idx=42): 1250 chars, cumulative=1250/50000
2026-07-25 10:00:16 [prompts] DEBUG: Selected chunk 1 (doc p.43, idx=43): 980 chars, cumulative=2230/50000
2026-07-25 10:00:16 [rag_service] DEBUG: Retrieved 5 → selected 2 (max_chars=50000) for query 'What does Article...'
2026-07-25 10:00:16 [rag_service] DEBUG: Context size: 2230 chars, 12 lines
2026-07-25 10:00:16 [ollama_client] DEBUG: Ollama chat request - model=gemma3:4b, temp=0.20, msg_count=2, msg_sizes=[2045, 2345]
2026-07-25 10:00:18 [ollama_client] DEBUG: Ollama response: 287 chars, first 100: "Article 21 provides..."
```

**Look for:**
- Are embedding vectors identical each run?
- Do hit scores change?
- Does the selected chunk count change?
- Is Article 21 consistently in the selected chunks?
- Does LLM receive different context each run?

---

## 🛠️ Fixes (Based on Findings)

### If: Retrieval scores inconsistent
**Fix:** Add deterministic seed to embedding model or Typesense

### If: Article 21 dropped due to character limit
**Fix:** Increase `RAG_MAX_CONTEXT_CHARS` in [backend/core/config.py](../backend/core/config.py)
```python
RAG_MAX_CONTEXT_CHARS: int = 50_000  # Try 100_000
```

### If: LLM inconsistent with same context
**Fix:** Set `temperature=0` for deterministic mode
```python
# In ollama_client.py chat() method
answer_text = await self._llm.chat(messages, temperature=0.0)
```

### If: Embedding variance
**Fix:** Check embedding model determinism:
```python
# Test in Python
from backend.models.embeddings_model import EmbeddingsModel
emb = EmbeddingsModel()
v1 = emb.embed_query("Article 21")
v2 = emb.embed_query("Article 21")
print(v1 == v2)  # Should be True
```

---

## ✅ Success Criteria

After debugging:
- [ ] Run 20 consecutive "Article 21" queries
- [ ] All return consistent answers
- [ ] All cite sources correctly
- [ ] No spurious "cannot find" responses
- [ ] Test logs show no variance in retrieval/context

---

## 📞 Need Help?

Check these files for more context:
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design
- [backend/core/config.py](../backend/core/config.py) - Settings (RAG_TOP_K, RAG_MAX_CONTEXT_CHARS)
- [backend/models/embeddings_model.py](../backend/models/embeddings_model.py) - Embedding details
- [backend/helpers/typesense_helper.py](../backend/helpers/typesense_helper.py) - Search config
