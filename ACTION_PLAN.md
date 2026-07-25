# Legal RAG System — Action Plan

Update it as API V 0.0.2
## 📋 Debugging Plan: LLM Unpredictability in RAG System

I've analyzed your project and created a comprehensive debugging plan to identify why your LLM gives inconsistent responses. Here's what I found:

---

### **Root Cause Hypotheses** (ordered by likelihood)

| **Hypothesis** | **Impact** | **Likelihood** |
|---|---|---|
| **Retrieval inconsistency** | Different contexts retrieved on same query | 🔴 High |
| **Context cutoff** | Article 21 excluded when exceeding `RAG_MAX_CONTEXT_CHARS` | 🟠 Medium-High |
| **LLM temperature variance** | Same context produces different responses | 🟠 Medium |
| **Embedding variance** | Query embeddings differ slightly each call | 🟡 Medium |
| **Typesense ranking** | Hybrid search returns different result order | 🟡 Medium |
| **Prompt ambiguity** | LLM interprets instruction differently with some contexts | 🟡 Low-Medium |

---

### **Phase 1: Instrumentation & Logging**
Add detailed logging to pinpoint where inconsistency occurs:

**Files to instrument:**

1. **rag_service.py** — Retrieval stage
   - Log retrieved hits (IDs, scores, content)
   - Log context size before/after cutoff
   - Add request correlation IDs

2. **search_service.py** — Search scoring
   - Log embedding used
   - Log raw Typesense results
   - Log final scores

3. **prompts.py** — Context selection
   - Log which chunks selected by `select_hits_within()`
   - Log why chunks dropped (character limit)
   - Log final context passed to LLM

4. **ollama_client.py** or Gemini client
   - Log exact messages sent to LLM
   - Log response received
   - Log model/temperature used

---

### **Phase 2: Reproduce & Isolate**
Create a test harness to surface the bug:

```python
# test_article21_consistency.py
import asyncio
from backend.core.dependencies import get_rag_service

async def test_article_21_consistency():
    """Run 20 identical queries and check response consistency."""
    service = get_rag_service()
    query = "What does Article 21 say?"
    
    responses = []
    retrieved_contexts = []
    
    for i in range(20):
        response = await service.answer(query)
        responses.append(response.answer)
        retrieved_contexts.append(response.chunks_used)
        print(f"Run {i+1}: {response.answer[:80]}...")
        await asyncio.sleep(0.1)  # Small delay
    
    # Analyze variance
    unique_answers = set(responses)
    unique_contexts = [len(c) for c in retrieved_contexts]
    
    print(f"\nVariance: {len(unique_answers)} unique responses in 20 runs")
    print(f"Context sizes: {set(unique_contexts)}")
    
    for answer in unique_answers:
        count = responses.count(answer)
        print(f"  [{count}x] {answer[:100]}...")
```

---

### **Phase 3: Targeted Investigations**

#### **Check 1: Retrieval Consistency**
```bash
# Run same query multiple times and compare retrieved chunks
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Article 21", "top_k": 5}'
# → Do you get same 5 results in same order each time?
```

**Debug questions:**
- Are the same chunks retrieved?
- Do scores change?
- Is the order different?

#### **Check 2: Context Cutoff Impact**
Look at prompts.py:
```python
def select_hits_within(hits: list[ChunkHit], *, max_chars: int) -> list[ChunkHit]:
    # If Article 21 is in hit #2 but hits #1 eats up max_chars, 
    # Article 21 never reaches the LLM
```

**Debug steps:**
1. Log the character count of each hit
2. Log cumulative size before each hit is added
3. Check if Article 21 chunk is included or dropped
4. Try increasing `RAG_MAX_CONTEXT_CHARS` and test again

#### **Check 3: LLM Temperature Variance**
Test if variance is from LLM itself:
```python
# Set temperature to 0 (deterministic)
# Run same context 5x through LLM
# Check if response is identical
```

**Debug steps:**
1. Modify LLM client to accept `temperature` parameter in config
2. Test with `temperature: 0` (deterministic)
3. Test with `temperature: 0.2` (current)
4. Compare variance

#### **Check 4: Embedding Determinism**
```python
# Check if same query produces same embedding twice
from backend.models.embeddings_model import EmbeddingsModel
embedding_model = EmbeddingsModel()
emb1 = embedding_model.embed_query("Article 21")
emb2 = embedding_model.embed_query("Article 21")
assert emb1 == emb2, f"Embeddings differ: {emb1[:3]} vs {emb2[:3]}"
```

---

### **Phase 4: Quick Wins & Fixes**

If you find the issue, here are targeted fixes:

| **Finding** | **Fix** |
|---|---|
| Article 21 dropped by context cutoff | Increase `RAG_MAX_CONTEXT_CHARS` or use hit-aware cutoff |
| Retrieval scores vary | Add deterministic seed to Typesense or re-rank locally |
| Different chunks retrieved | Log embedding distance; check if close matches exist |
| Temperature causing variance | Set `temperature: 0` for deterministic answers (less creative but consistent) |
| Prompt too strict | Add fallback rule: if context retrieved but LLM says "cannot answer", force attempt |

---

### **Phase 5: Advanced Diagnostics**

If simpler checks don't reveal the issue:

1. **Add cache** — Store (query → context → response) to detect if same context → different answer
2. **Duplicate queries** — Run same query 10s in quick succession (test threadpool race conditions)
3. **Model swapping** — Test with Gemini vs Ollama to isolate LLM from retrieval
4. **Chunk inspection** — Verify Article 21 exists in all PDF chunks; check for duplicates
5. **Typesense index health** — Check if index is corrupt/incomplete

