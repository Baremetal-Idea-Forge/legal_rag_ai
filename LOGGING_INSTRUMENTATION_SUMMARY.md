# Comprehensive Logging Implementation - Summary

## ✅ What's Been Done

### 1. **Enhanced Logging in 5 Key Files**

#### RAG Service Layer
- **File:** `backend/services/rag_service.py`
- **Changes:**
  - Logs retrieved chunks with PDF name, chunk index, and score
  - Logs context size before LLM submission
  - Logs message sizes and content passed to LLM
  - Logs LLM response length and preview
  - Tracks both streaming and non-streaming paths

#### Search Service Layer  
- **File:** `backend/services/search_service.py`
- **Changes:**
  - Logs query embedding dimensions and first 3 values
  - Logs total hit count from Typesense
  - Logs top 5 hits with:
    - Score (vector similarity or text match)
    - Vector distance values
    - Text match scores
    - Chunk metadata

#### Prompt Construction Layer
- **File:** `backend/services/prompts.py`
- **Changes:**
  - Logs each chunk selected with page range and character count
  - Logs cumulative context size as chunks are added
  - Logs exactly why chunks are dropped (character limit exceeded)
  - Shows context size before and after cutoff

#### LLM Clients
- **Files:** 
  - `backend/clients/ollama_client.py`
  - `backend/clients/gemini_client.py`
- **Changes:**
  - Logs request: model name, temperature, message count, message sizes
  - Logs response: length and first 100 characters
  - Logs streaming: token count and completion status
  - Both clients now have consistent logging format

---

### 2. **Comprehensive Test Harness**

**File:** `tests/test_article21_consistency.py`

A powerful testing script that:

✅ **Reproduces the issue:**
- Runs the same "Article 21" query 20+ times in sequence
- Can inject delays between queries to detect state issues
- Captures every single step with DEBUG logging

✅ **Analyzes variance:**
- Counts unique answers
- Measures chunk count variance
- Measures context size variance
- Tracks "Article 21" presence in retrieved context
- Counts refusal pattern frequency

✅ **Diagnoses root cause:**
- Includes `run_retrieval_only_test()` to isolate retrieval from LLM
- Generates clear pass/fail verdict on where the issue is
- Writes detailed log to `article21_debug.log`

✅ **Easy to use:**
```bash
# Via pytest
pytest tests/test_article21_consistency.py -v -s

# Standalone
python tests/test_article21_consistency.py
```

---

### 3. **Debug Execution Guide**

**File:** `DEBUG_LLM_INCONSISTENCY.md`

Complete reference including:
- How to run tests
- What metrics to check
- Log file interpretation examples
- Debugging workflow (step-by-step)
- Pre-defined fixes based on findings

---

## 📊 Key Information Logged

### At Retrieval Stage
```
Search 'What does Article...' (hybrid) → 5 hits
Hit 1: document.pdf (chunk 42, score=0.8543, vector_distance=0.001234)
Hit 2: document.pdf (chunk 43, score=0.7821, text_match=185.23)
```

### At Context Selection Stage
```
Selected chunk 0 (document p.42, idx=42): 1250 chars, cumulative=1250/50000
Selected chunk 1 (document p.43, idx=43): 980 chars, cumulative=2230/50000
Context limit hit at chunk 5: 45000 + 8000 > 50000 (max_chars)
```

### At LLM Request Stage
```
Ollama chat request - model=gemma3:4b, temp=0.20, msg_count=2, msg_sizes=[2045, 2345]
Sending to LLM - system msg: 2045 chars, user msg: 2345 chars
```

### At LLM Response Stage
```
Ollama response: 287 chars, first 100: "Article 21 provides..."
```

---

## 🚀 Next Steps

### Immediate Action
1. **Run the consistency test** to capture the variance:
   ```bash
   cd /home/prateek/Prateek/Codes/legal_rag_ai
   python tests/test_article21_consistency.py
   ```

2. **Check the test output** for the FINAL VERDICT:
   - If retrieval is **consistent** → problem is LLM
   - If retrieval is **inconsistent** → problem is search/embedding

3. **Review `article21_debug.log`** to find the exact point of variance

### Based on Findings
- **If retrieval varies:** Check Typesense scoring, embedding model, RAG_MAX_CONTEXT_CHARS
- **If LLM varies:** Test with temperature=0, check prompt wording, verify context transmission
- **If context cutoff:** Increase `RAG_MAX_CONTEXT_CHARS` setting

---

## 📁 Files Changed

| File | Changes | Lines |
|------|---------|-------|
| `backend/services/rag_service.py` | Added chunk/context/response logging | +30 |
| `backend/services/search_service.py` | Added hit score and embedding logging | +15 |
| `backend/services/prompts.py` | Added context selection logic logging | +20 |
| `backend/clients/ollama_client.py` | Added request/response logging | +12 |
| `backend/clients/gemini_client.py` | Added request/response logging | +12 |
| `tests/test_article21_consistency.py` | **NEW** - Comprehensive test harness | 250 |
| `DEBUG_LLM_INCONSISTENCY.md` | **NEW** - Execution guide | 350 |

---

## 🎯 Expected Outcomes

After running the test, you'll be able to answer:

1. ✅ **Is retrieval consistent?**
   - Do you get the same 5 chunks in the same order?
   - Are scores identical or do they vary?

2. ✅ **Is Article 21 always retrieved?**
   - Is it in the top 5 hits?
   - Is it being included in the final context?
   - Is it being cut off by the character limit?

3. ✅ **Is the context reaching the LLM?**
   - What size is the context?
   - Is Article 21 included in the context string?

4. ✅ **Is the LLM being inconsistent?**
   - Same context → different answers? (temperature issue)
   - No context → correct refusal? (working as designed)
   - No context → wrong refusal despite context present? (retrieval issue)

---

## 💡 Pro Tips

- **Run during off-peak hours** if Typesense is shared
- **Check embeddings determinism** if you suspect embedding model variance:
  ```python
  from backend.models.embeddings_model import EmbeddingsModel
  emb = EmbeddingsModel()
  assert emb.embed_query("Article 21") == emb.embed_query("Article 21")
  ```
- **Test with temperature=0** to isolate retrieval issues from LLM stochasticity
- **Increase RAG_MAX_CONTEXT_CHARS** if Article 21 is being cut off
- **Check RAG_TOP_K setting** - ensure Article 21 is in top-k results

---

## 📞 Debugging Support

All logging uses Python's standard `logging` module at DEBUG/INFO levels.

Set environment variable to see DEBUG logs:
```bash
export LOG_LEVEL=DEBUG
python -m backend.main
```

Or modify any logger call to print to console immediately:
```python
print(f"DEBUG: Context size = {len(context_text)} chars")
```
