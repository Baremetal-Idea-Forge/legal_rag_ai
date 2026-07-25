# ✅ Comprehensive LLM Debugging Plan - Implementation Complete

## 🎯 Mission: Find Why Article 21 Queries Are Inconsistent

### Problem
When asking "Article 21", LLM sometimes returns:
- ❌ "I cannot answer" (while citing sources)
- ✅ Correct answer with citations
- (Unpredictable behavior)

---

## 🛠️ Solution Implemented

### Phase 1: Instrumentation ✅ DONE
Add comprehensive logging to track each stage:

```
Query → [Search] → [Context Selection] → [LLM] → Answer
         ↓ log hits    ↓ log dropped      ↓ log    ↓ log response
         & scores      chunks            msgs
```

| Component | Logging Added | File |
|-----------|---------------|------|
| **Search** | Hit scores, embedding vectors, vector distances | search_service.py |
| **Context** | Selected chunks, dropped chunks, why dropped, cumulative size | prompts.py |
| **Retrieval** | Total retrieved, final selected, chunk metadata | rag_service.py |
| **LLM** | Model, temp, message sizes, response length | ollama_client.py, gemini_client.py |

### Phase 2: Test Harness ✅ DONE
Reproduce the issue with automated testing:

```python
# Run 20 identical queries → analyze variance
python tests/test_article21_consistency.py

# Output → FINAL VERDICT:
# "Retrieval is CONSISTENT" → Problem is LLM
# "Retrieval is INCONSISTENT" → Problem is search/embedding
```

### Phase 3: Documentation ✅ DONE
- **DEBUG_LLM_INCONSISTENCY.md** - Step-by-step debugging guide
- **LOGGING_INSTRUMENTATION_SUMMARY.md** - Implementation details
- **rag_service.py, search_service.py, prompts.py** - Inline debug comments

---

## 📊 What You Get

### Visibility Into Each Stage

#### 1️⃣ SEARCH RESULTS
```
Search 'What does Article 21 say?' (hybrid) → 5 hits
  Hit 1: document.pdf (chunk 42, score=0.8543, vector_distance=0.001234)
  Hit 2: document.pdf (chunk 43, score=0.7821, text_match=185.23)
  Hit 3: document.pdf (chunk 50, score=0.6432, ...)
  ...
```

#### 2️⃣ CONTEXT SELECTION
```
Selected chunk 0 (document p.42, idx=42): 1250 chars, cumulative=1250/50000 ✓
Selected chunk 1 (document p.43, idx=43): 980 chars, cumulative=2230/50000 ✓
Retrieved 5 → selected 2 (dropped 3 due to char limit)
```

#### 3️⃣ LLM REQUEST
```
Sending to LLM - system msg: 2045 chars, user msg: 2345 chars
Model: gemma3:4b | Temp: 0.20
```

#### 4️⃣ LLM RESPONSE
```
Ollama response: 287 chars, first 100: "Article 21 provides..."
```

---

## 🚀 Quick Start

### Run the Test
```bash
cd /home/prateek/Prateek/Codes/legal_rag_ai
python tests/test_article21_consistency.py
```

### Check Results
```bash
# Logs go to article21_debug.log
tail -f article21_debug.log | grep -E "(Retrieved|Selected|Sending|response)"
```

### Interpret Results
```
✅ If retrieval is consistent → LLM is the problem (temperature, prompt)
❌ If retrieval varies → Search is the problem (scores, embeddings, limit)
```

---

## 🔍 Key Metrics Tracked

### Retrieval Consistency
- Same chunks retrieved each run? → Check logs for hit IDs
- Same scores? → Search logs show score variance
- Same order? → Typesense ranking consistency

### Context Delivery
- Is Article 21 in top 5? → Search logs
- Is it in final context? → Context selection logs
- Is it being cut off? → "Context limit hit" warning in logs

### LLM Behavior
- Same response for same context? → Compare ollama_client logs
- Does refusal happen without reason? → Check context received by LLM
- Is temperature affecting responses? → Test with temp=0

---

## 📋 Debugging Workflow

```
1. Run: python tests/test_article21_consistency.py
   ↓
2. Check: FINAL VERDICT in output
   ├─→ Retrieval CONSISTENT? → Go to step 3 (LLM problem)
   └─→ Retrieval INCONSISTENT? → Go to step 4 (Search problem)
   ↓
3. LLM Debug
   - Check if same context → different answers?
   - Test with temperature=0 (deterministic mode)
   - Review logs: article21_debug.log
   ↓
4. Search Debug
   - Do embedding vectors vary for same query?
   - Do Typesense scores fluctuate?
   - Is character limit excluding Article 21?
   - Check: RAG_MAX_CONTEXT_CHARS setting in config.py
```

---

## 🎁 Bonus: Targeted Fixes

Based on what you find:

| Finding | Fix |
|---------|-----|
| **Retrieval varies** | Check if embeddings deterministic, add seed to Typesense |
| **Article 21 cut off** | Increase `RAG_MAX_CONTEXT_CHARS` in config.py |
| **LLM says "cannot find"** | Set `temperature=0` for consistency |
| **Prompt confusion** | Refine system prompt in prompts.py |

---

## 📁 What Was Created/Modified

### Modified Files (Logging Added)
- ✅ `backend/services/rag_service.py` - +30 lines
- ✅ `backend/services/search_service.py` - +15 lines
- ✅ `backend/services/prompts.py` - +20 lines
- ✅ `backend/clients/ollama_client.py` - +12 lines
- ✅ `backend/clients/gemini_client.py` - +12 lines

### New Files Created
- ✅ `tests/test_article21_consistency.py` - 250 lines (test harness)
- ✅ `DEBUG_LLM_INCONSISTENCY.md` - Detailed guide
- ✅ `LOGGING_INSTRUMENTATION_SUMMARY.md` - Implementation summary
- ✅ `LLM_DEBUG_CHECKLIST.md` - This file

---

## ✨ Expected Outcome

After running the test harness, you'll know:

| Question | Source |
|----------|--------|
| Is Article 21 retrieved consistently? | search_service logs |
| Is it included in final context? | prompts.py logs |
| Is it reaching the LLM? | rag_service logs |
| Is LLM inconsistent with same context? | ollama_client logs |
| **Where is the variance?** | FINAL VERDICT from test |

---

## 🎯 Success Criteria

✅ All 20 queries return identical answers
✅ All cite sources correctly  
✅ No spurious "cannot find" responses
✅ Test logs show no variance in retrieval/context
✅ Confidence: High (backed by detailed logs)

---

## 💡 Pro Tips

1. **Run at night** if your Typesense is shared with other services
2. **Test embedding determinism:**
   ```python
   from backend.models.embeddings_model import EmbeddingsModel
   emb = EmbeddingsModel()
   v1 = emb.embed_query("Article 21")
   v2 = emb.embed_query("Article 21")
   assert v1 == v2  # Should be identical
   ```

3. **Enable environment DEBUG logging:**
   ```bash
   export LOG_LEVEL=DEBUG
   python -m backend.main 2>&1 | tee app.log
   ```

4. **Compare runs side-by-side:**
   ```bash
   python tests/test_article21_consistency.py > run1.log 2>&1
   python tests/test_article21_consistency.py > run2.log 2>&1
   diff run1.log run2.log | head -50
   ```

---

## 📞 Questions Answered

**Q: Where does the inconsistency come from?**
A: Run the test. The FINAL VERDICT tells you: retrieval or LLM.

**Q: Will this slow down my app?**
A: DEBUG logs have minimal overhead. They only activate in debug mode.

**Q: Can I use this in production?**
A: Debug logging is safe. Change log level to INFO for production.

**Q: How long does a test run take?**
A: ~30-60 seconds for 20 queries (depends on LLM response time).

---

**Ready? Start here:** `python tests/test_article21_consistency.py`
