"""
Test harness to reproduce and debug Article 21 query inconsistency.

Usage:
    python -m pytest tests/test_article21_consistency.py -v -s
    
Or standalone:
    python tests/test_article21_consistency.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.config import Settings
from core.dependencies import get_rag_service
from models.schemas import ChatResponse

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("article21_debug.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


async def run_consistency_test(
    query: str = "What does Article 21 say?",
    iterations: int = 20,
    delay_seconds: float = 0.5,
) -> dict:
    """
    Run the same query multiple times and collect results.

    Args:
        query: Question to ask repeatedly
        iterations: Number of times to run the query
        delay_seconds: Delay between queries (to check for state issues)

    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 80)
    logger.info("Starting Article 21 Consistency Test")
    logger.info(f"Query: '{query}'")
    logger.info(f"Iterations: {iterations}, Delay: {delay_seconds}s")
    logger.info("=" * 80)

    service = get_rag_service()
    responses: list[ChatResponse] = []
    answers: list[str] = []
    context_sizes: list[int] = []
    chunk_counts: list[int] = []

    for i in range(iterations):
        logger.info(f"\n--- Run {i + 1}/{iterations} ---")
        try:
            response = await service.answer(query)
            responses.append(response)
            answers.append(response.answer)
            context_sizes.append(sum(len(c.content) for c in response.chunks_used))
            chunk_counts.append(len(response.chunks_used))

            logger.info(f"Answer (first 100 chars): {response.answer[:100]}...")
            logger.info(f"Chunks used: {chunk_counts[-1]}, context size: {context_sizes[-1]} chars")
            logger.info(f"Citations: {len(response.citations)}")

            if chunk_counts[-1] > 0:
                logger.info(
                    f"Chunks: {[f'{c.pdf_name}(p{c.page_start})' for c in response.chunks_used]}"
                )

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            answers.append(f"ERROR: {str(e)}")
            context_sizes.append(0)
            chunk_counts.append(0)

        if i < iterations - 1:
            await asyncio.sleep(delay_seconds)

    # Analyze results
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS RESULTS")
    logger.info("=" * 80)

    unique_answers = list(set(answers))
    logger.info(f"\n1. ANSWER VARIANCE: {len(unique_answers)} unique responses")
    for j, answer in enumerate(unique_answers, 1):
        count = answers.count(answer)
        percentage = (count / len(answers)) * 100
        logger.info(f"   [{j}] {count}x ({percentage:.1f}%): {answer[:80]}...")

    logger.info(f"\n2. CHUNK COUNT VARIANCE: min={min(chunk_counts)}, max={max(chunk_counts)}")
    logger.info(f"   Unique counts: {sorted(set(chunk_counts))}")
    for count in sorted(set(chunk_counts)):
        occurrences = chunk_counts.count(count)
        logger.info(f"   {count} chunks: {occurrences}x")

    logger.info(f"\n3. CONTEXT SIZE VARIANCE: min={min(context_sizes)}, max={max(context_sizes)}")
    logger.info(f"   Average: {sum(context_sizes) / len(context_sizes):.0f} chars")

    # Check if Article 21 is consistently retrieved
    logger.info("\n4. ARTICLE 21 RETRIEVAL CHECK:")
    article21_runs = 0
    for i, response in enumerate(responses):
        if response.chunks_used:
            combined_content = " ".join(c.content for c in response.chunks_used).lower()
            if "article 21" in combined_content or "article 21" in combined_content:
                article21_runs += 1
                logger.info(f"   Run {i + 1}: ✓ Article 21 found in context")
            else:
                logger.info(f"   Run {i + 1}: ✗ Article 21 NOT in context")

    logger.info(f"   Total: {article21_runs}/{len(responses)} runs included Article 21")

    # Check if "cannot find" is in responses
    logger.info("\n5. REFUSAL PATTERN:")
    refusal_pattern = "could not find information"
    refusal_count = sum(1 for a in answers if refusal_pattern.lower() in a.lower())
    logger.info(f"   Refusals: {refusal_count}/{len(answers)} ({(refusal_count/len(answers)*100):.1f}%)")

    # Build result dict
    result = {
        "query": query,
        "total_runs": iterations,
        "unique_responses": len(unique_answers),
        "chunk_count_variance": max(chunk_counts) - min(chunk_counts),
        "context_size_variance": max(context_sizes) - min(context_sizes),
        "article21_consistency": article21_runs / len(responses),
        "refusal_rate": refusal_count / len(answers),
        "is_consistent": len(unique_answers) == 1 and refusal_count == 0,
    }

    logger.info("\n6. SUMMARY:")
    logger.info(f"   Consistent: {result['is_consistent']}")
    logger.info(f"   Variance score: {len(unique_answers) + result['chunk_count_variance']}")

    return result


async def run_retrieval_only_test(
    query: str = "What does Article 21 say?",
    iterations: int = 20,
) -> dict:
    """
    Test if retrieval is consistent (independent of LLM).
    """
    logger.info("\n" + "=" * 80)
    logger.info("RETRIEVAL CONSISTENCY TEST (no LLM)")
    logger.info("=" * 80)

    service = get_rag_service()
    all_hits: list[list] = []

    for i in range(iterations):
        logger.info(f"\nRun {i + 1}/{iterations}:")
        result = await service._retrieve_context(query, top_k=None)
        hits_info = [
            {
                "pdf_name": h.pdf_name,
                "chunk_index": h.chunk_index,
                "page_start": h.page_start,
                "score": h.score,
                "content_len": len(h.content),
            }
            for h in result
        ]
        all_hits.append(hits_info)
        logger.info(f"Retrieved {len(result)} chunks")
        for j, h in enumerate(result[:3]):
            logger.info(f"  [{j}] {h.pdf_name} (p{h.page_start}, score={h.score})")

    # Check consistency
    unique_retrieval_patterns = []
    for pattern in all_hits:
        pattern_str = json.dumps(pattern, sort_keys=True)
        if pattern_str not in unique_retrieval_patterns:
            unique_retrieval_patterns.append(pattern_str)

    logger.info(f"\nRetrieval consistency: {len(unique_retrieval_patterns)} unique patterns")
    if len(unique_retrieval_patterns) == 1:
        logger.info("✓ Retrieval is CONSISTENT (all runs identical)")
    else:
        logger.info("✗ Retrieval is INCONSISTENT (different results across runs)")

    return {
        "retrieval_consistent": len(unique_retrieval_patterns) == 1,
        "unique_patterns": len(unique_retrieval_patterns),
        "total_runs": iterations,
    }


if __name__ == "__main__":
    # Run both tests
    result1 = asyncio.run(run_consistency_test(iterations=20, delay_seconds=0.2))
    result2 = asyncio.run(run_retrieval_only_test(iterations=20))

    logger.info("\n" + "=" * 80)
    logger.info("FINAL VERDICT")
    logger.info("=" * 80)

    if result2["retrieval_consistent"]:
        logger.info("→ Retrieval is consistent")
        if result1["is_consistent"]:
            logger.info("→ Answers are also consistent → No issue detected")
        else:
            logger.info("→ Answers are INCONSISTENT → Issue is in LLM/prompt layer")
    else:
        logger.info("→ Retrieval is INCONSISTENT → Issue is in retrieval/search layer")

    logger.info(
        f"\nDetailed logs: article21_debug.log\n"
        f"Review that file for full debug output with timestamps."
    )
