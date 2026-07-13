# Iteration Log

Structured log of major experiments and pipeline changes, per the format in `scratch/project_goals` (Iteration Logs section).

### Iteration 1: Baseline retrieval evaluation on attitudes_to_housing (60 synthetic QA pairs)
- **Date**: 2026-07-13
- **Change**: First full retrieval evaluation of the `20260713_02/attitudes_to_housing` run: BM25 (word tokenization) plus vector search over ChromaDB and Milvus indexes, each with text-embedding-3-small and text-embedding-3-large. 60 QA pairs (direct / inference / paraphrased).
- **Hypothesis**: Baseline configurations should land near the reference targets (MRR ≥ 0.85, Recall@5 ≥ 0.90) if chunking and QA generation are sound.
- **Result**: Poor across all retrieval types — Recall@5 ≤ 0.65 everywhere.
  - Best overall: chromadb/milvus + 3-large — MRR 0.429-0.430, R@1 0.317, R@5 0.600, R@10 0.667, NDCG@10 0.486-0.487
  - 3-small (chromadb/milvus): MRR 0.393-0.397, R@5 0.533-0.550
  - BM25 (word): MRR 0.305, R@5 0.383; paraphrased questions collapse to MRR 0.062 / R@5 0.100
  - By question type, direct > paraphrased > inference in every configuration (best direct R@5 only 0.650).
- **Review of QA quality**: Spot-check of generated questions found many that ask for information from "the chunk" rather than naming a topic, making them unanswerable/unmatchable as standalone retrieval queries. Two captured examples:
  - "What is the chapter title shown in the chunk?"
  - "What is the base sample size reported in the chunk?"

  Additionally, some questions were generated from chunks with no reason-able content — table-of-contents pages and sections that merely list page numbers / index entries.
- **Decision**: Reject this iteration.
- **Next step**: Revise QA generation prompts so questions never refer to "the chunk" and instead name the topic/subject the chunk covers. Also consider excluding low-information chunks (table of contents, index, page-number-only sections) from QA generation, since they contain nothing that can be reasoned about. Re-generate QA and re-run the evaluation.

### Iteration 2: QA generator rework — prompt rules, chunk rejection, resampling (KEEP)
- **Date**: 2026-07-13
- **Change**: Reworked `app/generate_qa.py` and its prompts to improve the quality of the QA benchmark itself (questions must be answerable, standalone retrieval queries that a retriever can plausibly match to their source chunk):
  - Prompts moved out of the code into `prompts/generate_qa/v1/` (`qa_system.prompt`, `qa_user.template`), mirroring the miniproject2 convention; output metadata records `prompt_version`.
  - Prompt rules added: questions must never refer to "the chunk"; no page-number/chapter-number questions (unless the text is literally a TOC/index); answers must be substantive.
  - Structured rejection: instructor response model is now `Union[ChunkQuestions, InsufficientInformation]`, so the model can reject a low-information chunk (page header/footer, bare TOC fragment, reference list) with a reason instead of being forced to fabricate questions. Rejections are recorded in output metadata.
  - Resampling: in random-sampling mode a rejected chunk is replaced by a fresh draw so the QA set stays at `--num-chunks` (no resampling in explicit `--chunks` mode).
  - Regression test `tests/live_test_bad_chunks.py` (live API, not pytest-collected): replays previously bad chunks against the current prompts. See `tests/README.md`.
- **Hypothesis**: Removing chunk-referencing and junk-chunk questions should make the QA set a valid measure of retrieval; genuinely specific questions should be matchable to their source chunk, so recall on a sound QA set should improve.
- **Result**:
  - Regression test on all 30 previously bad chunks: 20 CLEAN, 10 REJECTED, 0 STILL BAD (`tests/live_llm_output/bad_chunk_results_all.json`). Rejections correctly hit TOC fragments, chart-title-only chunks, page footers, mid-sentence fragments.
  - New 60-question QA set + full eval run (eval_20260713_125028): mixed vs Iteration 1 — direct recall improved across all indexes (+0.03 to +0.13), inference/paraphrased dropped at shallow k, R@10 up on average (+0.057).
  - **Caveat — not a controlled comparison**: neither QA run used `--seed`, and the two 20-chunk samples have ZERO overlap, so per-type deltas are dominated by which chunks were drawn (n=20/type; one question = 0.05 recall).
  - Failure analysis of the new set: most misses target near-identical survey-table chunks, where the distinguishing detail (the numbers) lives in the answer, not the question — embeddings cannot distinguish "satisfaction % by age/tenure" table A from table B on question text alone.
  - Also noted: many of Iteration 1's "bad" questions quoted chunk text verbatim ("According to the chunk, by what percent did..."), which is a keyword gift to retrieval — the old set's shallow recall was artificially inflated, so the new, honestly-harder set is the more accurate benchmark.
- **Decision**: Keep. The generator changes stand; the eval comparison is inconclusive only because the chunk samples differ.
- **Next step**: Controlled comparison — regenerate QA with the new prompt on the exact chunk indices of the Iteration 1 QA file (via `--chunks`, no resampling), eval against the same indexes, and compare per question type on the surviving chunks. Use `--seed` for all future random QA runs.

### Iteration 3: Controlled comparison — new prompt on Iteration 1's exact chunks (KEEP)
- **Date**: 2026-07-13
- **Change**: Regenerated QA with the Iteration 2 prompt on the exact 20 chunk indices of the Iteration 1 QA file (`--chunks`, no resampling): `qa_20260713_130754_gpt-5.4-mini.json`. 7 of the 20 chunks were REJECTED as low-information (TOC fragments, truncated headings, partial tables) — i.e. 35% of the Iteration 1 benchmark was built on chunks that cannot support a real question. Evaluated the surviving 39 questions against the same 5 indexes (`eval_summary_20260713_130829`), and rescored the Iteration 1 eval restricted to the same 13 chunks for an apples-to-apples comparison.
- **Hypothesis**: With chunks held constant, the new prompt's questions (topic-naming, no "the chunk" references) should retrieve better than the old ones.
- **Result**: Confirmed — recall improves nearly across the board on identical chunks/indexes.
  - text-embedding-3-large (chromadb/milvus identical): overall MRR 0.358→0.467, R@5 0.513→0.667 (+0.15); paraphrased R@5 +0.23, R@10 +0.23; inference +0.15 at R@3/5/10.
  - BM25 paraphrased went from collapse to functional: R@3 0.000→0.308, R@10 0.077→0.385 — old paraphrased questions were unmatchable for keyword search.
  - Soft spot: inference questions on 3-small and BM25 dropped at shallow k (R@3 −0.15/−0.23) — new inference questions avoid the chunk's wording by design, which weak embeddings and keyword search genuinely struggle with; 3-large handles them fine.
  - Conclusion: Iteration 2's apparently "mixed" random-sample eval was a sampling artifact (zero chunk overlap, table-heavy draw). Controlled for chunks, the prompt rework is a clear win.
- **Decision**: Keep. Use `qa_20260713_130754` / the new prompt for future benchmarks; always pass `--seed` when sampling.
- **Next step**: Retrieval quality itself is still far below targets (best R@5 0.667 vs target ≥ 0.90). Next levers: chunking strategy (table-aware chunking; many misses are near-identical survey-table chunks indistinguishable from question text alone), hybrid BM25+embedding retrieval, and/or reranking.
