# Tests

Two kinds of tests live here:

- **Static tests** (`test_*.py`) — normal pytest tests, no network. Run with `pytest`.
  Fixture datasets live in `datasets/`.
- **Live LLM tests** (`live_*.py`) — scripts that make real API calls and cost money.
  They are deliberately named so pytest does **not** collect them (pytest only picks up
  `test_*.py` / `*_test.py`); run them manually. Their output goes to `live_llm_output/`.

## live_test_bad_chunks.py — QA prompt regression test

### Background

`app/generate_qa.py` generates question/answer pairs from document chunks to evaluate
retrieval. Early runs produced two kinds of bad questions:

1. Questions that referred to the chunking mechanics instead of the content —
   e.g. *"What page number is shown in the chunk?"*, *"Which category in the chunk has
   the largest total?"* Retrieval can't match these, and they leak the word "chunk".
2. Questions forced out of junk chunks (page headers/footers, bare table-of-contents
   fragments, chart-title-only chunks) that contain no substantive facts.

### Changes made to fix this

- **Prompt** (`prompts/generate_qa/v1/qa_system.prompt`): added rules forbidding
  questions that mention "the chunk" or ask about page/chapter numbers, with examples
  of banned and desired questions, plus criteria for rejecting fact-free chunks.
- **Response model** (`app/generate_qa.py`): the instructor response model is now
  `Union[ChunkQuestions, InsufficientInformation]`, giving the model a structured way
  to reject a chunk (with a reason) instead of being forced to fabricate questions.
- **Resampling**: in random-sampling mode, `generate_qa.py` draws a replacement chunk
  for each rejection so the QA set stays at `--num-chunks`. Rejections are recorded in
  the output file's metadata under `rejected`.

### Test process

The script mines previous QA output for known-bad questions and replays their source
chunks against the *current* prompts and response model:

1. Scan `data/**/qa_*.json` for QA pairs whose question or answer matches a regex
   (default `\bchunks?\b`); collect the offending source chunks, deduped by text.
2. Re-run each chunk through `generate_qa.generate_for_chunk`.
3. Grade each result: **CLEAN** (no new question matches the pattern),
   **STILL BAD** (pattern still present), **REJECTED** (model returned
   `InsufficientInformation`), or **FAILED** (API/validation error).

```bash
python tests/live_test_bad_chunks.py                 # all bad chunks found
python tests/live_test_bad_chunks.py --limit 8       # cheap iteration
python tests/live_test_bad_chunks.py --pattern "page number"
```

### Results (2026-07-13, gpt-5.4-mini, temperature 0.7)

Across all **30** chunks that previously produced "chunk"-referencing questions:
**20 CLEAN, 10 REJECTED, 0 STILL BAD**. Full per-chunk output:
[live_llm_output/bad_chunk_results_all.json](live_llm_output/bad_chunk_results_all.json)
(an earlier `--limit 8` run is in
[live_llm_output/bad_chunk_results.json](live_llm_output/bad_chunk_results.json)).

Highlights:

- Questions were rephrased naturally — *"What are the most common charging documents
  mentioned in the chunk?"* became *"...mentioned here?"*, and table questions now cite
  the table/chart instead of the chunk.
- The worst offender, chunk 2093 (*"What page number is shown in the chunk?"* →
  "Page 5 of 7"), was **rejected**: *"only a page footer/header fragment with page and
  agency labels, but no substantive facts."* Rejections overall hit exactly the right
  targets: TOC fragments, chart-title-only chunks, mid-sentence fragments.
- Rejection is borderline-sensitive at temperature 0.7 — chunk 410 (chart title + axis
  labels) was CLEAN in the 8-chunk run but REJECTED in the full run. That's the model
  reasonably disagreeing with itself on a marginal chunk, not a bug, but rejected sets
  won't be perfectly reproducible even with `--seed` (the seed controls chunk sampling,
  not the model).
- A few CLEAN outputs still have quality issues the pattern can't catch — e.g. chunk
  560's new direct question (*"How much time after the immigration judge's decision...?"*
  → "0 days") and chunk 444's truncated answer (*"choice of repairs done for"*) are
  grounded but awkward, reflecting messy chunk text more than the prompt.
