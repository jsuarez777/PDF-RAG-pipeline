# RAG Pipeline Evaluation: Results & Conclusions

This document goes over the results and conclusions of about 440 evaluated experiments across 5 PDF documents, sweeping chunking strategy, embedding models, vector stores, retrieval methods, hybrid fusion weight, BM25 tokenizer, and reranking providers. The data here comes from the eval JSON artifacts under `data/`, which keep each experiment's per-question results. The charts referenced below live in
[visualizations/20260725_results_writeup/](visualizations/20260725_results_writeup/).

Process:  Each chunking configuration gets its own synthetic QA dataset
(gpt-4.1-mini, Instructor-validated, 20 sampled chunks × 2-3 question types), so the ground-truth
chunk IDs always match the configuration under test. Each question has exactly one gold chunk, which
caps Precision@K at 1/K and leaves MRR, Recall@K and NDCG@K as the metrics worth reading. The project
targets were MRR ≥ 0.85 and Recall@5 ≥ 0.90.

## The documents

| Document | Type | Pages | Content Analysis |
|---|---|---|---|
| FY10 Statistical Yearbook (`fy10syb`) | statistical yearbook | 119 | table heavy mostly statistics |
| UK Knowledge & Innovation Analysis | government statistical report | 29 | prose + open-style (unruled) tables |
| Attitudes to Housing | survey report | 180 | prose + hundreds of survey tables |
| Probabilistic Deep Learning | technical book | 297 | math/code-heavy prose |
| AI Agents and Applications | technical book | 450 | API/code-heavy prose |

## Headline results

| Document | Best configuration | MRR | R@5 | Targets met |
|---|---|---|---|---|
| AI Agents & Apps | fixed 512/50, hybrid α=0.3, 3-large, **Cohere rerank** | **0.958** | **1.000** | Yes |
| AI Agents & Apps (no rerank) | fixed 512/50, hybrid α=0.3, 3-large | 0.919 | 1.000 | Yes |
| UK Innovation report | plumber-struct, vector, 3-small | 0.958 | 1.000 | Yes |
| Attitudes to Housing | plumber-struct, hybrid α=0.5, 3-large | 0.933 | 1.000 | Yes |
| Probabilistic DL | fixed 512/100, hybrid α=0.7, bge-base, **Cohere rerank** | 0.900 | 0.950 | Yes |
| FY10 Yearbook | sentence-dynamic-min 8/2, vector, 3-small | 0.612 | 0.725 | No |

On AI Agents that 0.958 is a two-way tie: BM25 + Cohere rerank on the same fixed 512/50 chunks scores
identically (MRR 0.958, R@5 1.000). The hybrid row is listed because it is the stronger config before
reranking.

Four of the five documents clear both targets. FY10 was mostly a first-run document — tiny 100-256
character chunks (predating token-based sizing) and v1 QA prompts that produced unanswerable
questions, best MRR 0.581 — and it was set aside after that. The row above is a later re-run once the
methodology had been fixed, and it only reaches 0.612, so the gap is the document, not the vintage of
the pipeline. Running on FY10 Yearbook was still troublesome as 
it consisted of various similarly named and typed statistics tables, which mage generataing ground-truth chunk questions difficult to distinquish from posibilty questions in related tables.  A different approach
to this document type is needed, but I placed emphasis on fixing issues that address general document types.


---

## Finding 1: Document type decides BM25 vs. embeddings

![Best MRR by retrieval family per document](visualizations/20260725_results_writeup/retrieval_family_by_doc.png)

The strongest pattern in the data is that which retrieval family wins is a property of the
document, not of the method.

- Technical books favor BM25. On *AI Agents & Apps*, BM25 beats every embedding model tested (0.90
  against 0.80 for the best vector run); on *Probabilistic DL* it edges them out, 0.78 to 0.73.
  Questions about technical material reuse the book's exact terminology
  (`create_react_agent`, "aleatoric uncertainty", "PixelCNN"), which a keyword matcher gets for free,
  and there is no paraphrasing here for embeddings to be tolerant of.
- Statistical and survey reports favor embeddings. UK Innovation: vector 0.96 vs. BM25 0.81. Housing:
  0.87 vs. 0.78. FY10: 0.61 vs. 0.42. Report prose gets paraphrased in the questions ("proportion of
  people content" against "% satisfied"), and near-identical survey tables differ only in numbers
  that the question text never contains, so exact-term matching has nothing to grip.

Practical consideration: look at your corpus before picking a retriever. If user queries are going to
echo the document's own vocabulary, BM25 is a ~1 ms/query baseline that embeddings may well fail to
beat.

## Finding 2: There is an ideal chunk size, and bigger is not better

![MRR by chunking config on AI Agents](visualizations/20260725_results_writeup/chunking_ai_agents.png)

From the largest controlled grid (AI Agents, 40 questions per config, everything else held fixed):

| Chunking (tokens/overlap) | # chunks | BM25 | Vector 3-small | Hybrid α=0.3 3-large |
|---|---|---|---|---|
| fixed 256/30 | 928 | 0.70 | 0.59 | 0.76 |
| **fixed 512/50** | 454 | **0.90** | **0.73** | **0.92** |
| fixed 1024/120 | 232 | 0.72 | 0.46 | 0.69 |

- 512 tokens was the best size on both technical books; PDL's winning config is also 512-token.
- 1024-token chunks lose everywhere, and they lose worst for embeddings, where vector drops to 0.46.
  A single vector that averages several concepts matches none of them well. BM25 degrades more
  gracefully, since term frequency still works inside a long chunk.
- The very small chunks of the first runs (100-200 *characters*, on FY10, before the switch to
  token-based sizing) were the opposite failure mode: fragments too small to carry an answerable
  fact.

## Finding 3: Structure-aware chunking wins on table-heavy documents

Fixed-size and sentence chunking both flatten tables into shredded number fragments. The
`plumber-struct` chunker (pdfplumber's structured extraction: prose into sentence chunks, each
detected table into header-labeled row chunks, images into descriptor chunks) is the best strategy on
both report documents:

| Document | Best fixed-size | Best sentence | **plumber-struct** (avg gain) |
|---|---|---|---|
| Attitudes to Housing (hybrid) | 0.73 | 0.83 | **0.93** (+0.15) |
| UK Innovation (vector) | 0.78 | n/a | **0.96** (+0.18)|
| AI Agents (hybrid α=0.3) | 0.92 | 0.80 | 0.89 (-)|

On mostly prose books it stays competitive without winning, as there is little
structure to exploit, and its sentence-sized text chunks behave like sentence chunking. Two
supporting details fell out of the failure analysis.

- On Housing, plain 5-sentence chunking left **10 of 60 questions that no retriever could find** in
  the top-10, the gold facts having been split mid-table. The sentence-dynamic-min and plumber-struct
  configs had no all-miss questions at all.
- plumber-struct only became viable on the UK document after the open-table detection work in
  iteration 7. Default pdfplumber found 10 chart-frame false positives and none of the real tables;
  the rule-rect detector plus the fill guard recovered 8 real tables and dropped every fake.

## Finding 4: 3-small averaged better than 3-large at 20% the cost and half the size

![Vector MRR by embedding model](visualizations/20260725_results_writeup/embeddings_ai_agents.png)

- Averages across all tests: 3-small = 0.64 | 3-large = 0.62 | MiniLM-L6 = 0.57
- text-embedding-3-large and text-embedding-3-small are effectively tied on the books, differing by
  ±0.03-0.07 in either direction across configs. 3-large's only consistent edge is on the
  paraphrase-heavy reports (Housing 0.87 vs. 0.78). At roughly 6.5× the price per token it earns its
  keep only where paraphrase robustness is the bottleneck.
- all-MiniLM-L6-v2 (local, free) trails by approx 10% on MRR: usable for prototyping, not for quality.
- bge-base-en-v1.5 (local, free) matched or beat 3-small on 4 of the 5 chunk configs in the second
  AI Agents grid (0.73 vs. 0.70 at 256/30, 0.70 vs. 0.64 at 512/50). For this corpus a local
  embedding model is a legitimate cost-saver.
- Vector store choice made no difference to quality. Milvus and ChromaDB scored within 0.007 of each
  other on identical vectors. 
- The BM25 tokenizer (word vs. Porter stemming) was similar: ±0.05, no consistent winner.

## Finding 5: Hybrid fusion improves results, but α should lean toward the stronger signal

![Hybrid alpha sweep](visualizations/20260725_results_writeup/hybrid_alpha_sweep.png)

Hybrid fusion (min-max-normalized `α·vector + (1−α)·BM25` over a 5×k candidate pool) was the best or
near-best family on every document

- On the BM25-friendly book, α=0.3 (lexical-heavy) wins everywhere, and the project's old default of
  α=0.7 costs up to 0.09 MRR. On the embedding-friendly reports the best α landed between 0.5 and
  0.7.
- Hybrid's real value is robustness. It beat *both* of its parents on Probabilistic DL (0.83 against
  0.78 and 0.73), where the two signals were complementary, and it roughly matched the stronger
  parent everywhere else. If you can't A/B per corpus, hybrid fusion with a slight α toward your best guess is a reasonable choice.

## Finding 6: Reranking helps vector search most, but it isn't free or always positive

![Rerank impact](visualizations/20260725_results_writeup/rerank_impact.png)

Each triplet of bars is one retrieval (fixed-sized chunks 512/50 overlap [left chart]; plumber-struct chunking [right chart]) pass re-scored three ways (no rerank / local cross-encoder /
Cohere): on AI Agents (left) Cohere lifts every retrieval type while the local cross-encoder only helps
vector, and on Housing's best config (right) both rerankers lower MRR.
***


Whereas the chart above shows one config (fixed size 512/50), the table below averages over all chunking types/embedding stores/BM25 index/α for the AI Agents PDF. This shows an average over 124 reranked runs (92 local cross-encoder, 32 Cohere), each paired with its own un-reranked results.
The table is the mean MRR change over those pairs, so a reranker can
be positive on average overall and still lose on the single config in the chart:

| Retrieval | + local cross-encoder (ms-marco-MiniLM) | + Cohere rerank-v3.5 |
|---|---|---|
| BM25 | +0.03 | +0.08 |
| Hybrid | +0.05 | +0.08 |
| Vector | **+0.16** | **+0.19** |

Futher digging through data shows:
- Cohere beats the local cross-encoder on ever pair they were both invoked on, and it pushed the best hybrid config to the overall best result: 0.919 to 0.958, with R@5 of 1.000. It also rescued weak configs: BM25+Cohere on the same 512/50 chunks reached 0.958 with R@5 1.000, tying that overall best, and hit 0.940 with R@5 1.000 on the second grid.
- Vector search benefits most probably because embeddings retrieve the right neighborhood but misorder it, and a cross-encoder fixes the ordering. BM25's top-1 is usually already right when it retrieves
  anything at all.
- Reranking can hurt a ranking that is already excellent. On Housing's best config it *lowered* MRR
  from 0.933 to 0.879 (local) and 0.854 (Cohere), demoting correct table-row chunks whose text might look indistinguishable from similar texts in the document to a cross-encoder.

![Latency vs quality](visualizations/20260725_results_writeup/latency_vs_quality.png)

Per-query latency on this hardware: BM25 around 1 ms, vector 5-25 ms, plus 75-110 ms for the local
reranker or 150-180 ms for Cohere including the API round-trip. On the book, the quality/latency
Pareto set is BM25 (1 ms, 0.90), hybrid α=0.3 (23 ms, 0.92), and hybrid+Cohere (~180 ms, 0.958).

## Finding 7: the QA dataset is the ceiling on the whole evaluation

Every retrieval number in this document is measured against synthetic questions, so it is worth
looking at what those questions are. `app/generate_qa.py` asks the generator for three questions per
sampled chunk, each answerable from that chunk alone, deliberately spanning a difficulty range:

| Type | What it asks | Why it's in the set |
|---|---|---|
| `direct` | Straightforward question, may reuse the chunk's own wording | Easy case; the lexical-match baseline |
| `paraphrased` | Same question as direct, but keywords swapped for synonyms | Tests whether retrieval survives vocabulary mismatch |
| `inference` | Requires inferring from the chunk, avoiding its exact words | Tests matching on meaning rather than keywords |

The chart below is mean MRR broken out by those three types, so the spread between the bars is a
measure of how much of a config's score comes from questions that simply echo the source text.

![Question types](visualizations/20260725_results_writeup/question_types.png)

- Question type moves scores as much as any pipeline knob does. On the reports, paraphrased and
  inference questions score 0.14-0.28 MRR below direct questions; on the books that gap disappears,
  since technical paraphrases still reuse the key term.
- The biggest single quality lever in the project was fixing QA generation rather than retrieval
  (iteration log 1-4). Banning "according to the chunk..." phrasing, letting the model reject
  low-information chunks, and requiring a distinctive anchor in every question lifted measured R@5 by
  0.09-0.15 on identical chunk types and indexes. The 0.32-0.58 MRR of the first runs was mostly a broken
  benchmark, not broken retrieval.
- The one lingering problem, which I later found was stated intentionally in the prompt (I don't recall why) is regarding questions generated from book index and TOC pages ("What pages are associated with PixelCNN?").  I'm guessing I must have planned for future posiblities like a user asking to be directed to a topic by asking instead of looking up in the index. In the two *AI Agents & Apps* experiment grids these ran 1-2 questions per 40, audited by checking which questions *every* one of the 11-13 experiments missed.
  9 such pairs were manually fixed or removed (documented in each QA file's `manual_edits` metadata)
  and the affected evals re-run. Filtering index and TOC pages before sampling would remove the
  category.  In retrospect, perhaps it would be best to allow questions on TOC and index, but create special chunk types and future experiments to test retrieving data specifically for that purpose.

---

## Methodology notes & limitations

- Chunk IDs and boundaries differ per configuration, so every QA set
  here was generated from the specific chunk run it evaluates, which is the fair-evaluation
  requirement from the project spec.  It doesn't make sense to use different chunk strategies, but use a question that points to a specific chunk x in 512-token chunks when that section is actually chunk x/2 when using 1024-token sizes.  What might make sense is to re-use the question, but update the chunk appropriately. In this case, the method used was just to generate a new set of questions from a random chunk sampling.
- Sample sizes run 38-60 questions per config. One question is worth about 0.025 recall, so
  differences under roughly 0.05 are noise, and the conclusions above rest on patterns that repeat
  across configs and documents. 
- 1:1 gold labels undercount overlap. With overlapping fixed-size chunks, a neighbor containing the
  same fact as gold still counts as a miss. Multi-chunk gold labels would raise the absolute numbers, and also allow for true precision evaluation.  With only one gold chunk per query, precision is capped at 1/k.
- Rerank comparisons are paired (same retrieval pass, top-10 rescored), so the rerank deltas carry no
  retrieval variance.

## What would come next

1. Multi-chunk gold labels, crediting any chunk that contains the fact, to remove the overlap
   artifact, and allow truly measuring precision.
2. An index/TOC page filter ahead of QA sampling.
3. Per-corpus α tuning, gridded per document type. The data shows one global default is wrong.
4. Answer-stage evaluation. Retrieval metrics are only a proxy; wiring the retrieved chunks into a
   generator with citation checking would validate the thing end to end.
