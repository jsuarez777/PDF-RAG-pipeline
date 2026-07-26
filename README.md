# RAG Retrieval Benchmark — PDF Extraction, Chunking & Retrieval Evaluation Pipeline

## Overview

This project is an end-to-end pipeline for building and **evaluating retrieval
quality** over real-world report PDFs. It extracts a PDF's text, tables, and
images, splits the text into chunks with several competing strategies, uses an
LLM to synthesize grounded QA pairs that serve as a retrieval benchmark, builds
BM25 and vector indexes, and scores every configuration on standard IR metrics
(Recall@K, Precision@K, MRR, MAP, NDCG@K). A grid runner sweeps whole
experiment matrices in one command, and a multi-user Flask viewer drives the
same pipeline from the browser with chunk-box, QA, and eval-miss overlays.

**Key capabilities:**
- **Extract** per-page text, tables, and embedded images from PDFs with
  pdfplumber — including a fill guard that drops chart-frame false-positive
  tables and a rule-rect detector that recovers open-style (horizontal-ruled)
  tables. Alternate path: render pages to PNG (pdf2image) and OCR with EasyOCR.
- **Chunk** with five strategies: `fixed_size` (token windows), `sentence`,
  `sentence-dynamic-min` (floors short table fragments), `plumber-struct`
  (structure-aware: prose sentences + header-labeled table rows + image
  descriptors), and `semantic` (embedding-based topic-shift cuts).
- **Generate** a grounded QA benchmark with the `instructor` library — three
  question types per chunk (direct / inference / paraphrased), with structured
  rejection of low-information chunks and resampling to hold the set size.
- **Index** chunks as BM25 pickles (simple / word / porter tokenizers) and as
  vector stores (OpenAI `text-embedding-3-small`/`-large`, or local
  `all-MiniLM-L6-v2` / `bge-base-en-v1.5`) in Milvus Lite or ChromaDB.
- **Retrieve** with BM25, pure vector, or **hybrid** fusion (per-list min-max
  normalized `alpha·vector + (1−alpha)·bm25`), optionally **reranked** by a
  second-stage cross-encoder (Cohere rerank API or a local CrossEncoder).
- **Evaluate** every index against the QA gold chunks, reporting metrics overall
  and per question type, with per-question retrieval timing.
- **Orchestrate** the whole grid (chunking × embeddings × retrieval) in one
  command and collect every experiment's metrics with the best config by MRR.
- **Visualize** the grid with 8 chart types (MRR bars, recall/precision scatter,
  heatmaps, recall@K curves, metric correlation, per-question-type, and a
  time-vs-quality Pareto chart).
- **Inspect** everything in a multi-user Flask viewer: run each stage from the
  header, overlay chunk boxes and QA questions on page images, and trace why a
  gold chunk missed the top-k.

The pipeline is designed for iterative improvement: extract → chunk → generate
QA → index → evaluate → visualize → refine chunking/prompts → repeat.

## Setup

### Requirements
- Python 3.10+ (developed and linted against 3.13)
- OpenAI API account with available credits (embeddings + QA generation)
- [poppler](https://poppler.freedesktop.org/) on the system PATH (pdf2image
  shells out to `pdftoppm`)
- Optional: a Cohere API key, only if you use `--rerank cohere`

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-remote-url> miniproject3
   cd miniproject3
   ```

2. **Create a Python virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt          # runtime pipeline
   pip install -r requirements-dev.txt       # dev: pytest, ruff, matplotlib, pandas, seaborn
   ```

   `requirements.txt` is everything the pipeline needs to run (openai, pdfplumber,
   pymupdf, pdf2image, easyocr, chromadb, pymilvus[milvus_lite], rank-bm25,
   sentence-transformers, cohere, instructor, pysbd, tiktoken, nltk, flask, …).
   `requirements-dev.txt` adds the linter, pytest, and the plotting stack used by
   `generate_visualizations.py`.

4. **Install poppler** (required for `pdf_to_images.py` and the viewer's image
   rendering — without it you'll see `Unable to get page count. Is poppler
   installed and in PATH?`):
   ```bash
   # macOS
   brew install poppler
   # Debian/Ubuntu
   sudo apt-get install poppler-utils
   ```

5. **Configure your OpenAI API Key**

   **Step 1: Create an OpenAI account**
   - Go to [https://platform.openai.com/signup](https://platform.openai.com/signup)
   - Sign up and verify your email address

   **Step 2: Add a payment method**
   - Navigate to [https://platform.openai.com/account/billing/overview](https://platform.openai.com/account/billing/overview)
   - Add a credit card (required for API access) and set usage limits if desired

   **Step 3: Generate an API key**
   - Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Click "Create new secret key" and copy it (you won't see it again)

   **Step 4: Set the environment variable**
   ```bash
   export OPENAI_API_KEY='sk-...'
   ```

   To persist across sessions, add it to your shell profile (`~/.profile`,
   `~/.bashrc`, `~/.zshrc`, etc.):
   ```bash
   echo "export OPENAI_API_KEY='sk-...'" >> ~/.profile
   source ~/.profile
   ```
   The bundled OpenAI client reads the key from the `OPENAI_API_KEY` environment
   variable, falling back to `~/.profile` if it isn't set. Reranking with Cohere
   reads `COHERE_API_KEY` / `CO_API_KEY` the same way (only needed for
   `--rerank cohere`).

(A short version of this lives in [SETUP.md](SETUP.md).)

## How It Works

All stage scripts in this pipeline follow a consistent pattern:
- **Interactive menus**: run a script with no arguments and it lists the
  discovered datasets / chunk runs / indexes / QA files and prompts for a
  selection (with a sensible default, usually the latest). Every menu choice is
  also available as a CLI flag (`--dataset`, `--db`, `--qa`, …) for
  non-interactive runs, and most `--dataset`/`--db` flags accept the list number
  the menu prints.
- **Directory scanning**: scripts auto-discover work from the on-disk layout —
  chunk runs are any `<title>/<chunk dir>/` containing a `chunked_text.json`,
  indexes live under `bm25/` and `embedding_databases/`, and QA/eval files sit
  next to the chunk run they belong to.
- **Provenance checks**: each index and QA file records the exact chunk run it
  was built from, and `eval_retrieval.py` refuses to score an index against a QA
  file built from a different chunk run unless you pass `--force`.
- **Parallel processing**: QA generation uses a bounded thread pool
  (`--parallel`, default 20) for concurrent LLM calls.
- **Automatic output management**: every run logs to `logs/` (timestamped) and
  writes artifacts into the dataset's own folder tree with timestamped names.

Generated data (`data/`), run logs (`logs/`), `scratch/`, and rendered
`visualizations/` are gitignored, but a reference dataset **is committed** — the
`data/pdfplumber/20260715_01/uk_knowledge_and_innovation_analysis` run with its
QA files and `evaluations/` (including `eval_summary_*.json`) — so the headline
before/after numbers below can be inspected without regenerating anything.

## Quick Start

**Run the whole grid with one command** (extract → chunk → QA → index →
evaluate), from an already-extracted dataset or straight from a PDF:
```bash
python app/run_pipeline.py \
  --dataset data/pdfplumber/20260715_01/uk_knowledge_and_innovation_analysis \
  --chunk-types "fixed_size:256:50,fixed_size:512:100,sentence:5:1" \
  --embeddings small,large --retrievals bm25,vector,hybrid --qa-num 20 --seed 7
# -> <title>/pipeline_runs/pipeline_<ts>.json  (every experiment's metrics + best config by MRR)

python app/run_pipeline.py --pdf docs/report.pdf --method pdfplumber --dry-run  # print the grid & count
python app/run_pipeline.py --dataset <dir> --rerank cohere --alpha 0.5,0.7,0.9  # add a rerank pass / alpha sweep
```
The runner shells out to the stage scripts, snapshots each stage's new
artifacts, wires them into the next stage, prints a Rich summary table, and
writes the full results file.

Or run the stages individually:

1. **Configure the API key** (if not already done):
   ```bash
   export OPENAI_API_KEY='sk-...'
   ```

2. **Extract a PDF** to per-page text + tables + images:
   ```bash
   python app/pdfplumber_to_text.py path/to/report.pdf
   # -> data/pdfplumber/<YYYYMMDD_NN>/<title>/page_<n>.json  (+ page_<n>_image_<m>.png)
   ```

3. **Chunk the extracted text**:
   ```bash
   python app/chunk_text.py --type plumber-struct --dataset data/pdfplumber/<date>/<title>
   # -> <title>/<ts>_chunk_<type>_<size>_<overlap>/chunked_text.json
   ```

4. **Generate the QA benchmark** for that chunk run:
   ```bash
   python app/generate_qa.py --dataset <chunk dir> --num-chunks 20 --seed 7
   # -> <chunk dir>/qa_<ts>_<model>.json
   ```

5. **Build indexes** (BM25 and/or vector):
   ```bash
   python app/index_bm25.py --tokenizer word --dataset <chunk dir>
   python app/embed_chunks.py --model small,large --db milvus --dataset <chunk dir>
   ```

6. **Evaluate** every matching index against the QA file:
   ```bash
   python app/eval_retrieval.py --qa <chunk dir>/qa_<ts>_<model>.json --db all --topk 10 --ks 1,3,5,10
   ```

7. **Visualize** the grid:
   ```bash
   python app/generate_visualizations.py            # all charts, latest evals -> visualizations/<ts>/
   ```

8. **Or drive it all from the browser**:
   ```bash
   python app/pdf_viewer.py                          # http://127.0.0.1:5001
   ```

## Testing

The project includes a pytest suite covering the chunker CLI boundaries, the
text-reflow logic, the IR metrics, the hybrid score fusion, and the grid runner.

**Run all tests:**
```bash
python -m pytest tests/ -v
```

**Run a specific test file:**
```bash
python -m pytest tests/test_chunk_text.py -v
```

**Test coverage:**
- **test_chunk_text.py** (36 tests): CLI boundary tests for `chunk_text.py` —
  type parsing (`fixed_size`/`sentence`/`sentence-dynamic-min`/`plumber-struct`/
  `semantic`), token-window↔char-offset mapping, overlap-as-percent, table/short-
  fragment handling, and file I/O against static fixture datasets copied to
  `tmp_path`.
- **test_reflow_text.py** (8 tests): resilience of the pdfplumber paragraph
  reflow (joining margin-wrapped lines, de-hyphenation, layout-gap paragraph
  breaks) so a wrapped sentence reads as one sentence to the pysbd segmenter.
- **test_metrics.py** (9 tests): the IR metrics in `eval_retrieval.py`
  (`gold_rank`, and `aggregate` → Recall@K / Precision@K / MRR / MAP / NDCG@K)
  checked against hand-computed values.
- **test_hybrid.py** (12 tests): the hybrid retrieval score fusion in
  `retriever_topk.py` (`minmax_normalize`, `combine_hybrid`).
- **test_run_pipeline.py** (14 tests): the grid-configuration logic in
  `run_pipeline.py` (chunk/embedding/retrieval/alpha expansion, config parsing).

All **79** tests pass with the current codebase. Tests use temporary directories
and static fixtures under `tests/datasets/` so runs never pollute the repo.

There are also **live LLM tests** under `tests/` named `live_*.py` (e.g.
`live_test_bad_chunks.py`) that make real API calls and cost money; they are
deliberately named so pytest does **not** collect them — run them by hand. See
[tests/README.md](tests/README.md).

**Lint / format** (dev tooling only; config in [pyproject.toml](pyproject.toml)):
```bash
ruff check .
ruff format .
```

## Pipeline Overview

### 1. Inputs — PDFs, prompts & the on-disk layout

- **PDFs** live under `pdfs/`; extraction writes datasets under
  `data/<extractor>/<YYYYMMDD_NN>/<title>/`, where `<extractor>` is `pdfplumber`
  (text + tables + images) or `pdf2image` (page PNGs, later OCR'd).
- **QA-generation prompts** live under `prompts/generate_qa/v*/`
  (`qa_system.prompt` + `qa_user.template`), versioned so a prompt can be pinned
  for an A/B (`--prompt-version`); v4 is the current default.
- A **chunk run** is a `<title>/<ts>_chunk_<type>_<size>_<overlap>/` folder
  containing `chunked_text.json`; its QA files (`qa_*.json`), indexes
  (`bm25/`, `embedding_databases/`), and evals (`evaluations/`) all live nearby
  and reference the chunk run they came from.

### 2. PDF Extraction (`pdfplumber_to_text.py`)

Extracts per-page prose, structured tables, and embedded images. Prose is
paragraph-reflowed (margin wraps joined, de-hyphenated, layout gaps → paragraph
breaks) so wrapped sentences survive downstream sentence segmentation. Table
detection unions two detectors and drops chart-frame false positives (see the
"before/after" section for the fill guard + rule-rect detector).

  **Parameters:** positional `[path/to/document.pdf]` (prompts if omitted).

  **Example:**
  ```bash
  python app/pdfplumber_to_text.py pdfs/uk_knowledge_and_innovation_analysis.pdf
  ```

  **Output:** `data/pdfplumber/<YYYYMMDD_NN>/<title>/page_<n>.json` with `text`
  (table-filtered prose), `tables` (structured rows), `full_text` (unfiltered
  fallback), and `images` (each crop-rendered to `page_<n>_image_<m>.png`);
  `logs/<ts>_pdfplumber_to_text.log`.

  *Alternate extraction path:* `pdf_to_images.py` renders each page to
  `data/pdf2image/<date>/<title>/page_<n>.png`, and `easyocr_pdfimages.py` OCRs
  those pages to `page_<n>_easyocr.json`. (`pdfplumber_extract.py` is an earlier
  flat variant that writes page JSON directly under the date dir.)

### 3. Chunking (`chunk_text.py`)

Splits a dataset's page text into a `chunked_text.json` with one of five
strategies. Every chunk keeps char offsets back into the source text (so the
viewer can highlight it) and, for `plumber-struct`, a `source` kind
(text/table/image).

  **Chunk types:**
  - `fixed_size:<size>:<overlap>` — token windows (tiktoken `cl100k_base`);
    `<overlap>` may be a percent (`10%`), mapped back to char offsets.
  - `sentence:<n>[:<overlap>]` — n sentences/chunk; oversized flattened tables
    hard-split at a token cap.
  - `sentence-dynamic-min:<n>[:<ov>]` — like `sentence` but keeps packing past n
    until a min-token floor is met, so shredded-table fragments aren't left tiny.
  - `plumber-struct[:<n>]` — pdfplumber only: prose as sentence chunks, each
    table as header-labeled row chunks, each image as a descriptor chunk.
  - `semantic:<max>[:<percentile>]` — embed each sentence, cut where the topic
    shifts (cosine distance above `<percentile>`, default 90) or the token cap
    `<max>` is hit (needs an OpenAI key).

  **Parameters:** `--type <spec>`, `--dataset <path|list#>` (menu if omitted).

  **Example:**
  ```bash
  python app/chunk_text.py --type "fixed_size:256:50" \
    --dataset data/pdfplumber/20260715_01/uk_knowledge_and_innovation_analysis
  ```

  **Output:** `<title>/<ts>_chunk_<type>_<size>_<overlap>/chunked_text.json`;
  `logs/<ts>_chunk_text.log`.

### 4. QA Benchmark Generation (`generate_qa.py`)

Randomly samples chunks and, via the `instructor` library, generates three
grounded questions per chunk (direct / inference / paraphrased), each answerable
from that chunk alone. The response model is
`Union[ChunkQuestions, InsufficientInformation]`, so the model can **reject** a
low-information chunk (TOC fragment, page footer, reference list) with a reason
instead of fabricating questions; in random-sampling mode a rejected chunk is
replaced by a fresh draw so the set stays at `--num-chunks` (rejections recorded
in metadata).

  **Parameters:**
  - `--dataset PATH` — chunk run (menu if omitted)
  - `--model MODEL` — chat model (default `gpt-4.1-mini`)
  - `--num-chunks N` — chunks to sample
  - `--seed N` — reproducible sampling (always pass for benchmarks)
  - `--temperature T` — default 0.7
  - `--parallel N` — max parallel API calls (default 20)
  - `--prompt-version V` — pin a prompt version for A/B (default: latest, v4)
  - `--chunks a,b,c` — use exact chunk indices instead of sampling (no resampling)
  - `--types` — subset of question types to keep
  - `--add FILE` — merge into an existing `qa_*.json` instead of writing a new one

  **Example:**
  ```bash
  python app/generate_qa.py --dataset <chunk dir> --num-chunks 20 --seed 7 --model gpt-4.1-mini
  ```

  **Output:** `<chunk dir>/qa_<ts>_<model>.json` (each item: source chunk,
  `chunk_index`, prompts, metadata incl. `rejected`); `logs/<ts>_generate_qa.log`.

### 5. Indexing

#### 5a. BM25 (`index_bm25.py`)
Builds a self-contained BM25 pickle (holds the `BM25Okapi`, tokenizer name, and
the chunks) from a chunk run.

  **Parameters:** `--tokenizer simple|word|porter` (comma-separated),
  `--dataset <chunk dir>`, `--all-options` (every tokenizer).

  **Example:**
  ```bash
  python app/index_bm25.py --tokenizer word --dataset <chunk dir>
  ```

  **Output:** `<title>/bm25/bm25_<ts>_<tokenizer>.pkl` (+ `.json` metadata sidecar).

#### 5b. Vector embeddings (`embed_chunks.py`)
Embeds the chunk run and stores the vectors in a vector DB.

  **Parameters:** `--model small|large|<full name>` (also local `minilm`/`bge`;
  comma-separated), `--db milvus|chromadb`, `--dataset <chunk dir>`,
  `--all-options` (every model × every db).

  **Example:**
  ```bash
  python app/embed_chunks.py --model small,large --db milvus --dataset <chunk dir>
  ```

  **Output:** `<title>/embedding_databases/milvus_<ts>_<model>.db` (+ sidecar) or
  `chromadb_<ts>_<model>.chroma/` (directory).

### 6. Retrieval Evaluation (`eval_retrieval.py`)

Runs each QA question against one or more stored indexes and scores the ranked
results against the question's gold `chunk_index`. Because each question has a
single gold chunk, Recall@K is a 0/1 hit rate, Precision@K = Recall@K / K, and
MAP == MRR. Metrics are reported overall and per question type, with
per-question retrieval timing.

  **Parameters:**
  - `--qa PATH` — QA file (menu if omitted)
  - `--db PATH,… | all` — indexes to score (`all` = every matching index)
  - `--hybrid VEC+BM25 | auto` — hybrid retrieval pairs, with `--alpha` (vector
    weight, default 0.7; comma-separated to sweep)
  - `--topk N` (default 10), `--ks 1,3,5,10` (metric cutoffs)
  - `--rerank cohere|local` (+ `--rerank-model`) — second-stage cross-encoder
    reorder, written to a separate `eval_*_rerank.json`
  - `--force` — score even if an index was built from a different chunk run

  **Example:**
  ```bash
  python app/eval_retrieval.py --qa <chunk dir>/qa_<ts>.json --db all --topk 10 --ks 1,3,5,10
  ```

  **Output:** one `<title>/evaluations/eval_<ts>_<index>.json` per index
  (per-question records + aggregates), plus an `eval_summary_<ts>.json` when
  several indexes are compared.

### 7. Reranking (`rerank.py`, `rerank_eval.py`)

`rerank.py` is the shared second-stage reranker used by `eval_retrieval.py`,
`retriever_topk.py`, and `run_pipeline.py`: a cross-encoder scores each
(query, chunk) pair and reorders the retrieved top-k. Providers: `cohere`
(rerank-v3.5 API; reads `COHERE_API_KEY`/`CO_API_KEY`) and `local`
(sentence-transformers `ms-marco-MiniLM-L-6-v2`, no key, scores sigmoid-squashed
to [0, 1]). `rerank_eval.py` reranks **already-written** eval files from their
stored retrieved lists (no re-retrieval), for cheap after-the-fact comparison.

  **Example:**
  ```bash
  python app/rerank_eval.py --eval <title>/evaluations/eval_<ts>_<index>.json --provider both
  ```

  **Output:** an `eval_*_rerank.json` alongside the source eval file.

### 8. Ad-hoc Retrieval (`retriever_topk.py`)

Retrieves the top-k chunks for a single free-text or JSON query against one
index (or a vector+BM25 hybrid), optionally reranked — useful for spot-checking
an index outside the eval loop.

  **Parameters:** `--db <vector index>`, `--bm25-db <bm25 index>` (+ `--alpha`)
  for hybrid, `--topk N`, `--rerank cohere|local`, and `--query-text "…"` or
  `--query-json <file>`.

  **Example:**
  ```bash
  python app/retriever_topk.py --db <chunk dir>/../embedding_databases/milvus_<ts>_small.db \
    --topk 5 --query-text "innovation activity by enterprise size"
  ```

### 9. Grid Runner (`run_pipeline.py`)

Orchestrates the full grid — chunking strategies × embedding models × retrieval
methods (bm25/vector/hybrid), optionally × alpha × rerank — by shelling out to
the stage scripts, snapshotting each stage's new artifacts, wiring them into the
next stage, and collecting every experiment's IR metrics into one results file
with the best config by MRR. Input is an existing `--dataset` or a `--pdf`
(first extracted via `--method pdfplumber|pdf2image`).

  **Parameters:** `--pdf`/`--dataset`, `--method`, `--chunk-types`,
  `--embeddings`, `--vector-db`/`--vector-configs`, `--tokenizers`,
  `--retrievals`, `--alpha`, `--qa-num`/`--qa-types`/`--qa-model`/`--seed`,
  `--rerank`/`--rerank-model`, `--topk`, `--ks`, `--dry-run`.

  **Example:**
  ```bash
  python app/run_pipeline.py --dataset data/pdfplumber/20260715_01/uk_knowledge_and_innovation_analysis \
    --chunk-types "fixed_size:256:50,fixed_size:512:100,sentence:5:1" \
    --embeddings small,large --retrievals bm25,vector,hybrid --qa-num 20 --seed 7
  ```

  **Output:** `<title>/pipeline_runs/pipeline_<ts>.json` (all experiments + best
  config); `logs/<ts>_run_pipeline.log`.

### 10. Visualization (`generate_visualizations.py`)

Reads the `eval_*.json` files (or one `--pipeline` results file) and renders the
required charts to `visualizations/<ts>/`.

  **Charts:** `mrr` (MRR bars by retrieval method), `scatter` (Recall@K vs
  Precision@K, top-5 labeled), `heatmap` (chunk config × index), `retrieval`
  (grouped bars per chunk config), `correlation` (metric correlation matrix),
  `recall-curve` (Recall@K vs K small multiples), `qtype` (metric by question
  type), `time-quality` (avg retrieval time vs MRR, Pareto-annotated).

  **Parameters:** `--chart <names…>` (default all), `--metric`, `-k`,
  `--pipeline <file>`, `--all-evals`, `--checkpoint <label>`, `--no-show`.

  **Example:**
  ```bash
  python app/generate_visualizations.py --chart mrr qtype --metric recall -k 5
  ```

  **Output:** `visualizations/<ts>/*.png`.

### 11. Multi-user PDF Viewer (`pdf_viewer.py`)

A Flask app (default `http://127.0.0.1:5001`) that drives the entire pipeline
from the browser. Each account gets an isolated workspace under
`data/users/<id>/` (pipeline scripts pointed at it via `PDF_DATA_DIR`).
Long-running work runs on a background job worker (`jobs.py`) that the frontend
polls. The header is laid out in pipeline order — Load PDF · Chunk Text ·
Index/Embed · Generate QA · View QA · Eval — and the page view overlays chunk
boxes and QA questions on the rendered pages, shows each question's gold-chunk
rank under a selected eval (red on a miss), and opens a "full results" popover
that walks the retrieved chunks against the gold chunk in place.

  ```bash
  python app/pdf_viewer.py                                                   # dev server
  PDF_VIEWER_PROD=1 gunicorn -w 1 --threads 16 -b 127.0.0.1:5001 app.pdf_viewer:app   # prod
  ```

## Development and Iteration Process

Iteration is tracked in [iteration_log.md](iteration_log.md), which records each
experiment — the change, hypothesis, before/after metrics, and keep/reject
decision. The reference targets throughout are **MRR ≥ 0.85** and
**Recall@5 ≥ 0.90**. The reference dataset committed to the repo
(`data/pdfplumber/20260715_01/uk_knowledge_and_innovation_analysis`) lets the
key numbers below be re-inspected from the committed `evaluations/` files. (The
rendered charts referenced below live under `visualizations/` and are gitignored
local artifacts, not committed — regenerate them with
`generate_visualizations.py`.)

### The QA benchmark had to be fixed before the metrics meant anything

The very first full evaluation (Iteration 1, 60 QA pairs on `attitudes_to_housing`)
came out terrible across the board — **Recall@5 ≤ 0.65 everywhere**, best-case
`text-embedding-3-large` at MRR 0.43 / R@5 0.60, and BM25 collapsing to MRR 0.06
/ R@5 0.10 on paraphrased questions
(`visualizations/first_run_terrible_results/`). The diagnosis was that the
*benchmark itself* was broken: many generated questions asked about "the chunk"
rather than naming a topic (*"What is the base sample size reported in the
chunk?"*), and others were forced out of table-of-contents / page-number
fragments with no answerable content (`visualizations/using_viewer_find_bad_questions/Example_bad_questions.png`).
Such questions can't be matched to a source chunk, so the scores measured prompt
noise, not retrieval.

The fix (Iterations 2–4) moved QA prompts into versioned files, forbade
chunk-referencing and metadata questions, and gave the model a structured
`InsufficientInformation` rejection path plus resampling. **Controlled A/B**
holding chunking, model, chunks, and indexes constant and varying *only* the
prompt confirmed the win — e.g. on the committed UK-doc run
(`eval_20260715_221153` = v2 vs `eval_20260715_222840` = v3, 57 questions,
top-k 5):

| Index (v2 → v3) | MRR | Recall@5 | NDCG@5 |
| --- | --- | --- | --- |
| BM25 / word | 0.501 → **0.577** | 0.614 → **0.702** | +0.080 |
| 3-large (chromadb/milvus) | 0.738 → **0.777** | 0.877 → **0.895** | +0.032 |
| 3-small | −0.001 (flat) | 0.842 → **0.877** | +0.008 |

v3 wins or ties everywhere and nothing regresses; the gain is largest for BM25,
confirming that anchoring questions to distinctive content matters most for
keyword search. A key lesson recorded in the log: Iteration 1's inflated shallow
recall was partly an artifact of "bad" questions quoting chunk text verbatim (a
keyword gift to retrieval), so the honest, harder benchmark is the more accurate
one. Iteration 8 (prompt v4) later stopped the model from wrongly rejecting
mechanically truncated sentence-chunk edges.

### Structure-aware chunking is what finally cleared the targets

With a sound benchmark, retrieval was still well below target (best R@5 ≈ 0.67),
and failure analysis showed ~half of the remaining misses were *structural*:
near-identical survey-table chunks whose distinguishing detail (the numbers)
lives in the answer, and overlapping fixed-size chunks duplicating a fact into a
neighbor that then outranks the gold chunk.

Two changes addressed this. **Hybrid retrieval** (Iteration 5) fused per-list
min-max-normalized BM25 + vector scores; as the reference warns, hybrid never
beat pure vector on these table-heavy docs — but for the legitimate reason
(BM25's weaker signal dilutes vector's), not the broken-normalization collapse
to ~0–6% MRR, which the unit-tested normalization avoids. The decisive change
was **`plumber-struct` chunking** (Iteration 6): chunking each detected table
into header-labeled row chunks ("Year: 2012; Rate: 45%") instead of letting
pysbd shred flattened tables into low-signal fragments. On the UK innovation doc
(29 pages → 134 chunks: 127 text / 6 table / 1 image, 12 questions, seed 7):

| Config (3-small vector) | MRR | Recall@10 | NDCG@10 |
| --- | --- | --- | --- |
| fixed_size 256/50 | 0.776 | — | — |
| **plumber-struct** | **0.958** | **1.000** | **0.969** |

This was the first configuration to clear the **MRR ≥ 0.85 / Recall@5 ≥ 0.90**
project targets. The grid that produced these comparisons is charted by
`generate_visualizations.py` (`visualizations/20260716_0837/` —
`mrr_comparison.png`, `chunking_strategy_heatmap_mrr.png`, `question_type_mrr.png`,
`time_vs_quality.png`, etc.).

### Table detection had to be fixed for `plumber-struct` to have tables to chunk

`plumber-struct` is only as good as the tables the extractor finds, and on the
UK doc pdfplumber's default `lines` strategy failed at both ends: it flagged
chart axis-frames as huge near-empty "tables" and missed the real
horizontal-ruled, whitespace-columned data tables — **10 detected tables, all
chart false positives, 0 real tables**
(`visualizations/Enhanced_nonbordered_table_detection_pdfplumber/Before_enhancements_table_not_detected.png`).
Iteration 7 added a **fill guard** (drop any detected table under 25% non-empty
cells; chart frames run 0–10% filled, genuine tables 50–100%) and a **rule-rect
detector** that reconstructs open-style tables from the page's thin rule-rects
and word baselines. The UK doc went from **10 fake / 0 real** to **0 fake / 8
real tables** (`After_enhancements.png`); a regression diff on the 180-page
`attitudes_to_housing` extraction changed only 8 pages, every one an all-empty
junk grid being dropped, with no content-bearing table altered — additive where
default detection already worked.

### The viewer as a diagnostic tool

Several of the findings above were spotted in the Flask viewer rather than in
charts, and it grew alongside the pipeline: it overlays QA questions and chunk
boxes on the page images, marks each question's gold-chunk rank under a selected
eval (red on a miss), and opens a per-question "full results" popover that lists
the retrieved chunks with scores and scrolls them against the gold chunk in
place — which is how the table-lookalike misses and the chunk-referencing bad
questions were first seen. It can also run every stage (chunk, index, QA,
eval) from its header, so the whole iterate-and-re-eval loop happens without
leaving the browser.
