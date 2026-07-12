#!/usr/bin/env python3
"""Generate grounded QA pairs from a chunk run to evaluate retrieval quality.

Scans data/{pdf2image,pdfplumber}/<date>/<title>/<chunk dir>/ for chunk runs
(same layout as embed_chunks.py), presents a menu (or takes --dataset), then
randomly samples chunks and uses the instructor library to generate three
questions per chunk, each answerable from that chunk alone:

  direct      - straightforward question using the chunk's own wording
  inference   - requires inferring from the chunk's information; avoids the
                chunk's exact words to test understanding of meaning
  paraphrased - same style of question as direct, but keywords are replaced
                with paraphrases/synonyms

Output goes into the chunk run dir as qa_<datetime>_<model>.json and records
each question with the source chunk, chunk_index, prompts used, and metadata,
so top-k retrieval (BM25 or embeddings) can later be scored against the
originating chunk id.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai_client.openai_client import MyOpenAIClient  # noqa: E402
from embed_chunks import scan_chunk_runs, choose_chunk_run, load_chunks  # noqa: E402

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_NUM_CHUNKS = 20

QUESTION_TYPES = {
    "direct": "Straightforward question answerable from the chunk, may use the chunk's wording.",
    "inference": "Requires inference from the chunk's information; avoids the chunk's exact "
                 "words so retrieval must match meaning, not keywords.",
    "paraphrased": "Same style as the direct question, but the chunk's keywords are replaced "
                   "with paraphrases or synonyms.",
}

SYSTEM_PROMPT = """\
You write question/answer pairs to evaluate a retrieval system over a chunked document.
You are given one text chunk. Every question you write must be answerable using ONLY the
information in that chunk - no outside knowledge, and nothing that depends on other parts
of the document. Questions must be specific enough that this chunk is clearly the best
source for the answer (avoid generic questions many chunks could answer).

Produce exactly three question/answer pairs:
1. direct: a straightforward factual question; it may reuse the chunk's own wording.
2. inference: a question whose answer must be inferred or synthesized from information in
   the chunk. Do NOT reuse the chunk's distinctive words or phrases; test understanding of
   meaning rather than keyword matching.
3. paraphrased: ask about the same kind of fact as a direct question would, but replace the
   chunk's keywords and distinctive terms with paraphrases or synonyms throughout.

Answers must be short, correct, and grounded in the chunk text.
"""

USER_PROMPT_TEMPLATE = """\
Generate the three question/answer pairs for this chunk:

<chunk>
{chunk_text}
</chunk>
"""


class QAPair(BaseModel):
    question: str = Field(description="The question, answerable from the chunk alone")
    answer: str = Field(description="Short answer grounded in the chunk text")


class ChunkQuestions(BaseModel):
    """Exactly one QA pair of each required type for a single chunk."""

    direct: QAPair = Field(description=QUESTION_TYPES["direct"])
    inference: QAPair = Field(description=QUESTION_TYPES["inference"])
    paraphrased: QAPair = Field(description=QUESTION_TYPES["paraphrased"])


def generate_for_chunk(client, model, temperature, chunk):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_retries=2,
        response_model=ChunkQuestions,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])},
        ],
    )
    return [
        {"question_type": qtype, "question": pair.question, "answer": pair.answer}
        for qtype, pair in (("direct", resp.direct),
                            ("inference", resp.inference),
                            ("paraphrased", resp.paraphrased))
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from a chunk run.")
    parser.add_argument("--dataset", help="Chunk run path (data/.../<title>/<chunk dir>) or list number")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Chat model (default {DEFAULT_MODEL})")
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS,
                        help=f"Chunks to sample (default {DEFAULT_NUM_CHUNKS})")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible chunk sampling")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default 0.7)")
    args = parser.parse_args()

    if args.num_chunks <= 0:
        sys.exit("ERROR: --num-chunks must be > 0")

    try:
        import instructor
    except ImportError:
        sys.exit("ERROR: instructor not installed. Run: pip install instructor")

    runs, empty = scan_chunk_runs()
    run = choose_chunk_run(runs, empty, preselect=args.dataset)
    chunk_meta, chunks = load_chunks(run)

    num = min(args.num_chunks, len(chunks))
    if num < args.num_chunks:
        print(f"WARNING: only {len(chunks)} chunks available; sampling all of them")
    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(chunks, num), key=lambda c: c["chunk_index"])
    print(f"Sampled {num} of {len(chunks)} chunks"
          + (f" (seed {args.seed})" if args.seed is not None else ""))

    api = MyOpenAIClient(model=args.model, temperature=args.temperature)
    api.validate_api_key()
    client = instructor.from_openai(api.get_client())

    items = []
    failures = []
    for i, chunk in enumerate(sampled, 1):
        print(f"  [{i}/{num}] chunk {chunk['chunk_index']} ...", end=" ", flush=True)
        try:
            qa_pairs = generate_for_chunk(client, args.model, args.temperature, chunk)
        except Exception as e:  # noqa: BLE001 - record and continue with other chunks
            print(f"FAILED: {e}")
            failures.append({"chunk_index": chunk["chunk_index"], "error": str(e)})
            continue
        print("ok")
        items.append({
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["text"],
            **{k: chunk[k] for k in ("start_char", "end_char", "start_page", "end_page")
               if k in chunk},
            "qa_pairs": qa_pairs,
        })

    if not items:
        sys.exit("ERROR: all chunks failed; no QA file written")

    now = datetime.now()
    result = {
        "metadata": {
            "datetime": now.isoformat(timespec="seconds"),
            "chunk_run": run["rel"],
            "chunk_metadata": chunk_meta,
            "model": args.model,
            "temperature": args.temperature,
            "seed": args.seed,
            "num_chunks_requested": args.num_chunks,
            "num_chunks_sampled": num,
            "num_chunks_answered": len(items),
            "questions_per_chunk": len(QUESTION_TYPES),
            "question_types": QUESTION_TYPES,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT_TEMPLATE,
            "failures": failures,
        },
        "items": items,
    }

    out_file = run["path"] / f"qa_{now.strftime('%Y%m%d_%H%M%S')}_{args.model}.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    total_q = sum(len(it["qa_pairs"]) for it in items)
    print(f"\nWrote {total_q} questions from {len(items)} chunks"
          + (f" ({len(failures)} chunk(s) failed)" if failures else ""))
    print(f"Output: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
