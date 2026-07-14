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
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed_chunks import choose_chunk_run, load_chunks, scan_chunk_runs  # noqa: E402
from logging_utils import setup_logging  # noqa: E402

from openai_client.openai_client import MyOpenAIClient  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_NUM_CHUNKS = 20
PROMPTS_DIR = ROOT / "prompts" / "generate_qa"

QUESTION_TYPES = {
    "direct": "Straightforward question answerable from the chunk, may use the chunk's wording.",
    "inference": "Requires inference from the chunk's information; avoids the chunk's exact "
                 "words so retrieval must match meaning, not keywords.",
    "paraphrased": "Same style as the direct question, but the chunk's keywords are replaced "
                   "with paraphrases or synonyms.",
}


def _latest_prompt_version(prompts_dir):
    versions = [d for d in prompts_dir.iterdir() if d.is_dir() and re.fullmatch(r"v\d+", d.name)]
    if not versions:
        raise FileNotFoundError(f"No prompt version directories (v0, v1, ...) found in {prompts_dir}")
    return max(versions, key=lambda d: int(d.name[1:])).name


def _read_prompt(version_dir, name):
    path = version_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Required prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


PROMPT_VERSION = _latest_prompt_version(PROMPTS_DIR)
SYSTEM_PROMPT = _read_prompt(PROMPTS_DIR / PROMPT_VERSION, "qa_system.prompt")
USER_PROMPT_TEMPLATE = _read_prompt(PROMPTS_DIR / PROMPT_VERSION, "qa_user.template")


class QAPair(BaseModel):
    question: str = Field(description="The question, answerable from the chunk alone")
    answer: str = Field(description="Short answer grounded in the chunk text")


class ChunkQuestions(BaseModel):
    """Exactly one QA pair of each required type for a single chunk."""

    direct: QAPair = Field(description=QUESTION_TYPES["direct"])
    inference: QAPair = Field(description=QUESTION_TYPES["inference"])
    paraphrased: QAPair = Field(description=QUESTION_TYPES["paraphrased"])


class InsufficientInformation(BaseModel):
    """Use ONLY when the chunk lacks enough substantive content for grounded QA pairs,
    e.g. it is just a page header/footer, a bare table-of-contents fragment, a reference
    list, or boilerplate with no concrete facts."""

    insufficient_information: Literal[True]
    reason: str = Field(description="Why the chunk cannot support specific, answerable questions")


def generate_for_chunk(client, model, temperature, chunk):
    """Returns ChunkQuestions, or InsufficientInformation if the model rejects the chunk."""
    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_retries=2,
        response_model=Union[ChunkQuestions, InsufficientInformation],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk_text=chunk["text"])},
        ],
    )


def qa_pairs_from(resp):
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
    parser.add_argument("--chunks", help="Comma-separated chunk indices to use instead of "
                                         "random sampling (e.g. 3,17,42)")
    parser.add_argument("--types", help="Comma-separated subset of question types to keep "
                                        f"(default: all of {','.join(QUESTION_TYPES)})")
    parser.add_argument("--add", help="Existing qa_*.json to add the generated items to, "
                                      "instead of writing a new file. Items are kept sorted "
                                      "by chunk index, and an existing item for the same "
                                      "chunk is replaced rather than duplicated")
    args = parser.parse_args()

    setup_logging("generate_qa")

    if not args.chunks and args.num_chunks <= 0:
        sys.exit("ERROR: --num-chunks must be > 0")

    keep_types = list(QUESTION_TYPES)
    if args.types:
        keep_types = [t.strip() for t in args.types.split(",") if t.strip()]
        unknown = [t for t in keep_types if t not in QUESTION_TYPES]
        if unknown:
            sys.exit(f"ERROR: unknown question type(s): {', '.join(unknown)}")

    try:
        import instructor
    except ImportError:
        sys.exit("ERROR: instructor not installed. Run: pip install instructor")

    runs, empty = scan_chunk_runs()
    run = choose_chunk_run(runs, empty, preselect=args.dataset)
    chunk_meta, chunks = load_chunks(run)

    rng = None  # set in random-sampling mode; enables resampling after rejections
    pool = []
    if args.chunks:
        try:
            wanted = sorted({int(x) for x in args.chunks.split(",") if x.strip()})
        except ValueError:
            sys.exit("ERROR: --chunks must be comma-separated integers")
        if not wanted:
            sys.exit("ERROR: --chunks is empty")
        by_index = {c["chunk_index"]: c for c in chunks}
        missing = [i for i in wanted if i not in by_index]
        if missing:
            sys.exit(f"ERROR: chunk index(es) not in this run: {missing}")
        sampled = [by_index[i] for i in wanted]
        num = len(sampled)
        log.info(f"Using {num} chunk(s) from --chunks: {', '.join(map(str, wanted))}")
    else:
        num = min(args.num_chunks, len(chunks))
        if num < args.num_chunks:
            log.warning(f"WARNING: only {len(chunks)} chunks available; sampling all of them")
        rng = random.Random(args.seed)
        sampled = sorted(rng.sample(chunks, num), key=lambda c: c["chunk_index"])
        sampled_indices = {c["chunk_index"] for c in sampled}
        pool = [c for c in chunks if c["chunk_index"] not in sampled_indices]
        log.info(f"Sampled {num} of {len(chunks)} chunks"
              + (f" (seed {args.seed})" if args.seed is not None else ""))

    api = MyOpenAIClient(model=args.model, temperature=args.temperature)
    api.validate_api_key()
    client = instructor.from_openai(api.get_client())

    items = []
    failures = []
    rejected = []
    queue = list(sampled)
    while queue:
        chunk = queue.pop(0)
        prefix = f"  [{len(items) + 1}/{num}] chunk {chunk['chunk_index']} ..."
        try:
            resp = generate_for_chunk(client, args.model, args.temperature, chunk)
        except Exception as e:  # noqa: BLE001 - record and continue with other chunks
            log.info(f"{prefix} FAILED: {e}")
            failures.append({"chunk_index": chunk["chunk_index"], "error": str(e)})
            continue
        if isinstance(resp, InsufficientInformation):
            log.info(f"{prefix} rejected: {resp.reason}")
            rejected.append({"chunk_index": chunk["chunk_index"], "reason": resp.reason})
            # random-sampling mode: draw a replacement so the QA set stays at --num-chunks
            if rng is not None and pool:
                replacement = pool.pop(rng.randrange(len(pool)))
                queue.append(replacement)
                log.info(f"           resampled chunk {replacement['chunk_index']} as replacement")
            continue
        log.info(f"{prefix} ok")
        items.append({
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["text"],
            **{k: chunk[k] for k in ("start_char", "end_char", "start_page", "end_page")
               if k in chunk},
            # the model always produces all three types; keep only the requested ones
            "qa_pairs": [p for p in qa_pairs_from(resp) if p["question_type"] in keep_types],
        })
    items.sort(key=lambda it: it["chunk_index"])

    if not items:
        sys.exit("ERROR: all chunks failed or were rejected; no QA file written")

    now = datetime.now()

    if args.add:
        out_file = Path(args.add)
        try:
            existing = json.loads(out_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            sys.exit(f"ERROR: could not read --add file {out_file}: {e}")
        if existing.get("metadata", {}).get("chunk_run") != run["rel"]:
            sys.exit(f"ERROR: --add file belongs to chunk run "
                     f"'{existing.get('metadata', {}).get('chunk_run')}', not '{run['rel']}'")
        old_items = existing.setdefault("items", [])
        new_indices = {it["chunk_index"] for it in items}
        replaced = sorted({it["chunk_index"] for it in old_items
                           if it["chunk_index"] in new_indices})
        old_items[:] = [it for it in old_items if it["chunk_index"] not in new_indices]
        old_items.extend(items)
        old_items.sort(key=lambda it: it["chunk_index"])
        meta = existing.setdefault("metadata", {})
        meta["num_chunks_answered"] = len(old_items)
        meta.setdefault("adds", []).append({
            "datetime": now.isoformat(timespec="seconds"),
            "model": args.model,
            "temperature": args.temperature,
            "chunks": [it["chunk_index"] for it in items],
            "replaced": replaced,
            "question_types": keep_types,
            "rejected": rejected,
            "failures": failures,
        })
        out_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        total_q = sum(len(it["qa_pairs"]) for it in items)
        log.info(f"\nAdded {total_q} question(s) from {len(items)} chunk(s)"
              + (f", replacing existing items for chunk(s) {replaced}" if replaced else "")
              + (f" ({len(rejected)} chunk(s) rejected)" if rejected else "")
              + (f" ({len(failures)} chunk(s) failed)" if failures else ""))
        log.info(f"Output: {out_file.relative_to(ROOT) if out_file.is_relative_to(ROOT) else out_file}")
        return

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
            "questions_per_chunk": len(keep_types),
            "question_types": {t: QUESTION_TYPES[t] for t in keep_types},
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT_TEMPLATE,
            "rejected": rejected,
            "failures": failures,
        },
        "items": items,
    }

    out_file = run["path"] / f"qa_{now.strftime('%Y%m%d_%H%M%S')}_{args.model}.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    total_q = sum(len(it["qa_pairs"]) for it in items)
    log.info(f"\nWrote {total_q} questions from {len(items)} chunks"
          + (f" ({len(rejected)} chunk(s) rejected)" if rejected else "")
          + (f" ({len(failures)} chunk(s) failed)" if failures else ""))
    log.info(f"Output: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
