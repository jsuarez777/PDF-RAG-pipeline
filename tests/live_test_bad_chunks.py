#!/usr/bin/env python3
"""Regression-test the QA prompts against chunks that previously produced bad questions.

Scans data/**/qa_*.json for QA pairs whose question or answer matches a pattern
(default: the word "chunk"), collects the source chunks, and re-runs each one
through generate_qa.generate_for_chunk with the CURRENT prompts and response
model. For each chunk it reports whether the new output:

  CLEAN    - three questions, none matching the pattern
  STILL BAD- at least one new question/answer matches the pattern
  REJECTED - the model returned InsufficientInformation

Use --limit to keep API costs down while iterating on the prompt.

This is a LIVE test (real API calls, costs money) and is deliberately named so
pytest does NOT collect it (only test_*.py / *_test.py files are). Run it manually:

    python tests/live_test_bad_chunks.py [--limit N] [--out results.json]

Results are written to tests/live_llm_output/ unless --out says otherwise.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import generate_qa as gq  # noqa: E402
from logging_utils import setup_logging  # noqa: E402

from openai_client.openai_client import MyOpenAIClient  # noqa: E402

log = logging.getLogger(__name__)


def find_bad_items(pattern):
    """Yield one entry per offending chunk across all qa_*.json files (deduped by text)."""
    rx = re.compile(pattern, re.IGNORECASE)
    seen = set()
    found = []
    for f in sorted((ROOT / "data").rglob("qa_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            bad = [p for p in item["qa_pairs"]
                   if rx.search(p["question"]) or rx.search(p["answer"])]
            if not bad or item["chunk_text"] in seen:
                continue
            seen.add(item["chunk_text"])
            found.append({
                "source": str(f.relative_to(ROOT)),
                "chunk_index": item["chunk_index"],
                "chunk_text": item["chunk_text"],
                "old_bad": bad,
            })
    return found


def main():
    parser = argparse.ArgumentParser(description="Re-run previously bad chunks through the QA prompts.")
    parser.add_argument("--pattern", default=r"\bchunks?\b",
                        help=r"Regex marking a bad question/answer (default \bchunks?\b)")
    parser.add_argument("--model", default=gq.DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, help="Test at most N chunks")
    parser.add_argument("--out", help="Write full results to this JSON file (default: "
                                      "tests/live_llm_output/bad_chunk_results_<datetime>.json)")
    args = parser.parse_args()

    setup_logging("live_test_bad_chunks")

    try:
        import instructor
    except ImportError:
        sys.exit("ERROR: instructor not installed. Run: pip install instructor")

    bad_items = find_bad_items(args.pattern)
    if not bad_items:
        sys.exit(f"No QA pairs matching /{args.pattern}/ found in data/**/qa_*.json")
    log.info(f"Found {len(bad_items)} chunk(s) with bad QA pairs matching /{args.pattern}/")
    if args.limit:
        bad_items = bad_items[:args.limit]
        log.info(f"Testing first {len(bad_items)} (--limit)")

    api = MyOpenAIClient(model=args.model, temperature=args.temperature)
    api.validate_api_key()
    client = instructor.from_openai(api.get_client())

    rx = re.compile(args.pattern, re.IGNORECASE)
    counts = {"CLEAN": 0, "STILL BAD": 0, "REJECTED": 0, "FAILED": 0}
    results = []
    for i, item in enumerate(bad_items, 1):
        log.info(f"\n[{i}/{len(bad_items)}] chunk {item['chunk_index']} from {item['source']}")
        for p in item["old_bad"]:
            log.info(f"  old {p['question_type']}: {p['question']}")
        try:
            resp = gq.generate_for_chunk(client, args.model, args.temperature,
                                         {"text": item["chunk_text"]})
        except Exception as e:  # noqa: BLE001 - keep testing the other chunks
            verdict = "FAILED"
            log.info(f"  -> FAILED: {e}")
            results.append({**item, "verdict": verdict, "error": str(e)})
            counts[verdict] += 1
            continue
        if isinstance(resp, gq.InsufficientInformation):
            verdict = "REJECTED"
            log.info(f"  -> REJECTED: {resp.reason}")
            results.append({**item, "verdict": verdict, "reason": resp.reason})
        else:
            new_pairs = gq.qa_pairs_from(resp)
            still_bad = [p for p in new_pairs if rx.search(p["question"]) or rx.search(p["answer"])]
            verdict = "STILL BAD" if still_bad else "CLEAN"
            log.info(f"  -> {verdict}")
            for p in new_pairs:
                flag = " [BAD]" if p in still_bad else ""
                log.info(f"     new {p['question_type']}{flag}: {p['question']}")
                log.info(f"         A: {p['answer']}")
            results.append({**item, "verdict": verdict, "new_pairs": new_pairs})
        counts[verdict] += 1

    log.info("\n=== Summary ===")
    for verdict, n in counts.items():
        if n:
            log.info(f"  {verdict}: {n}")

    if args.out:
        out = Path(args.out)
    else:
        from datetime import datetime
        out_dir = ROOT / "tests" / "live_llm_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"bad_chunk_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"pattern": args.pattern, "model": args.model,
                               "counts": counts, "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Results written to {out}")


if __name__ == "__main__":
    main()
