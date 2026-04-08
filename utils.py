"""
Shared utilities for the MCDataGen pipeline.

JSONL I/O with resume support and MC-specific helpers.
"""

import json
import os
from typing import Dict, Iterator, List, Optional, Set, Tuple


# -----------------------------------------------------------------------
# JSONL helpers
# -----------------------------------------------------------------------
def get_last_processed_id(filepath: str) -> int:
    """Return the highest `id` written to *filepath*, or -1 if empty/missing."""
    return get_max_int_field(filepath, "id")


def get_max_int_field(filepath: str, field: str) -> int:
    """Return the highest value of an integer *field* in a JSONL, or -1."""
    if not os.path.exists(filepath):
        return -1
    best = -1
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                val = obj.get(field)
                if isinstance(val, int) and val > best:
                    best = val
            except (json.JSONDecodeError, AttributeError):
                continue
    return best


def iter_jsonl_batches(
    path: str,
    batch_size: int,
    start_from_id: int = 0,
    required_fields: Optional[List[str]] = None,
) -> Iterator[List[Dict]]:
    """Yield batches of dicts from a JSONL, skipping rows with id < start_from_id."""
    required_fields = required_fields or []
    with open(path, "r", encoding="utf-8") as f:
        batch: List[Dict] = []
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(k in obj for k in required_fields):
                continue
            if not isinstance(obj.get("id"), int):
                continue
            if obj["id"] < start_from_id:
                continue
            batch.append(obj)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def write_jsonl_batch(filepath: str, batch: List[Dict]) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        for obj in batch:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl_line(filepath: str, obj: Dict) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def count_jsonl_lines(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# -----------------------------------------------------------------------
# MC dataset resume helpers
# -----------------------------------------------------------------------
def read_next_seq(path: str) -> int:
    """Compute the next seq_id by scanning the output JSONL."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    next_seq_id = 0
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "seq_id" in obj:
                next_seq_id = max(next_seq_id, int(obj["seq_id"]) + 1)
    return next_seq_id


def build_existing_index(path: str) -> Tuple[int, Set[int]]:
    """
    Scan output JSONL and build:
    - done_ctx: set of context_ids already written
    - first_incomplete_ctx: earliest context_id (dataset idx) NOT in done_ctx

    Each context_id gets exactly one style (sampled), so presence means done.
    """
    done: Set[int] = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0, done

    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ctx = obj.get("context_id")
            if ctx is None:
                continue
            try:
                done.add(int(ctx))
            except Exception:
                continue

    if not done:
        return 0, done

    expected = 0
    for ctx_id in sorted(done):
        if ctx_id > expected:
            return expected, done
        expected = ctx_id + 1

    return expected, done


def skip_until_idx(ds_iter, start_idx: int, idx_col_name: str) -> Optional[Dict]:
    """
    Advance a streaming iterator until record[idx_col_name] >= start_idx.

    Returns the first matching record so the caller can process it; iterators cannot
    be rewound, so without returning it, that record would be lost.
    """
    if start_idx <= 0:
        return None
    for rec in ds_iter:
        rid = rec.get(idx_col_name)
        if rid is None:
            continue
        try:
            rid_int = int(rid)
        except Exception:
            continue
        if rid_int >= start_idx:
            return rec
    return None
