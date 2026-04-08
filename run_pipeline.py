#!/usr/bin/env python3
"""
MCDataGen Pipeline Runner

Orchestrates all phases:
  Phase 1: MC conversation generation + inline evaluation (async vLLM)
  Phase 2: (standalone) Re-evaluate existing Phase 1 output (async vLLM)

The pipeline is intentionally single-phase for the generation path:
phase1_mc_gen.py generates each assistant response and immediately evaluates
it with phase2_mc_eval.evaluate_record(). If the quality is below
PASSING_QUALITY (default: "excelente") the response is regenerated up to
MAX_RETRIES times. The best attempt is always written to disk.

Usage:
    # Full pipeline (generation + inline evaluation)
    python -m MCDataGen.run_pipeline --num-rows 10000

    # Phase 1 only (generation)
    python -m MCDataGen.run_pipeline --phase 1

    # Phase 2 only (re-evaluate an existing file)
    python -m MCDataGen.run_pipeline --phase 2 --input mc_phase1_conversations.jsonl

    # Resume from a specific phase
    python -m MCDataGen.run_pipeline --start-phase 1
"""

import argparse
import asyncio
import os
import sys

from MCDataGen.config import (
    BATCH_SIZE,
    NUM_ROWS,
    PASSING_QUALITY,
    MAX_RETRIES,
    PHASE1_OUTPUT,
    PHASE2_OUTPUT,
)


# -----------------------------------------------------------------------
# Phase runners
# -----------------------------------------------------------------------
async def run_phase1(num_rows: int, batch_size: int, output: str):
    print("\n" + "=" * 60)
    print("PHASE 1: MC Conversation Generation + Inline Evaluation")
    print("=" * 60)
    from MCDataGen.phase1_mc_gen import main as p1

    await p1(output_file=output, num_rows=num_rows, batch_size=batch_size)


async def run_phase2(input_file: str, batch_size: int, output: str):
    print("\n" + "=" * 60)
    print("PHASE 2: MC Quality Re-Evaluation")
    print("=" * 60)
    from MCDataGen.phase2_mc_eval import evaluate_record, QUALITY_ORDER
    from MCDataGen.utils import (
        iter_jsonl_batches,
        write_jsonl_batch,
        count_jsonl_lines,
        get_last_processed_id,
        get_max_int_field,
    )
    from openai import AsyncOpenAI
    from MCDataGen.config import VLLM_BASE_URL, VLLM_API_KEY
    from tqdm import tqdm

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    models = await client.models.list()
    model_id = models.data[0].id
    print(f"Phase 2 — model: {model_id}")

    last_out_id = get_last_processed_id(output)
    next_id = last_out_id + 1 if last_out_id >= 0 else 0

    last_input_id = get_max_int_field(output, "input_id")
    resume_from_input = last_input_id + 1 if last_input_id >= 0 else 0

    total = count_jsonl_lines(input_file)
    pbar = tqdm(total=total, desc="Phase 2: re-evaluation")

    import asyncio as _asyncio
    sem = _asyncio.Semaphore(batch_size)

    total_seen = 0
    total_kept = 0

    for batch in iter_jsonl_batches(
        input_file,
        batch_size=batch_size,
        start_from_id=resume_from_input,
        required_fields=["conversation", "response_style", "gold_answer"],
    ):
        total_seen += len(batch)

        async def _eval_one(item):
            async with sem:
                return await evaluate_record(item, model_id)

        results = await _asyncio.gather(*[_eval_one(it) for it in batch])

        rows = []
        for item, (explicacao, qualidade) in zip(batch, results):
            item["input_id"] = item["id"]
            item["id"] = next_id
            item["qualidade"] = qualidade
            item["explicacao_qualidade"] = explicacao
            rows.append(item)
            next_id += 1

        if rows:
            write_jsonl_batch(output, rows)
            total_kept += len(rows)

        pbar.update(len(batch))

    pbar.close()
    print(f"Phase 2 done — {total_kept} records evaluated → {output}")


# -----------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------
async def run_full_pipeline(
    num_rows: int,
    batch_size: int,
    start_phase: int = 1,
    p1_out: str = PHASE1_OUTPUT,
    p2_out: str = PHASE2_OUTPUT,
):
    if start_phase <= 1:
        await run_phase1(num_rows, batch_size, p1_out)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Final output: {p1_out}")


# -----------------------------------------------------------------------
# Single phase
# -----------------------------------------------------------------------
async def run_single_phase(
    phase: int,
    input_file: str | None,
    output_file: str | None,
    batch_size: int,
    num_rows: int,
):
    if phase == 1:
        await run_phase1(num_rows, batch_size, output_file or PHASE1_OUTPUT)
    elif phase == 2:
        await run_phase2(
            input_file or PHASE1_OUTPUT,
            batch_size,
            output_file or PHASE2_OUTPUT,
        )
    else:
        print(f"Invalid phase: {phase}. Must be 1 or 2.")
        sys.exit(1)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MCDataGen Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python -m MCDataGen.run_pipeline --num-rows 10000
    python -m MCDataGen.run_pipeline --phase 1 --num-rows 5000 --output gen.jsonl
    python -m MCDataGen.run_pipeline --phase 2 --input mc_phase1_conversations.jsonl
    python -m MCDataGen.run_pipeline --passing-quality boa
""",
    )
    parser.add_argument("--phase", type=int, choices=[1, 2])
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--input", type=str, help="Override input file for the phase")
    parser.add_argument("--output", type=str, help="Override output file for the phase")
    parser.add_argument("--num-rows", type=int, default=NUM_ROWS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--passing-quality",
        type=str,
        default=PASSING_QUALITY,
        choices=["muito ruim", "ruim", "média", "boa", "excelente"],
        help=f"Minimum quality to accept without retrying (default: {PASSING_QUALITY})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Max regeneration attempts per sample (default: {MAX_RETRIES})",
    )

    args = parser.parse_args()

    # Propagate CLI overrides into env so phase modules pick them up
    os.environ["PASSING_QUALITY"] = args.passing_quality
    os.environ["MAX_RETRIES"] = str(args.max_retries)

    print("=" * 60)
    print("MCDataGen Pipeline")
    print("=" * 60)
    print(f"vLLM URL        : {os.environ.get('VLLM_BASE_URL', 'http://localhost:8020/v1')}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Num rows        : {args.num_rows}")
    print(f"Passing quality : {args.passing_quality}")
    print(f"Max retries     : {args.max_retries}")

    if args.phase:
        print(f"Running: Phase {args.phase} only")
        asyncio.run(run_single_phase(
            phase=args.phase,
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size,
            num_rows=args.num_rows,
        ))
    else:
        print(f"Running: Phase {args.start_phase} onwards")
        asyncio.run(run_full_pipeline(
            num_rows=args.num_rows,
            batch_size=args.batch_size,
            start_phase=args.start_phase,
        ))


if __name__ == "__main__":
    main()
