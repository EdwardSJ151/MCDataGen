"""
Phase 1: MC Conversation Generation

For each record streamed from the MC dataset:
  1. Sample one interaction style by weighted percentage.
  2. Build the user message programmatically from the style template.
  3. Optionally rewrite the "frame" (non-question/choices text) with a persona.
  4. For "apenas_letra" styles the assistant answer is the gold letter (no LLM).
     For other styles the LLM generates the answer (gold hint in system prompt).
  5. Each generated response is immediately evaluated by phase2_mc_eval.evaluate_record().
     If the quality is below PASSING_QUALITY the response is regenerated (up to
     MAX_RETRIES attempts). The best attempt is always written to disk.

Output JSONL fields per record:
    seq_id, id, conversation, response_style, context_id, gold_answer,
    target_letter, level, subject, persona, qualidade, explicacao_qualidade

Resume: context_id set — same idx-based logic, skips already-written context_ids.
"""

import argparse
import asyncio
import json
import math
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

from openai import AsyncOpenAI, BadRequestError
from tqdm import tqdm

from MCDataGen.config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    MC_DATASET_ID,
    MC_DATASET_SUBSET,
    IDX_COL_NAME,
    USE_PERSONA,
    RESP_TEMPERATURE,
    RESP_TOP_P,
    RESP_MAX_TOKENS,
    REWRITE_TEMPERATURE,
    REWRITE_TOP_P,
    REWRITE_MAX_TOKENS,
    MAX_PERSONA_REWRITE_ATTEMPTS,
    MAX_RETRIES,
    NUM_ROWS,
    BATCH_SIZE,
    MAX_ASYNC_CHAT,
    MAX_ASYNC_QGEN,
    QUEUE_MAXSIZE,
    PHASE1_OUTPUT,
    STOP_STRINGS,
    STOP_TOKEN_IDS,
    LOGITS_PROCESSORS,
)
from MCDataGen.prompts_mc import (
    PERSONA_DATASET_ID,
    PERSONA_DATASET_SUBSET,
    PERSONA_TEXT_FIELD,
    PERSONA_REWRITE_SYSTEM,
    STYLE_CONFIGS,
    ALL_STYLES,
    STYLE_WEIGHTS,
)
from MCDataGen.utils import (
    build_existing_index,
    read_next_seq,
    skip_until_idx,
    write_jsonl_batch,
)

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

# Cycling counter for "is_correct" style (A→B→C→D→A→…)
_IS_CORRECT_LETTERS = ["A", "B", "C", "D"]
_is_correct_counter = 0


# =====================================================================
# Persona provider
# =====================================================================
class PersonaProvider:
    """
    Streams personas from a HuggingFace dataset.
    Auto-restarts when exhausted so we can generate more rows than personas.
    """

    def __init__(self, dataset_id: str, subset: str, text_field: str):
        self.dataset_id = dataset_id
        self.subset = subset
        self.text_field = text_field
        self._iterator: Optional[Iterator] = None
        self._current_index: int = -1

    def _get_iterator(self) -> Iterator:
        if self._iterator is None:
            from datasets import load_dataset
            ds = load_dataset(
                self.dataset_id,
                self.subset,
                split="train",
                streaming=True,
            )
            self._iterator = iter(ds)
        return self._iterator

    def get_persona(self) -> Dict[str, Any]:
        iterator = self._get_iterator()
        while True:
            try:
                record = next(iterator)
                self._current_index += 1
            except StopIteration:
                self._iterator = None
                self._current_index = -1
                iterator = self._get_iterator()
                record = next(iterator)
                self._current_index += 1

            persona_text = record.get(self.text_field, "")
            if not isinstance(persona_text, str) or not persona_text.strip():
                continue

            return {
                "persona": persona_text.strip(),
                "persona_index": self._current_index,
            }


def _get_persona_provider() -> PersonaProvider:
    return PersonaProvider(
        dataset_id=PERSONA_DATASET_ID,
        subset=PERSONA_DATASET_SUBSET,
        text_field=PERSONA_TEXT_FIELD,
    )


# =====================================================================
# Dataset helpers
# =====================================================================
def _mc_stream_iter():
    """Return a streaming iterator over the MC dataset."""
    from datasets import load_dataset
    return load_dataset(MC_DATASET_ID, MC_DATASET_SUBSET, split="train", streaming=True)


def _get_train_num_examples() -> int:
    """Get split size from dataset metadata."""
    from datasets import load_dataset_builder
    builder = load_dataset_builder(MC_DATASET_ID, MC_DATASET_SUBSET)
    splits = getattr(builder.info, "splits", None)
    if not splits or "train" not in splits:
        raise RuntimeError("Dataset metadata missing train split size.")
    n = getattr(splits["train"], "num_examples", None)
    if n is None:
        raise RuntimeError("Dataset train split size is None.")
    return int(n)


def _extract_mc_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and extract the fields we need from an MC record.
    Returns None if the record is malformed.
    """
    question = record.get("question", "")
    answer = record.get("answer", "")
    choices = record.get("choices")
    level = record.get("level", "")
    subject = record.get("subject", "")

    if not question or not answer or not choices:
        return None
    if not isinstance(choices, dict) or "label" not in choices or "text" not in choices:
        return None
    if len(choices["label"]) < 2:
        return None

    return {
        "question": question,
        "answer": answer,
        "choices": choices,
        "level": level,
        "subject": subject,
    }


# =====================================================================
# Message building helpers
# =====================================================================
def _format_choices(choices: Dict[str, Any]) -> str:
    """
    Format the choices dict into a readable string.
    Input:  {'label': ['A', 'B', ...], 'text': ['...', '...', ...]}
    Output: 'A) ...\nB) ...\n...'
    """
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    return "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, texts))


def _get_choice_text(choices: Dict[str, Any], letter: str) -> str:
    """Return the text of a specific alternative by its letter."""
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    for lbl, txt in zip(labels, texts):
        if lbl == letter:
            return txt
    return ""


def _next_is_correct_letter() -> str:
    """Return the next letter in the A→B→C→D cycle for is_correct prompts."""
    global _is_correct_counter
    letter = _IS_CORRECT_LETTERS[_is_correct_counter % len(_IS_CORRECT_LETTERS)]
    _is_correct_counter += 1
    return letter


def _strip_protection_markers(text: str) -> str:
    """Remove the «INÍCIO_PROTEGIDO» and «FIM_PROTEGIDO» markers from text."""
    text = text.replace("«INÍCIO_PROTEGIDO»\n", "")
    text = text.replace("«INÍCIO_PROTEGIDO»", "")
    text = text.replace("\n«FIM_PROTEGIDO»", "")
    text = text.replace("«FIM_PROTEGIDO»", "")
    return text


def _split_frame_and_protected(text: str) -> Tuple[List[str], str]:
    """
    Split a marked-up user message into frame segments and the protected block.

    Returns (frame_parts, protected_content) where:
      - frame_parts: list of text chunks outside the markers (before and after)
      - protected_content: the text between «INÍCIO_PROTEGIDO» and «FIM_PROTEGIDO»
    """
    start_marker = "«INÍCIO_PROTEGIDO»"
    end_marker = "«FIM_PROTEGIDO»"

    s = text.find(start_marker)
    e = text.find(end_marker)

    if s == -1 or e == -1:
        return [text], ""

    before = text[:s].rstrip("\n")
    after = text[e + len(end_marker):].lstrip("\n")
    protected = text[s + len(start_marker):e].strip("\n")

    frame_parts = [p for p in [before, after] if p.strip()]
    return frame_parts, protected


def sample_style() -> str:
    """Sample one style using the configured percentage weights."""
    return random.choices(ALL_STYLES, weights=STYLE_WEIGHTS, k=1)[0]


def build_user_message(
    style: str,
    question: str,
    choices: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    """
    Build the user message programmatically from the style template.

    Returns:
        (user_message_with_markers, target_letter_or_None)

    The returned message still contains «INÍCIO_PROTEGIDO»/«FIM_PROTEGIDO»
    markers so the persona rewriter knows what to preserve.  Call
    _strip_protection_markers() after rewriting (or if no persona is used).
    """
    cfg = STYLE_CONFIGS[style]
    choices_block = _format_choices(choices)

    target_letter: Optional[str] = None
    target_text = ""

    if style == "is_correct":
        target_letter = _next_is_correct_letter()
        target_text = _get_choice_text(choices, target_letter)
        if not target_text:
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if labels and texts:
                target_letter = labels[0]
                target_text = texts[0]

    msg = cfg["frame"].format(
        question=question,
        choices_block=choices_block,
        target_letter=target_letter or "",
        target_text=target_text or "",
    )

    return msg, target_letter


# =====================================================================
# LLM calls
# =====================================================================
async def _llm_call(
    messages: List[Dict[str, str]],
    model_id: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_thinking: bool = False,
) -> str:
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "stop": STOP_STRINGS,
        "stop_token_ids": STOP_TOKEN_IDS,
        "logits_processors": LOGITS_PROCESSORS,
    }
    resp = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content or ""


async def safe_llm_call(*args, **kwargs) -> Optional[str]:
    """Swallows exceptions and returns None on failure. Retries with thinking enabled on BadRequestError."""
    try:
        return await _llm_call(*args, **kwargs)
    except BadRequestError:
        try:
            return await _llm_call(*args, **{**kwargs, "enable_thinking": True})
        except Exception:
            return None
    except Exception:
        return None


# =====================================================================
# Persona rewrite (only the frame, not the question/choices)
# =====================================================================
async def rewrite_with_persona(
    model_id: str,
    user_message_with_markers: str,
    persona: str,
) -> str:
    """
    Rewrite ONLY the frame (instruction) portions of the user message using
    persona as a diversity seed. The question and choices are NEVER sent to
    the LLM — they are stitched back programmatically, guaranteeing verbatim
    preservation.

    Returns the final user message with the original question+choices intact.
    Falls back to the original (stripped) if the LLM call fails.
    """
    frame_parts, protected = _split_frame_and_protected(user_message_with_markers)
    fallback = _strip_protection_markers(user_message_with_markers)

    if not frame_parts or not protected:
        return fallback

    frame_text = "\n".join(frame_parts)

    messages = [
        {"role": "system", "content": PERSONA_REWRITE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"PERSONA: {persona}\n\n"
                f"TEXTO DE INSTRUÇÃO:\n{frame_text}"
            ),
        },
    ]

    for _ in range(MAX_PERSONA_REWRITE_ATTEMPTS):
        result = await safe_llm_call(
            messages,
            model_id,
            temperature=REWRITE_TEMPERATURE,
            top_p=REWRITE_TOP_P,
            max_tokens=REWRITE_MAX_TOKENS,
        )

        if not result or not result.strip():
            continue

        rewritten_frame = result.strip()
        rewritten_frame = _strip_protection_markers(rewritten_frame)

        if protected in rewritten_frame:
            rewritten_frame = rewritten_frame.replace(protected, "").strip()

        if not rewritten_frame:
            continue

        return f"{rewritten_frame}\n\n{protected}"

    return fallback


# =====================================================================
# Generate assistant response
# =====================================================================
async def generate_assistant_response(
    model_id: str,
    style: str,
    user_message: str,
    answer_letter: str,
    correct_text: str,
) -> Optional[str]:
    """
    Generate the assistant turn.

    - programmatic_answer styles: return gold letter directly (no LLM call).
    - Other styles: call the LLM with the gold hint injected in the system prompt.
    """
    cfg = STYLE_CONFIGS[style]

    if cfg["programmatic_answer"]:
        return answer_letter

    base_system = cfg["system"]
    gold_hint = (
        f"\n\nA alternativa correta é: {answer_letter}) {correct_text}\n"
        "Use esta informação para fundamentar sua resposta."
    )
    system_prompt = base_system + gold_hint

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    return await safe_llm_call(
        messages,
        model_id,
        temperature=RESP_TEMPERATURE,
        top_p=RESP_TOP_P,
        max_tokens=RESP_MAX_TOKENS,
    )


# =====================================================================
# Main
# =====================================================================
async def main(
    output_file: str = PHASE1_OUTPUT,
    num_rows: int = NUM_ROWS,
    batch_size: int = BATCH_SIZE,
):
    """
    Main loop: for each MC record, sample one style and generate one
    single-turn conversation with inline evaluation.

    Concurrency:
    - Producers: build user messages (+ optional persona rewrite) and enqueue.
    - Consumers: generate assistant responses, evaluate, and write JSONL.
    """
    from MCDataGen.phase2_mc_eval import evaluate_record, quality_passes, QUALITY_ORDER

    models = await client.models.list()
    model_id = models.data[0].id
    print(f"Phase 1 — model: {model_id}")

    first_incomplete_ctx, done_ctx = build_existing_index(output_file)
    seq_id = read_next_seq(output_file)

    ds_iter = iter(_mc_stream_iter())
    pending_rec = skip_until_idx(ds_iter, first_incomplete_ctx, IDX_COL_NAME)

    dataset_n = _get_train_num_examples()

    total_remaining = max(0, num_rows - seq_id)
    remaining_ctx = max(0, dataset_n - first_incomplete_ctx)
    total_remaining = min(total_remaining, remaining_ctx)

    contexts_per_batch = max(1, min(batch_size, total_remaining))
    total_batches = (
        math.ceil(total_remaining / contexts_per_batch)
        if contexts_per_batch > 0 else 0
    )

    persona_provider = _get_persona_provider() if USE_PERSONA else None

    with open(output_file, "a", encoding="utf-8") as fout:
        pbar = tqdm(total=total_batches, desc="Phase 1: MC conversation generation")
        write_lock = asyncio.Lock()

        while seq_id < num_rows:
            batch_items: List[dict] = []

            while len(batch_items) < contexts_per_batch:
                if pending_rec is not None:
                    rec = pending_rec
                    pending_rec = None
                else:
                    try:
                        rec = next(ds_iter)
                    except StopIteration:
                        rec = None

                if rec is None:
                    break

                rid = rec.get(IDX_COL_NAME)
                if rid is None:
                    continue
                try:
                    ctx_id = int(rid)
                except Exception:
                    continue

                if ctx_id in done_ctx:
                    continue

                mc_record = _extract_mc_record(rec)
                if mc_record:
                    batch_items.append({**mc_record, "ctx_sample_id": ctx_id})

            if not batch_items:
                break

            conv_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
            producer_sem = asyncio.Semaphore(MAX_ASYNC_QGEN)

            async def producer(item: dict):
                async with producer_sem:
                    ctx_id = item["ctx_sample_id"]
                    question = item["question"]
                    choices = item["choices"]
                    answer_letter = item["answer"]
                    correct_text = _get_choice_text(choices, answer_letter)

                    style = sample_style()

                    user_msg_marked, target_letter = build_user_message(
                        style, question, choices,
                    )

                    persona_text = ""
                    if persona_provider is not None:
                        persona_data = persona_provider.get_persona()
                        persona_text = persona_data["persona"]
                        user_msg = await rewrite_with_persona(
                            model_id, user_msg_marked, persona_text,
                        )
                    else:
                        user_msg = _strip_protection_markers(user_msg_marked)

                    _, protected = _split_frame_and_protected(user_msg_marked)
                    if protected and protected not in user_msg:
                        user_msg = _strip_protection_markers(user_msg_marked)

                    choices_block = _format_choices(choices)
                    await conv_queue.put({
                        "ctx_id": ctx_id,
                        "style": style,
                        "user_msg": user_msg,
                        "answer_letter": answer_letter,
                        "correct_text": correct_text,
                        "target_letter": target_letter,
                        "level": item["level"],
                        "subject": item["subject"],
                        "persona_text": persona_text,
                        "original_question": question,
                        "original_choices_block": choices_block,
                    })

            producers = [asyncio.create_task(producer(item)) for item in batch_items]

            async def consumer():
                nonlocal seq_id
                while True:
                    work = await conv_queue.get()
                    if work is None:
                        conv_queue.task_done()
                        break

                    try:
                        async with write_lock:
                            if seq_id >= num_rows:
                                conv_queue.task_done()
                                return

                        cfg = STYLE_CONFIGS[work["style"]]
                        is_programmatic = cfg["programmatic_answer"]

                        best_text: Optional[str] = None
                        best_explicacao: str = ""
                        best_qualidade: str = QUALITY_ORDER[0]
                        best_rank: int = -1

                        for _ in range(MAX_RETRIES):
                            assistant_text = await generate_assistant_response(
                                model_id=model_id,
                                style=work["style"],
                                user_message=work["user_msg"],
                                answer_letter=work["answer_letter"],
                                correct_text=work["correct_text"],
                            )

                            if assistant_text is None:
                                continue

                            if is_programmatic:
                                best_text = assistant_text
                                best_qualidade = "excelente"
                                best_explicacao = "Resposta programática (letra correta)."
                                break

                            candidate = {
                                "conversation": [
                                    {"role": "user",      "content": work["user_msg"]},
                                    {"role": "assistant", "content": assistant_text},
                                ],
                                "response_style":         work["style"],
                                "gold_answer":            work["answer_letter"],
                                "target_letter":          work["target_letter"],
                                "level":                  work["level"],
                                "subject":                work["subject"],
                                "original_question":      work["original_question"],
                                "original_choices_block": work["original_choices_block"],
                            }
                            explicacao, qualidade = await evaluate_record(candidate, model_id)

                            try:
                                rank = QUALITY_ORDER.index(qualidade)
                            except ValueError:
                                rank = -1

                            if rank > best_rank:
                                best_rank = rank
                                best_text = assistant_text
                                best_qualidade = qualidade
                                best_explicacao = explicacao

                            if quality_passes(qualidade):
                                break

                        if best_text is None:
                            conv_queue.task_done()
                            continue

                        conversation = [
                            {"role": "user",      "content": work["user_msg"]},
                            {"role": "assistant", "content": best_text},
                        ]

                        async with write_lock:
                            if seq_id < num_rows:
                                record = {
                                    "id":                   seq_id,
                                    "seq_id":               seq_id,
                                    "conversation":         conversation,
                                    "response_style":       work["style"],
                                    "context_id":           work["ctx_id"],
                                    "gold_answer":          work["answer_letter"],
                                    "target_letter":        work["target_letter"],
                                    "level":                work["level"],
                                    "subject":              work["subject"],
                                    "persona":              work["persona_text"],
                                    "qualidade":            best_qualidade,
                                    "explicacao_qualidade": best_explicacao,
                                }
                                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                                fout.flush()
                                done_ctx.add(work["ctx_id"])
                                seq_id += 1
                    except Exception:
                        import traceback
                        traceback.print_exc()
                    finally:
                        conv_queue.task_done()

            consumers = [asyncio.create_task(consumer()) for _ in range(MAX_ASYNC_CHAT)]

            await asyncio.gather(*producers)
            for _ in range(MAX_ASYNC_CHAT):
                await conv_queue.put(None)

            await conv_queue.join()
            await asyncio.gather(*consumers, return_exceptions=True)

            pbar.update(1)

        pbar.close()
        print(f"Phase 1 done — {seq_id} conversations written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: MC conversation generation")
    parser.add_argument("--output", default=PHASE1_OUTPUT)
    parser.add_argument("--num-rows", type=int, default=NUM_ROWS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    asyncio.run(main(args.output, args.num_rows, args.batch_size))
