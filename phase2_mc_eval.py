"""
Phase 2: MC Quality Evaluation (LLM Judge)

Used as a library by phase1_mc_gen.py: each generated assistant turn is evaluated
immediately, and the generator retries until the response reaches PASSING_QUALITY
(default: "excelente") or MAX_RETRIES is exhausted.

NOTE: "apenas_letra_*" styles use a hardcoded programmatic answer (the gold
letter) and are NEVER sent here — phase1_mc_gen.py short-circuits them before
calling evaluate_record(). Only "letra_raciocinio" and "is_correct" are judged.

Evaluation is split into separate focused judgments (one LLM call each):
  1. CORREÇÃO        — is the gold answer correctly identified?
  2. ADERÊNCIA       — does the response match the expected style?
  3. QUALIDADE GERAL — language, no leaked hints, no hallucinated choices, specificity

The final quality is the MINIMUM across all three categories.

Public API
----------
  PASSING_QUALITY   : str   — minimum quality label to accept ("excelente")
  QUALITY_ORDER     : list  — ordered scale from worst to best
  evaluate_record() : async — judge one record, return (explicacao, qualidade)
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, BadRequestError

from MCDataGen.config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    EVAL_TEMPERATURE,
    EVAL_TOP_P,
    EVAL_MAX_TOKENS,
    PASSING_QUALITY,
    STOP_STRINGS,
    STOP_TOKEN_IDS,
    LOGITS_PROCESSORS,
)

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

QUALITY_ORDER = ["muito ruim", "ruim", "média", "boa", "excelente"]

# ---------------------------------------------------------------------
# Style descriptions
# ---------------------------------------------------------------------
STYLE_DESCRIPTIONS = {
    "letra_raciocinio": "letra correta + raciocínio explicativo detalhado",
    "is_correct": "confirmação/negação da alternativa indicada + justificativa",
}

# ---------------------------------------------------------------------
# Per-category prompt templates
# Each returns a focused judgment on ONE aspect only.
# All expect JSON output: {"explicacao": "...", "qualidade": "..."}
# ---------------------------------------------------------------------

_CORRECAO_PROMPT = """\
Você é um avaliador rigoroso de respostas de múltipla escolha em pt-BR.
Avalie SOMENTE se a resposta do assistente identifica a alternativa correta.

=== QUESTÃO ORIGINAL ===
{original_question}

{original_choices}

=== METADADOS ===
Resposta correta (gabarito): {gold_answer}
Alternativa alvo (perguntada ao modelo): {target_letter}
Estilo da resposta: {style_desc}

=== RESPOSTA DO ASSISTENTE ===
{assistant_response}

=== CRITÉRIO ===
Para estilo "letra_raciocinio": o assistente identifica a letra {gold_answer} como correta?
Para estilo "is_correct": a confirmação/negação do assistente sobre a alternativa {target_letter} está factualmente correta?
  (Se {target_letter} == {gold_answer}, o assistente deve confirmar que está correta.
   Se {target_letter} != {gold_answer}, o assistente deve negar que está correta.)

=== ESCALA ===
- muito ruim : Resposta errada — identifica alternativa incorreta ou confirma/nega erroneamente.
- ruim       : Resposta ambígua — não fica claro qual alternativa o assistente considera correta.
- média      : Resposta correta mas com ressalvas ou hesitações desnecessárias.
- boa        : Resposta correta e clara.
- excelente  : Resposta correta, clara e inequívoca.

Formato de saída OBRIGATÓRIO (JSON estrito, sem texto extra, sem crases):
{{
  "explicacao": "<string — descreva se a correção está ok ou qual erro encontrou>",
  "qualidade": "muito ruim" | "ruim" | "média" | "boa" | "excelente"
}}"""

_ADERENCIA_LETRA_RACIOCINIO = """\
Você é um avaliador rigoroso de respostas de múltipla escolha em pt-BR.
Avalie SOMENTE a aderência ao estilo "letra_raciocinio".

=== QUESTÃO ORIGINAL ===
{original_question}

{original_choices}

=== RESPOSTA DO ASSISTENTE ===
{assistant_response}

=== CRITÉRIOS DE ESTILO ===
1. A resposta começa identificando claramente a letra correta.
2. Após a letra, apresenta raciocínio técnico e específico ao conteúdo da questão:
   - Explica por que a alternativa correta está certa.
   - Explica por que as demais alternativas estão erradas (pode usar exemplos para
     ilustrar, desde que os exemplos não inventem alternativas inexistentes).
   - O raciocínio NÃO é uma mera reformulação do enunciado nem uma resposta genérica
     que serviria para qualquer questão do mesmo assunto.

=== ESCALA ===
- muito ruim : Não segue o estilo (ex.: não indica a letra, ou não dá raciocínio nenhum).
- ruim       : Indica a letra mas o raciocínio é genérico ou reformula o enunciado.
- média      : Estilo aproximado, mas raciocínio superficial ou incompleto.
- boa        : Estilo adequado, raciocínio presente e relevante.
- excelente  : Estilo perfeito — letra clara no início, raciocínio técnico e específico.

Formato de saída OBRIGATÓRIO (JSON estrito, sem texto extra, sem crases):
{{
  "explicacao": "<string — descreva se o estilo está ok ou qual falha encontrou>",
  "qualidade": "muito ruim" | "ruim" | "média" | "boa" | "excelente"
}}"""

_ADERENCIA_IS_CORRECT = """\
Você é um avaliador rigoroso de respostas de múltipla escolha em pt-BR.
Avalie SOMENTE a aderência ao estilo "is_correct".

=== QUESTÃO ORIGINAL ===
{original_question}

{original_choices}

=== METADADOS ===
Alternativa perguntada: {target_letter}

=== RESPOSTA DO ASSISTENTE ===
{assistant_response}

=== CRITÉRIOS DE ESTILO ===
1. A resposta confirma ou nega de forma direta e inequívoca se a alternativa
   {target_letter} está correta.
2. A justificativa é concreta e específica ao conteúdo da questão — não genérica.
   Ela deve explicar o motivo técnico/conceitual pelo qual a alternativa é correta
   ou incorreta, podendo usar exemplos ilustrativos sem inventar alternativas novas.

=== ESCALA ===
- muito ruim : Não confirma nem nega, ou a justificativa é inexistente.
- ruim       : Confirma/nega mas sem justificativa substancial ou de forma ambígua.
- média      : Confirma/nega com justificativa superficial.
- boa        : Confirma/nega claramente com justificativa relevante.
- excelente  : Confirma/nega de forma direta e inequívoca, justificativa técnica e específica.

Formato de saída OBRIGATÓRIO (JSON estrito, sem texto extra, sem crases):
{{
  "explicacao": "<string — descreva se o estilo está ok ou qual falha encontrou>",
  "qualidade": "muito ruim" | "ruim" | "média" | "boa" | "excelente"
}}"""

_QUALIDADE_GERAL_PROMPT = """\
Você é um avaliador rigoroso de respostas de múltipla escolha em pt-BR.
Avalie SOMENTE os aspectos de qualidade geral listados abaixo (NÃO avalie correção nem estilo).

=== QUESTÃO ORIGINAL ===
{original_question}

{original_choices}

=== RESPOSTA DO ASSISTENTE ===
{assistant_response}

=== CRITÉRIOS ===
1. A resposta está em pt-BR?
2. A resposta NÃO vaza a dica interna de gabarito — ou seja, NÃO contém frases como
   "a alternativa correta é X)" ou "de acordo com o gabarito"? O assistente deve
   apresentar o conhecimento como próprio, sem revelar que recebeu a resposta pronta.
3. A resposta NÃO afirma ter acesso a contexto externo além do que está na questão
   (ex.: NÃO diz "com base no texto fornecido", "de acordo com o contexto", "conforme
   o material apresentado" ou expressões similares)?
4. A resposta NÃO cita alternativas que não existem na questão (inventar letras E, F
   etc. é uma falha; usar exemplos próprios para ilustrar o raciocínio é permitido)?
5. A resposta NÃO é vaga ou genérica — ela é específica ao conteúdo desta questão?

=== ESCALA ===
- muito ruim : Idioma errado OU vaza gabarito OU afirma ter contexto externo.
- ruim       : Critério 4 ou 5 violado (inventa alternativas ou é genérica).
- média      : Todos os critérios quase satisfeitos, com falha menor.
- boa        : Todos os critérios satisfeitos com ressalva mínima.
- excelente  : Todos os critérios acima satisfeitos sem ressalvas.

Formato de saída OBRIGATÓRIO (JSON estrito, sem texto extra, sem crases):
{{
  "explicacao": "<string — cite qual critério falhou, se houver>",
  "qualidade": "muito ruim" | "ruim" | "média" | "boa" | "excelente"
}}"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def quality_passes(qualidade: str) -> bool:
    """Return True if qualidade >= PASSING_QUALITY on the QUALITY_ORDER scale."""
    try:
        return QUALITY_ORDER.index(qualidade) >= QUALITY_ORDER.index(PASSING_QUALITY)
    except ValueError:
        return False


def _quality_rank(qualidade: str) -> int:
    try:
        return QUALITY_ORDER.index(qualidade)
    except ValueError:
        return -1


def _min_quality(*qualities: str) -> str:
    """Return the lowest quality label from the given set."""
    ranks = [_quality_rank(q) for q in qualities]
    min_rank = min(ranks)
    if min_rank < 0:
        return "muito ruim"
    return QUALITY_ORDER[min_rank]


# ---------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------
async def _chat_call(
    messages: List[Dict[str, str]],
    model_id: str,
    *,
    enable_thinking: bool = False,
) -> str:
    extra_body: Dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "stop": STOP_STRINGS,
        "stop_token_ids": STOP_TOKEN_IDS,
        "logits_processors": LOGITS_PROCESSORS,
    }
    resp = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        max_tokens=EVAL_MAX_TOKENS,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content or ""


async def safe_chat_call(
    messages: List[Dict[str, str]],
    model_id: str,
) -> Optional[str]:
    """Swallows exceptions; retries with thinking enabled on BadRequestError."""
    try:
        return await _chat_call(messages, model_id)
    except BadRequestError:
        try:
            return await _chat_call(messages, model_id, enable_thinking=True)
        except Exception:
            return None
    except Exception:
        return None


def _parse_judge_response(raw: Optional[str]) -> Tuple[str, str]:
    """Parse a judge JSON response into (explicacao, qualidade)."""
    if not raw:
        return "Resposta vazia do modelo avaliador", "muito ruim"

    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("explicacao", ""), parsed.get("qualidade", "invalida")
    except json.JSONDecodeError:
        pass

    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            parsed = json.loads(raw[s: e + 1])
            return parsed.get("explicacao", ""), parsed.get("qualidade", "invalida")
        except json.JSONDecodeError:
            pass

    return raw, "invalida"


# ---------------------------------------------------------------------
# Individual category evaluators
# ---------------------------------------------------------------------
async def _judge_correcao(
    model_id: str,
    original_question: str,
    original_choices: str,
    gold_answer: str,
    target_letter: str,
    style_desc: str,
    assistant_response: str,
) -> Tuple[str, str]:
    prompt = _CORRECAO_PROMPT.format(
        original_question=original_question,
        original_choices=original_choices,
        gold_answer=gold_answer,
        target_letter=target_letter or "N/A",
        style_desc=style_desc,
        assistant_response=assistant_response,
    )
    raw = await safe_chat_call([{"role": "user", "content": prompt}], model_id)
    return _parse_judge_response(raw)


async def _judge_aderencia(
    model_id: str,
    style: str,
    original_question: str,
    original_choices: str,
    target_letter: str,
    assistant_response: str,
) -> Tuple[str, str]:
    if style == "is_correct":
        prompt = _ADERENCIA_IS_CORRECT.format(
            original_question=original_question,
            original_choices=original_choices,
            target_letter=target_letter or "N/A",
            assistant_response=assistant_response,
        )
    elif style == "letra_raciocinio":
        prompt = _ADERENCIA_LETRA_RACIOCINIO.format(
            original_question=original_question,
            original_choices=original_choices,
            assistant_response=assistant_response,
        )
    else:
        return "Estilo não avaliável", "excelente"

    raw = await safe_chat_call([{"role": "user", "content": prompt}], model_id)
    return _parse_judge_response(raw)


async def _judge_qualidade_geral(
    model_id: str,
    original_question: str,
    original_choices: str,
    assistant_response: str,
) -> Tuple[str, str]:
    prompt = _QUALIDADE_GERAL_PROMPT.format(
        original_question=original_question,
        original_choices=original_choices,
        assistant_response=assistant_response,
    )
    raw = await safe_chat_call([{"role": "user", "content": prompt}], model_id)
    return _parse_judge_response(raw)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
async def evaluate_record(
    record: Dict[str, Any],
    model_id: str,
) -> Tuple[str, str]:
    """
    Judge a single mc_gen record by evaluating three categories independently.

    Parameters
    ----------
    record : dict with keys: conversation, response_style, gold_answer,
             target_letter, level, subject, original_question,
             original_choices_block
    model_id : vLLM model identifier

    Returns
    -------
    (explicacao, qualidade)
    Falls back to ("Erro na avaliação", "muito ruim") on any failure.
    """
    style = record.get("response_style", "")
    style_desc = STYLE_DESCRIPTIONS.get(style, style)
    gold = record.get("gold_answer", "?")
    target = record.get("target_letter") or "N/A"

    original_question = record.get("original_question", "")
    original_choices_block = record.get("original_choices_block", "")

    conversation = record.get("conversation", [])
    assistant_response = ""
    for turn in conversation:
        if turn.get("role") == "assistant":
            assistant_response = turn.get("content", "")

    if not assistant_response:
        return "Resposta do assistente vazia", "muito ruim"

    correcao_task = _judge_correcao(
        model_id, original_question, original_choices_block,
        gold, target, style_desc, assistant_response,
    )
    aderencia_task = _judge_aderencia(
        model_id, style, original_question, original_choices_block,
        target, assistant_response,
    )
    qualidade_task = _judge_qualidade_geral(
        model_id, original_question, original_choices_block,
        assistant_response,
    )

    results = await asyncio.gather(
        correcao_task, aderencia_task, qualidade_task,
        return_exceptions=True,
    )

    explicacoes = []
    qualities = []

    labels = ["CORREÇÃO", "ADERÊNCIA", "QUALIDADE GERAL"]
    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            explicacoes.append(f"{label}: Erro na avaliação ({result})")
            qualities.append("muito ruim")
        else:
            exp, qual = result
            explicacoes.append(f"{label}: [{qual}] {exp}")
            qualities.append(qual)

    final_quality = _min_quality(*qualities)
    combined_explanation = " | ".join(explicacoes)

    return combined_explanation, final_quality
