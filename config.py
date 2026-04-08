import os
from pathlib import Path

from dotenv import load_dotenv

_cfg_dir = Path(__file__).resolve().parent
_repo_root = _cfg_dir.parent
load_dotenv(_repo_root / ".env")
load_dotenv(_cfg_dir / ".env", override=True)

# vLLM SERVER
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8020/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "no-key-needed")

# SOURCE DATASET
MC_DATASET_ID = os.environ.get("MC_DATASET_ID", "TokenLab/multiple_choice_v0")
MC_DATASET_SUBSET = os.environ.get("MC_DATASET_SUBSET", "default")
IDX_COL_NAME = os.environ.get("IDX_COL_NAME", "idx")

# PERSONA DATASET
USE_PERSONA = os.environ.get("USE_PERSONA", "1") not in ("0", "false", "False", "no")

# GENERATION PARAMETERS
# Phase 1: response generation
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.7"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "1.0"))

RESP_TEMPERATURE = float(os.environ.get("RESP_TEMPERATURE", str(GEN_TEMPERATURE or 0.3)))
RESP_TOP_P = float(os.environ.get("RESP_TOP_P", str(min(GEN_TOP_P, 0.9))))
RESP_MAX_TOKENS = int(os.environ.get("RESP_MAX_TOKENS", "4096"))

# Phase 1: persona rewrite
REWRITE_TEMPERATURE = float(os.environ.get("REWRITE_TEMPERATURE", "0.7"))
REWRITE_TOP_P = float(os.environ.get("REWRITE_TOP_P", "0.95"))
REWRITE_MAX_TOKENS = int(os.environ.get("REWRITE_MAX_TOKENS", "2048"))
MAX_PERSONA_REWRITE_ATTEMPTS = int(os.environ.get("MAX_PERSONA_REWRITE_ATTEMPTS", "3"))

# Phase 2: evaluation judge
EVAL_TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0.0"))
EVAL_TOP_P = float(os.environ.get("EVAL_TOP_P", "0.9"))
EVAL_MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "1024"))

# QUALITY CONTROL
PASSING_QUALITY = os.environ.get("PASSING_QUALITY", "excelente")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# EXECUTION CONTROL
NUM_ROWS = int(os.environ.get("NUM_ROWS", "600000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))

MAX_ASYNC_TOTAL = int(os.environ.get("MAX_ASYNC", "64"))
CHAT_SHARE = float(os.environ.get("CHAT_SHARE", "0.8"))
MAX_ASYNC_CHAT = max(1, int(MAX_ASYNC_TOTAL * CHAT_SHARE))
MAX_ASYNC_QGEN = max(1, MAX_ASYNC_TOTAL - MAX_ASYNC_CHAT)
QUEUE_MAXSIZE = int(os.environ.get("QUEUE_MAXSIZE", str(MAX_ASYNC_CHAT * 4)))

# OUTPUT FILES
PHASE1_OUTPUT = os.environ.get("PHASE1_OUTPUT", "mc_phase1_conversations.jsonl")
PHASE2_OUTPUT = os.environ.get("PHASE2_OUTPUT", "mc_phase2_evaluated.jsonl")

# vLLM EXTRAS
STOP_STRINGS = ["<|im_end|>", "<|end_of_text|>"]
STOP_TOKEN_IDS = None
LOGITS_PROCESSORS: list = []
