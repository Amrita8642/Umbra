"""
UMBRA Dataset Loader — fetches and caches all training datasets from HuggingFace Hub.

Datasets used:
  1. Anthropic/hh-rlhf         → GRPO preference training (chosen/rejected pairs)
  2. ai4privacy/pii-masking-300k → Sentrix PII validation test cases
  3. truthful_qa               → Calibration / honest-uncertainty reward signal
  4. lmsys/toxic-chat          → Adversarial & prompt-injection lines for Manipulator NPC
  5. daily_dialog              → Natural multi-turn dialogue for NPC script expansion

All datasets are downloaded once and cached under .cache/umbra_datasets/.
Set UMBRA_OFFLINE=1 env var to skip downloads and use only cached data.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/umbra_datasets")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OFFLINE = os.environ.get("UMBRA_OFFLINE", "0") == "1"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_hh_rlhf(split: str = "train", max_samples: int = 5000):
    """
    Anthropic/hh-rlhf — chosen / rejected conversation pairs.
    Used for: GRPO preference training prompt extraction.
    Schema: {"chosen": str, "rejected": str}
    """
    from datasets import load_dataset
    logger.info(f"[Dataset] Loading hh-rlhf split={split}, max={max_samples}")
    ds = load_dataset(
        "Anthropic/hh-rlhf",
        split=split,
        cache_dir=str(CACHE_DIR),
    )
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    logger.info(f"[Dataset] hh-rlhf loaded: {len(ds)} rows")
    return ds


def load_pii_masking(split: str = "train", max_samples: int = 2000):
    """
    ai4privacy/pii-masking-300k — text with Indian PII annotated.
    Used for: Sentrix regex validation and false-positive measurement.
    Schema: {"source_text": str, "target_text": str, "privacy_mask": list}
    """
    from datasets import load_dataset
    logger.info(f"[Dataset] Loading pii-masking-300k split={split}, max={max_samples}")
    ds = load_dataset(
        "ai4privacy/pii-masking-300k",
        split=split,
        cache_dir=str(CACHE_DIR),
    )
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    logger.info(f"[Dataset] pii-masking-300k loaded: {len(ds)} rows")
    return ds


def load_truthful_qa(split: str = "validation"):
    """
    truthful_qa — factual questions with best/correct answers.
    Used for: calibration reward (rewarding 'I don't know' on hard questions).
    Schema: {"question": str, "best_answer": str, "correct_answers": list, "incorrect_answers": list}
    """
    from datasets import load_dataset
    logger.info(f"[Dataset] Loading truthful_qa split={split}")
    ds = load_dataset(
        "truthful_qa",
        "generation",
        split=split,
        cache_dir=str(CACHE_DIR),
    )
    logger.info(f"[Dataset] truthful_qa loaded: {len(ds)} rows")
    return ds


def load_toxic_chat(split: str = "train", max_samples: int = 1000):
    """
    lmsys/toxic-chat — adversarial user inputs including prompt injections.
    Used for: expanding Manipulator NPC script pool with real injection attempts.
    Schema: {"user_input": str, "model_output": str, "toxicity_annotation": int, "jailbreaking": bool}
    """
    from datasets import load_dataset
    logger.info(f"[Dataset] Loading toxic-chat split={split}, max={max_samples}")
    ds = load_dataset(
        "lmsys/toxic-chat",
        "toxicchat0124",
        split=split,
        cache_dir=str(CACHE_DIR),
    )
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    logger.info(f"[Dataset] toxic-chat loaded: {len(ds)} rows")
    return ds


def load_daily_dialog(split: str = "train", max_samples: int = 2000):
    """
    daily_dialog — multi-turn natural conversations with dialogue act labels.
    Used for: expanding Agreeable, Liar, and Emotional NPC script pools.
    Act labels: 1=inform, 2=question, 3=directive, 4=commissive
    Schema: {"dialog": list[str], "act": list[int], "emotion": list[int]}

    NOTE: The original daily_dialog dataset uses a loading script which is
    blocked in datasets>=2.20. Falls back to silicone/dyda_da (parquet),
    then to a minimal synthetic dataset so training always proceeds.
    """
    from datasets import load_dataset, Dataset
    from collections import defaultdict

    # ── Attempt 1: silicone/dyda_da (parquet mirror of daily_dialog acts) ──
    try:
        logger.info(f"[Dataset] Loading silicone/dyda_da (daily_dialog replacement) split={split}")
        da_ds = load_dataset(
            "silicone", "dyda_da",
            split=split,
            cache_dir=str(CACHE_DIR),
        )
        # Group flat utterance rows back into dialog format by Dialogue_ID
        dialogs: dict = defaultdict(list)
        acts_map: dict = defaultdict(list)
        for row in da_ds:
            did = row["Dialogue_ID"]
            dialogs[did].append(row["Utterance"])
            # dyda_da act labels: 0=__dummy__, 1=inform, 2=question, 3=directive, 4=commissive
            acts_map[did].append(int(row["Dialogue_Act"]))

        rows = []
        for did in list(dialogs.keys())[:max_samples]:
            utt = dialogs[did]
            rows.append({
                "dialog":  utt,
                "act":     acts_map[did],
                "emotion": [0] * len(utt),   # dyda_da has no emotion field
            })
        ds = Dataset.from_list(rows)
        logger.info(f"[Dataset] silicone/dyda_da loaded: {len(ds)} dialogs")
        return ds

    except Exception as e1:
        logger.warning(f"[Dataset] silicone/dyda_da unavailable ({e1}), using synthetic fallback")

    # ── Fallback: minimal hardcoded dialogues (NPC static scripts still work) ──
    SYNTHETIC = [
        {"dialog": ["Sure, I'll help you with that.", "Of course, anything for you!", "Happy to assist anytime."],
         "act": [4, 4, 4], "emotion": [0, 0, 0]},
        {"dialog": ["You really should do this now.", "I'm telling you, it's urgent!", "Don't wait any longer."],
         "act": [3, 3, 3], "emotion": [1, 2, 1]},
        {"dialog": ["I didn't say that at all.", "Actually, that's not what I meant.", "You're misremembering."],
         "act": [1, 1, 1], "emotion": [0, 0, 0]},
        {"dialog": ["What do you think about it?", "Can you explain that further?", "Are you sure?"],
         "act": [2, 2, 2], "emotion": [0, 0, 0]},
        {"dialog": ["I promise I'll do better.", "You can count on me this time.", "I won't let you down."],
         "act": [4, 4, 4], "emotion": [0, 0, 0]},
        {"dialog": ["Listen to me carefully.", "You need to act on this immediately.", "There's no time to hesitate."],
         "act": [3, 3, 3], "emotion": [2, 3, 2]},
    ]
    # Repeat to fill max_samples
    repeated = (SYNTHETIC * (max_samples // len(SYNTHETIC) + 1))[:max_samples]
    ds = Dataset.from_list(repeated)
    logger.info(f"[Dataset] Using {len(ds)} synthetic fallback dialogues for NPC expansion")
    return ds


# ── Builders ──────────────────────────────────────────────────────────────────

def build_grpo_prompts(hh_ds, max_prompts: int = 2000) -> list[str]:
    """
    Extract prompt strings from hh-rlhf chosen/rejected pairs for GRPO training.
    The format is: "\\n\\nHuman: <prompt>\\n\\nAssistant: <response>"
    We extract only the Human turn as the prompt.
    Returns a list of prompt strings (deduplicated).
    """
    prompts = []
    seen = set()
    for row in hh_ds:
        text = row.get("chosen", "")
        # Split on first Assistant turn to isolate the human prompt
        parts = text.split("\n\nAssistant:")
        if not parts:
            continue
        raw_prompt = parts[0].replace("\n\nHuman:", "").strip()
        # Filter too-short or duplicate prompts
        if len(raw_prompt) < 20 or raw_prompt in seen:
            continue
        seen.add(raw_prompt)
        prompts.append(raw_prompt)
        if len(prompts) >= max_prompts:
            break
    logger.info(f"[Dataset] Built {len(prompts)} GRPO prompts from hh-rlhf")
    return prompts


def build_npc_scripts_from_datasets(toxic_ds, dialog_ds) -> dict[str, list[str]]:
    """
    Build expanded NPC script pools from real dataset samples.

    Manipulator  ← toxic-chat rows with toxicity_annotation==1 or jailbreaking==True
    Agreeable    ← daily_dialog commissive acts (act==4)
    Emotional    ← daily_dialog directive acts (act==3) with high emotion score
    Liar         ← daily_dialog inform acts rephrased with contradiction prefix

    Returns dict: {npc_type: [list_of_script_lines]}
    Static SCRIPTS in npc_agents.py are always kept; these are appended.
    """
    manipulator_lines: list[str] = []
    agreeable_lines: list[str] = []
    emotional_lines: list[str] = []
    liar_lines: list[str] = []

    # ── Manipulator from toxic-chat ──────────────────────
    for row in toxic_ds:
        user_input = row.get("user_input", "") or ""
        is_toxic = bool(row.get("toxicity_annotation", 0)) or bool(row.get("jailbreaking", False))
        if is_toxic and len(user_input) > 10:
            manipulator_lines.append(user_input[:200].strip())
        if len(manipulator_lines) >= 30:
            break

    # ── Agreeable / Emotional / Liar from daily_dialog ──
    for row in dialog_ds:
        dialog: list[str] = row.get("dialog", [])
        acts: list[int] = row.get("act", [])
        emotions: list[int] = row.get("emotion", [])

        for i, utterance in enumerate(dialog):
            utterance = utterance.strip()
            if not utterance or len(utterance) < 10:
                continue
            act = acts[i] if i < len(acts) else 0
            emotion = emotions[i] if i < len(emotions) else 0

            if act == 4 and len(agreeable_lines) < 25:          # commissive → agreeable
                agreeable_lines.append(utterance[:200])
            elif act == 3 and emotion > 0 and len(emotional_lines) < 25:  # directive + emotion
                emotional_lines.append(utterance[:200])
            elif act == 1 and i > 0 and len(liar_lines) < 25:  # inform → twist into contradiction
                prev = dialog[i - 1].strip()[:60]
                liar_lines.append(f"Actually, that's not what I said — {utterance[:150]}")

        if (len(agreeable_lines) >= 25 and len(emotional_lines) >= 25
                and len(liar_lines) >= 25):
            break

    result = {
        "Manipulator": manipulator_lines,
        "Agreeable":   agreeable_lines,
        "Emotional":   emotional_lines,
        "Liar":        liar_lines,
    }
    for npc_type, lines in result.items():
        logger.info(f"[Dataset] NPC '{npc_type}': {len(lines)} dataset script lines")
    return result


def build_pii_test_cases(pii_ds, max_cases: int = 500) -> list[dict]:
    """
    Build PII test cases from ai4privacy dataset.
    Each case: {"text": original_text, "redacted": masked_text, "pii_labels": list}
    Used to benchmark Sentrix false-positive and false-negative rates.
    """
    cases = []
    for row in pii_ds:
        source = row.get("source_text", "") or ""
        target = row.get("target_text", "") or ""
        masks = row.get("privacy_mask", []) or []
        if source and target:
            cases.append({
                "text": source,
                "redacted": target,
                "pii_labels": [m.get("label", "") for m in masks if isinstance(m, dict)],
            })
        if len(cases) >= max_cases:
            break
    logger.info(f"[Dataset] Built {len(cases)} PII test cases from pii-masking-300k")
    return cases


def build_calibration_qa(tqa_ds, max_cases: int = 200) -> list[dict]:
    """
    Build (question, best_answer, correct_answers) tuples from truthful_qa.
    Used to reward the model for expressing appropriate uncertainty.
    """
    qa_pairs = []
    for row in tqa_ds:
        question = row.get("question", "") or ""
        best_answer = row.get("best_answer", "") or ""
        correct = row.get("correct_answers", []) or []
        if question and best_answer:
            qa_pairs.append({
                "question": question,
                "answer": best_answer,
                "correct_answers": correct,
            })
        if len(qa_pairs) >= max_cases:
            break
    logger.info(f"[Dataset] Built {len(qa_pairs)} calibration QA pairs from truthful_qa")
    return qa_pairs


# ── Sentrix Benchmarking ───────────────────────────────────────────────────────

def benchmark_sentrix(pii_test_cases: list[dict]) -> dict:
    """
    Run Sentrix PII guard against the ai4privacy test cases.
    Returns: {"true_positives": int, "false_positives": int, "false_negatives": int,
              "precision": float, "recall": float}
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent.parent))
    from sentrix.pii_guard import run as sentrix_run, SentrixBlockException

    tp = fp = fn = 0
    for case in pii_test_cases:
        has_pii = bool(case.get("pii_labels"))
        try:
            result = sentrix_run(case["text"])
            detected = result.get("pii_found", False) or result.get("severity") in ("block", "warn")
        except SentrixBlockException:
            detected = True

        if has_pii and detected:
            tp += 1
        elif not has_pii and detected:
            fp += 1
        elif has_pii and not detected:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    logger.info(f"[Sentrix Benchmark] {metrics}")
    return metrics


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Loading all UMBRA datasets (this downloads ~2-3 GB on first run)...\n")

    hh_ds      = load_hh_rlhf(max_samples=100)
    pii_ds     = load_pii_masking(max_samples=100)
    tqa_ds     = load_truthful_qa()
    toxic_ds   = load_toxic_chat(max_samples=100)
    dialog_ds  = load_daily_dialog(max_samples=100)

    prompts    = build_grpo_prompts(hh_ds, max_prompts=10)
    npc_pool   = build_npc_scripts_from_datasets(toxic_ds, dialog_ds)
    pii_cases  = build_pii_test_cases(pii_ds, max_cases=50)
    qa_pairs   = build_calibration_qa(tqa_ds, max_cases=20)

    print(f"\n=== UMBRA Dataset Summary ===")
    print(f"  GRPO prompts      : {len(prompts)}")
    print(f"  NPC script lines  : { {k: len(v) for k, v in npc_pool.items()} }")
    print(f"  PII test cases    : {len(pii_cases)}")
    print(f"  Calibration QA    : {len(qa_pairs)}")
    print(f"\n  Sample GRPO prompt  : {prompts[0][:100]}...")
    print(f"\nRunning Sentrix benchmark on {len(pii_cases)} PII cases...")
    metrics = benchmark_sentrix(pii_cases)
    print(f"  Sentrix F1: {metrics['f1']}  Precision: {metrics['precision']}  Recall: {metrics['recall']}")
    print("\nAll datasets OK.")
