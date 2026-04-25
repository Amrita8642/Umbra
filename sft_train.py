"""
UMBRA SFT (Supervised Fine-Tuning) script — Colab T4 ready.
Converts best RL episodes into prompt-completion pairs and fine-tunes
HuggingFaceTB/SmolLM-135M with SFTTrainer + LoRA (4-bit QLoRA).

Action space (8 actions):
  0=observe  1=challenge_mild  2=challenge_strong  3=request_evidence
  4=disengage  5=flag_coalition  6=flag_deception  7=escalate

Run in Colab:
  !cd /content/Umbra/Umbra && python sft_train.py
"""

import sys, json, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("UMBRA-SFT")

# ── Action descriptions ───────────────────────────────────────────────────────
ACTION_NAMES = {
    0: "observe",
    1: "challenge_mild",
    2: "challenge_strong",
    3: "request_evidence",
    4: "disengage",
    5: "flag_coalition",
    6: "flag_deception",
    7: "escalate",
}

# ── 1. Load and filter high-reward episodes ───────────────────────────────────
def load_best_episodes(path: str = "logs/episodes.jsonl", percentile: float = 0.5):
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    rewards = [e["total_reward"] for e in episodes]
    threshold = sorted(rewards)[int(len(rewards) * percentile)]
    best = [e for e in episodes if e["total_reward"] >= threshold and e["actions_taken"]]
    logger.info(f"Loaded {len(episodes)} episodes, selected {len(best)} above threshold {threshold:.3f}")
    return best


# ── 2. Convert episodes → prompt-completion pairs ────────────────────────────
def episode_to_text(ep: dict) -> str:
    npcs = ", ".join(ep.get("active_npcs", ["Unknown"]))
    stage = ep.get("stage", 1)
    actions = ep.get("actions_taken", [])
    reward = ep.get("total_reward", 0.0)
    flags = ep.get("verifier_flags", [])
    signals = ep.get("signals_fired", [])

    action_str = " → ".join(ACTION_NAMES.get(a, str(a)) for a in actions)

    prompt = (
        f"### UMBRA Decision Task\n"
        f"Stage: {stage}\n"
        f"Active NPCs: {npcs}\n"
        f"Signals detected: {', '.join(signals) if signals else 'none'}\n"
        f"Verifier flags: {', '.join(flags) if flags else 'none'}\n"
        f"\n### Optimal Action Sequence"
    )
    completion = f"\n{action_str}\n### Reward: {reward:.3f}"

    return prompt + completion


def build_sft_dataset(episodes: list) -> list:
    return [{"text": episode_to_text(ep)} for ep in episodes]


# ── 3. SFT Training ───────────────────────────────────────────────────────────
def run_sft():
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    MODEL_ID = "HuggingFaceTB/SmolLM-135M"
    OUTPUT_DIR = "umbra_sft_ckpt"

    # Load episodes
    episodes = load_best_episodes("logs/episodes.jsonl", percentile=0.5)
    records = build_sft_dataset(episodes)
    dataset = Dataset.from_list(records)
    logger.info(f"SFT dataset: {len(dataset)} examples")

    # 4-bit QLoRA config (T4 compatible — FP16 only)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFT training args (T4 safe — fp16=True, bf16=False)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=10,
        save_steps=100,
        max_seq_length=256,
        dataset_text_field="text",
        optim="adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting SFT training…")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"SFT complete — model saved to {OUTPUT_DIR}/")

    return OUTPUT_DIR


# ── 4. Optional: push to HuggingFace Hub ─────────────────────────────────────
def push_to_hub(output_dir: str, repo_id: str, hf_token: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Pushing merged model to Hub: {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    tokenizer.push_to_hub(repo_id, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M", device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, output_dir)
    merged = model.merge_and_unload()
    merged.push_to_hub(repo_id, token=hf_token)
    logger.info(f"Model pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    output_dir = run_sft()

    # Uncomment and fill in to push to HuggingFace Hub:
    # HF_TOKEN = "hf_YOUR_TOKEN_HERE"
    # REPO_ID  = "your-username/umbra-sft"
    # push_to_hub(output_dir, REPO_ID, HF_TOKEN)
