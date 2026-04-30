"""
evaluate_bnn.py — Evaluates the full patched_interact pipeline:
                  kwickChat (GPT) + persona_bnn (BNN persona tagger)

Metrics
-------
  BLEU    : n-gram overlap between generated completion and gold reply
  WER     : word error rate between generated completion and gold reply
  Cosine  : semantic similarity via sentence embedder
  Tags    : number of persona tags the BNN inferred per dialogue

Usage
-----
  cd kwickChat
  python evaluate_bnn.py \\
      --model_checkpoint runs/your_checkpoint \\
      --bnn_checkpoint ../models/bnn.pkl \\
      --dataset_path /cluster/tufts/hcilab_llm/personachat.json \\
      --output_path eval_results.json \\
      --num_dialogues 100
"""

import os
import sys
import json
import random
import pickle
import argparse
import warnings

import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BNN_DIR       = os.path.join(SCRIPT_DIR, "..", "persona_bnn")
KWICKCHAT_DIR = SCRIPT_DIR

sys.path.insert(0, BNN_DIR)
sys.path.insert(0, KWICKCHAT_DIR)

from utils import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import download_pretrained_model
from interact import sample_sequence
from learner import PersonaLearner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 1.0
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie   = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def compute_cosine(embedder, text_a: str, text_b: str) -> float:
    emb = embedder.encode([text_a, text_b], convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(
        emb[0].unsqueeze(0), emb[1].unsqueeze(0)
    ).item()


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_kwickchat(model_checkpoint: str):
    checkpoint = model_checkpoint or download_pretrained_model()
    tokenizer  = OpenAIGPTTokenizer.from_pretrained(checkpoint)
    model      = OpenAIGPTLMHeadModel.from_pretrained(checkpoint)
    model.to(DEVICE)
    add_special_tokens_(model, tokenizer)
    return model, tokenizer


def load_bnn(bnn_checkpoint: str) -> PersonaLearner:
    learner = PersonaLearner()
    if bnn_checkpoint and os.path.exists(bnn_checkpoint):
        with open(bnn_checkpoint, "rb") as f:
            data = pickle.load(f)
        learner.vocab   = data["vocab"]
        learner.tfidf   = data["tfidf"]
        learner.model   = data["model"]
        learner._replay = data["replay"]
        print(f"[eval] BNN loaded — {len(learner.vocab)} tags")
    else:
        print("[eval] No BNN checkpoint found — persona tags disabled")
    return learner


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading  (same format as kwickChat utils.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_dialogues(dataset_path: str, num_dialogues: int):
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"[eval] ERROR: dataset not found at {dataset_path}")
        print("[eval] Download with:")
        print("  wget https://s3.amazonaws.com/datasets.huggingface.co"
              "/personachat/personachat_self_original.json "
              "-O /cluster/tufts/hcilab_llm/personachat.json")
        sys.exit(1)

    if dataset_path.endswith(".json"):
        print(f"[eval] Loading PersonaChat JSON from {dataset_path} ...")
        with open(dataset_path, "r") as f:
            data = json.load(f)
        dialogues = data.get("valid", data.get("train", []))
    else:
        print("[eval] Loading dataset from torch cache ...")
        dataset   = torch.load(dataset_path, weights_only=False)
        dialogues = dataset.get("valid", dataset.get("train", []))

    print(f"[eval] {len(dialogues)} dialogues available — using {num_dialogues}")
    return dialogues[:num_dialogues]


# ══════════════════════════════════════════════════════════════════════════════
# Generation  (mirrors patched_interact.py logic)
# ══════════════════════════════════════════════════════════════════════════════

class EvalArgs:
    device      = DEVICE
    max_length  = 40
    min_length  = 3
    temperature = 0.85
    top_k       = 50
    top_p       = 0.9
    no_sample   = False
    max_history = 2


def tag_to_sentence(tag: str) -> str:
    templates = {
        "interest":    "i enjoy {v}.",
        "job":         "i work as a {v}.",
        "edu":         "i attended {v}.",
        "trait":       "i am {v}.",
        "life":        "i am {v}.",
        "pet":         "i have a {v}.",
        "value":       "i believe in {v}.",
        "fav_food":    "my favorite food is {v}.",
        "fav_color":   "my favorite color is {v}.",
        "skill":       "i can {v}.",
        "achievement": "i achieved {v}.",
    }
    if ":" not in tag:
        return f"i am interested in {tag.replace('_', ' ')}."
    prefix, value = tag.split(":", 1)
    value = value.replace("_", " ")
    return templates.get(prefix, "i am associated with {v}.").format(v=value)


def build_personality(bnn: PersonaLearner, tokenizer, text: str,
                      seen_tags: set) -> list:
    """
    Run BNN on text, convert new tags to encoded persona sentences.
    Returns list of newly encoded sentences (caller appends to persona_cache).
    """
    new_sentences = []
    if bnn.model is None or not text.strip():
        return new_sentences
    tags, uncertainty = bnn.infer(text)
    for tag in sorted(tags, key=lambda t: uncertainty.get(t, 1.0)):
        if tag not in seen_tags:
            sentence = tag_to_sentence(tag)
            encoded  = tokenizer.encode(sentence)
            new_sentences.append(encoded)
            seen_tags.add(tag)
    return new_sentences


def generate_completion(kwick_model, tokenizer, personality: list,
                        history: list, input_so_far: str,
                        args: EvalArgs) -> str:
    """
    Generate one completion of input_so_far conditioned on personality.
    Mirrors the autocomplete logic in bridge.py.
    """
    key_phrase  = [tokenizer.convert_tokens_to_ids(["<key>"])]
    partial_ids = tokenizer.encode(input_so_far)
    history_enc = [tokenizer.encode(h) if isinstance(h, str) else h
                   for h in history]

    with torch.no_grad():
        out_ids = sample_sequence(
            personality,
            history_enc + [partial_ids],
            tokenizer,
            key_phrase,
            kwick_model,
            args,
        )
    return tokenizer.decode(out_ids, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    print(f"[eval] Device: {DEVICE}")

    print("[eval] Loading kwickChat model ...")
    kwick_model, tokenizer = load_kwickchat(args.model_checkpoint)

    print("[eval] Loading BNN ...")
    bnn = load_bnn(args.bnn_checkpoint)

    print("[eval] Loading sentence embedder ...")
    embedder = SentenceTransformer("distilbert-base-nli-mean-tokens")

    dialogues = load_dialogues(args.dataset_path, args.num_dialogues)
    eval_args = EvalArgs()

    results = {
        "bleu":   [],
        "wer":    [],
        "cosine": [],
        "n_tags": [],   # how many persona tags the BNN inferred per dialogue
    }

    print(f"\n[eval] Evaluating on {len(dialogues)} dialogues ...\n")

    for d_idx, dialogue in enumerate(dialogues):

        # ── per-dialogue state (mirrors patched_interact.py) ──────────────────
        persona_cache: list = []
        seen_tags:     set  = set()
        history:       list = []

        # seed persona from the dialogue's own personality field if present
        # (PersonaChat JSON has pre-defined personality sentences)
        personality_sentences = dialogue.get("personality", [])
        for sentence in personality_sentences:
            if isinstance(sentence, str):
                persona_cache.append(tokenizer.encode(sentence))
            elif isinstance(sentence, list):
                persona_cache.append(sentence)  # already tokenized

        utterances = dialogue.get("utterances", [])

        for utt in utterances:
            candidates = utt.get("candidates", [])
            gold_reply = candidates[-1] if candidates else ""
            utt_history = utt.get("history", [])
            # the last history entry is what the speaking partner just said
            partner_text = utt_history[-1] if utt_history else ""

            if not gold_reply or not partner_text:
                continue

            # ── BNN: update persona from partner text ─────────────────────────
            new_sents = build_personality(
                bnn, tokenizer, partner_text, seen_tags
            )
            persona_cache.extend(new_sents)

            # ── generate completion using kwickChat + persona ─────────────────
            try:
                generated = generate_completion(
                    kwick_model, tokenizer,
                    persona_cache, history,
                    partner_text, eval_args,
                )
            except Exception as e:
                warnings.warn(f"Generation failed on dialogue {d_idx}: {e}")
                generated = ""

            if not generated:
                continue

            # ── compute metrics ───────────────────────────────────────────────
            results["bleu"].append(compute_bleu(gold_reply, generated))
            results["wer"].append(compute_wer(gold_reply, generated))
            results["cosine"].append(
                compute_cosine(embedder, gold_reply, generated)
            )

            # update history
            history.append(tokenizer.encode(partner_text))
            history = history[-(2 * eval_args.max_history + 1):]

        results["n_tags"].append(len(seen_tags))

        if (d_idx + 1) % 10 == 0:
            print(f"  Processed {d_idx + 1}/{len(dialogues)} dialogues ...")

    # ── aggregate ─────────────────────────────────────────────────────────────
    print("\n=== RESULTS ===\n")
    summary = {
        "bleu_mean":   float(np.mean(results["bleu"]))   if results["bleu"]   else 0.0,
        "wer_mean":    float(np.mean(results["wer"]))    if results["wer"]    else 0.0,
        "cosine_mean": float(np.mean(results["cosine"])) if results["cosine"] else 0.0,
        "tags_mean":   float(np.mean(results["n_tags"])) if results["n_tags"] else 0.0,
        "n_dialogues": len(dialogues),
        "n_utterances": len(results["bleu"]),
    }
    for k, v in summary.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")

    with open(args.output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] Results saved to {args.output_path}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate kwickChat + BNN persona pipeline"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="",
        help="Path to kwickChat model checkpoint. "
             "Leave empty to download the pretrained model.",
    )
    parser.add_argument(
        "--bnn_checkpoint",
        type=str,
        default="../models/bnn.pkl",
        help="Path to trained BNN .pkl file from train_bnn.py",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/cluster/tufts/hcilab_llm/personachat.json",
        help="Path to personachat_self_original.json or torch cache file",
    )
    parser.add_argument(
        "--num_dialogues",
        type=int,
        default=100,
        help="Number of dialogues to evaluate on",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eval_results.json",
        help="Where to save the JSON results summary",
    )
    args = parser.parse_args()
    evaluate(args)
