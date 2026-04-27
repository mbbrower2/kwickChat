import logging
import random
import sys
import os
from argparse import ArgumentParser
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from utils import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import download_pretrained_model

# [BNN] ── imports ─────────────────────────────────────────────────────────────
BNN_PATH = os.path.join(os.path.dirname(__file__), "..", "persona_bnn")
if BNN_PATH not in sys.path:
    sys.path.insert(0, BNN_PATH)
from data import load_convai2, synthetic_data
from learner import PersonaLearner
from bnn_persona_bridge import PersonaBridge
# ─────────────────────────────────────────────────────────────────────────────


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probabilities > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(personality, history, tokenizer, key_phrase, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, key_phrase, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        logits = model(input_ids, token_type_ids=token_type_ids)
        logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)
        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break
                prev = torch.multinomial(probs, num_samples=1)
        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())
    return current_output


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache')
    parser.add_argument("--model", type=str, default="openai-gpt", choices=['openai-gpt', 'gpt2'])
    parser.add_argument("--model_checkpoint", type=str,
                        default="openai-gpt/Aug08_11-34-29_b765d32f222d_openai-gpt")
    parser.add_argument("--max_history", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_sample", action='store_true')
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_suggestions", type=int, default=4)
    # [BNN] ── new args ────────────────────────────────────────────────────────
    parser.add_argument("--bnn_update_every", type=int, default=3,
                        help="Run BNN persona update every N turns (default 3)")
    parser.add_argument("--bnn_api_key", type=str, default="",
                        help="Anthropic API key for LLM tagger (or set ANTHROPIC_API_KEY env var)")
    # ─────────────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()

    if args.seed != 0:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (
        (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2'
        else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    )
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    # [BNN] ── train BNN on ConvAI2 ───────────────────────────────────────────
    print("\n[BNN] Training persona tagger on ConvAI2 …")
    api_key = args.bnn_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    bnn_learner = PersonaLearner(api_key=api_key)
    rows = load_convai2()
    random.shuffle(rows)
    bnn_learner.fit(rows[:int(len(rows) * 0.9)])
    print("[BNN] Persona tagger ready.\n")
    # ─────────────────────────────────────────────────────────────────────────

    history   = []
    # personality is now a plain list — the bridge mutates it in-place
    personality: list = []

    num_personas = input('Please enter the number of starting personas (0 to let BNN discover them): >>> ')
    for i in range(int(num_personas)):
        persona = input(f'Enter persona {i}: >>> ')
        personality.append(tokenizer.encode(persona))

    # [BNN] ── create bridge ───────────────────────────────────────────────────
    bridge = PersonaBridge(
        learner        = bnn_learner,
        tokenizer      = tokenizer,
        personality    = personality,   # shared reference
        min_update_turns = args.bnn_update_every,
    )
    # ─────────────────────────────────────────────────────────────────────────

    while True:
        raw_text     = input("Speaking Partner >>> ")
        raw_text_key = input("Your Key Words   >>> ")

        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Speaking Partner >>> ")
        while not raw_text_key:
            print('Prompt should not be empty!')
            raw_text_key = input("Your Key Words   >>> ")

        history.append(tokenizer.encode(raw_text))
        key_phrase = [tokenizer.encode(raw_text_key)]

        # [BNN] ── feed turn into bridge ──────────────────────────────────────
        # Both the partner's utterance and the user's keywords are informative
        bridge.update(raw_text + " " + raw_text_key)
        # ─────────────────────────────────────────────────────────────────────

        out_idx_list  = []
        out_text_list = []
        for seed in range(args.num_suggestions):
            random.seed(seed)
            torch.random.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            with torch.no_grad():
                # personality is passed here — the bridge keeps it updated
                out_ids = sample_sequence(
                    personality, history, tokenizer, key_phrase, model, args
                )

            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            print(f"Suggested Reply {seed} >>> {out_text}")
            out_idx_list.append(out_ids)
            out_text_list.append(out_text)

        select_idx = int(input("Your Selection >>> "))
        selected   = out_text_list[select_idx]
        print(f"Your Reply >>> {selected}")

        # [BNN] ── feed selected reply back into bridge ───────────────────────
        bridge.update(selected)
        # ─────────────────────────────────────────────────────────────────────

        history.append(out_idx_list[select_idx])
        history = history[-(2 * args.max_history + 1):]

        # [BNN] ── show current persona every 5 turns ─────────────────────────
        if len(history) % 5 == 0:
            bridge.status()
        # ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    run()