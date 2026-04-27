"""
bnn_persona_bridge.py — connects the BNN persona tagger to kwickChat.

Drop this file into the kwickChat directory alongside interact.py.

How it works
------------
1. At startup, PersonaBridge wraps a trained PersonaLearner.
2. After every conversation turn, update(utterance) is called.
3. The BNN infers new persona tags from the accumulated conversation.
4. Any new tags are converted to natural-language persona sentences and
   injected into kwickChat's `personality` list (the same list that gets
   passed to sample_sequence as encoded persona tokens).
5. If truly new tags are discovered (not seen during training), the BNN
   output layer grows and a continual learning step runs automatically.

Usage in interact.py — see patched_interact.py below.
"""

import sys
import os
from typing import List, Optional

# ── locate the BNN package ────────────────────────────────────────────────────
# Assumes persona_bnn/ sits one directory up from kwickChat/, e.g.:
#   project/
#     persona_bnn/
#     kwickChat/
#       bnn_persona_bridge.py   ← this file
# Adjust BNN_PATH if your layout differs.
BNN_PATH = os.path.join(os.path.dirname(__file__), "..", "persona_bnn")
if BNN_PATH not in sys.path:
    sys.path.insert(0, BNN_PATH)

from learner import PersonaLearner  # noqa: E402  (path added above)


# ──────────────────────────────────────────────────────────────────────────────
# Tag → natural language sentence
# ──────────────────────────────────────────────────────────────────────────────

# Maps canonical BNN tag prefixes to a sentence template.
# The part after the colon is inserted into the template.
_TEMPLATES = {
    "interest":    "i enjoy {value}.",
    "job":         "i work as a {value}.",
    "edu":         "i attended {value}.",
    "trait":       "i am {value}.",
    "life":        "i am {value}.",
    "pet":         "i have a {value}.",
    "value":       "i believe in {value}.",
    "fav_food":    "my favorite food is {value}.",
    "fav_color":   "my favorite color is {value}.",
    "skill":       "i have the skill: {value}.",
    "achievement": "i achieved {value}.",
}

def tag_to_sentence(tag: str) -> str:
    """
    Convert a BNN tag like "interest:beekeeping" into a natural-language
    persona sentence like "i enjoy beekeeping." that kwickChat can consume.
    """
    if ":" not in tag:
        return f"i am interested in {tag.replace('_', ' ')}."
    prefix, value = tag.split(":", 1)
    value = value.replace("_", " ")
    template = _TEMPLATES.get(prefix, "i am associated with {value}.")
    return template.format(value=value)


# ──────────────────────────────────────────────────────────────────────────────
# PersonaBridge
# ──────────────────────────────────────────────────────────────────────────────

class PersonaBridge:
    """
    Wraps PersonaLearner and keeps kwickChat's personality list in sync.

    Parameters
    ----------
    learner      : a trained PersonaLearner instance
    tokenizer    : the kwickChat tokenizer (OpenAIGPT or GPT2)
    personality  : the mutable list kwickChat passes to sample_sequence.
                   This list is updated in-place so interact.py sees changes
                   without any extra wiring.
    min_update_turns : only run a continual-learning update every N turns
                       (avoids an API call on every single utterance).
    """

    def __init__(self,
                 learner: PersonaLearner,
                 tokenizer,
                 personality: List[List[int]],
                 min_update_turns: int = 3):
        self.learner          = learner
        self.tokenizer        = tokenizer
        self.personality      = personality   # shared reference — mutated in-place
        self.min_update_turns = min_update_turns

        self._conversation_buffer: List[str] = []  # raw text turns accumulated
        self._turn_count: int                = 0
        self._known_tags: set                = set(learner.vocab.id2tag)

        # Encode any tags already in the vocab as initial persona sentences
        self._sync_personality()

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, utterance: str):
        """
        Call after every conversation turn with the raw utterance text.
        Accumulates turns and, every `min_update_turns`, runs the BNN to
        infer new tags and optionally triggers continual learning.
        """
        self._conversation_buffer.append(utterance)
        self._turn_count += 1

        if self._turn_count % self.min_update_turns != 0:
            return

        # Build a single text from the buffered turns
        combined_text = " ".join(self._conversation_buffer)

        # ── infer tags from conversation so far ───────────────────────────────
        tags, uncertainty = self.learner.infer(combined_text)

        # ── check for truly new tags (triggers continual learning) ───────────
        new_tags = [t for t in tags if t not in self._known_tags]
        if new_tags:
            print(f"\n[BNN] {len(new_tags)} new persona tag(s) discovered: {new_tags}")
            row = _turns_to_row(self._conversation_buffer)
            self.learner.update([row])
            self._known_tags.update(self.learner.vocab.id2tag)

            # Re-infer with the updated model
            tags, uncertainty = self.learner.infer(combined_text)

        # ── sync personality list with inferred tags ──────────────────────────
        if tags:
            self._inject_tags(tags, uncertainty)

    def status(self):
        """Print the current BNN-derived persona to stdout."""
        print("\n[BNN] Current persona tags:")
        decoded = [
            self.tokenizer.decode(p, skip_special_tokens=True)
            for p in self.personality
        ]
        for sentence in decoded:
            print(f"  • {sentence}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _tag_to_encoded(self, tag: str) -> Optional[List[int]]:
        sentence = tag_to_sentence(tag)
        try:
            return self.tokenizer.encode(sentence)
        except Exception:
            return None

    def _inject_tags(self, tags: List[str], uncertainty: dict):
        """
        Add encoded persona sentences for newly inferred tags.
        Only adds a sentence if it isn't already present (avoids duplicates).
        Sorts by confidence (lowest uncertainty first) so the most certain
        tags appear earliest in the persona, matching kwickChat's convention.
        """
        existing_sentences = {
            self.tokenizer.decode(p, skip_special_tokens=True).strip().lower()
            for p in self.personality
        }

        added = []
        for tag in sorted(tags, key=lambda t: uncertainty.get(t, 1.0)):
            sentence = tag_to_sentence(tag)
            if sentence.strip().lower() not in existing_sentences:
                encoded = self._tag_to_encoded(tag)
                if encoded:
                    self.personality.append(encoded)
                    existing_sentences.add(sentence.strip().lower())
                    added.append(tag)

        if added:
            print(f"\n[BNN] Injected {len(added)} persona sentence(s): {added}")

    def _sync_personality(self):
        """
        On startup, run a quick inference over any text already in the
        conversation buffer (empty at init, so this is a no-op by default).
        Subclasses can override to pre-populate from a saved state.
        """
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Helper: wrap raw turns into a ConvAI2-shaped row for learner.update()
# ──────────────────────────────────────────────────────────────────────────────

def _turns_to_row(turns: List[str]) -> dict:
    return {
        "dialog_id":    "live_conversation",
        "bot_profile":  [],
        "user_profile": [],
        "dialog":       [{"text": t} for t in turns],
    }