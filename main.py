#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-lingual WSD benchmark generator — SAME JSON FIELD ORDER:
task_id, task_type, source_word, source_lang, target_lang,
context_sentence, question, choices, answer, meta

- Offline-friendly BabelNet Python API (local indices or RPC)
- Supports tiers via language_config.py or explicit --targets
- Words file accepts "word,POS" or "word<TAB>POS" (POS in {NOUN,VERB,ADJ/ADJECTIVE,ADV/ADVERB})
- Choice quality filters (script sanity, placeholder removal, dedup)
- Distractors from other senses of the same English lemma & POS
- Example-based questions (English context by default)
- STRICT: context sentence must contain the English source word
- Meta with synset IDs, pos, answer_idx, provenance
- Exhaustive mode (--exhaustive) to sweep ALL (word × language) pairs
- Per-language cap in exhaustive mode (--max-per-lang)
- Per-sense generation (multiple items per word×lang, one per sense with data)
- --max-senses-per-pair to limit items per (word×target language)
- Optional: --require-exact-choices (skip items that don't reach exactly N choices)
"""

import argparse
import json
import logging
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

# ------------------ BabelNet Python API (offline/RPC) ------------------
try:
    import babelnet as bn
    from babelnet.language import Language
    from babelnet.pos import POS
except Exception as e:
    raise RuntimeError(
        "Failed to import BabelNet Python API. Ensure it's installed and configured for offline or RPC use."
    ) from e

GENERATOR_VERSION = "clwsd-gen@v1.6"

# ------------------ POS + script helpers ------------------
POS_MAP = {
    "NOUN": POS.NOUN,
    "VERB": POS.VERB,
    "ADJ": POS.ADJ,          # Python API uses POS.ADJ / POS.ADV
    "ADJECTIVE": POS.ADJ,
    "ADV": POS.ADV,
    "ADVERB": POS.ADV,
}

NON_LATIN_LANGS = {"JA", "KO", "ZH", "AR", "HE", "RU", "UK", "BG", "EL"}

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def looks_placeholder(s: str) -> bool:
    # ALL-CAPS tokens or anything with digits (e.g., BANK1, ABC123)
    return bool(re.fullmatch(r"[A-Z0-9]+", s) or re.search(r"\d", s))

def contains_non_latin(s: str) -> bool:
    for ch in s:
        if ch.isalpha():
            if "LATIN" not in unicodedata.name(ch, ""):
                return True
    return False

def valid_for_lang(lang_upper: str, s: str) -> bool:
    if not s or not s.strip():
        return False
    if looks_placeholder(s):
        return False
    if lang_upper in NON_LATIN_LANGS:
        return contains_non_latin(s)
    return True

def dedup_keep_order(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        k = nfc(x).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

# ------------------ word list ------------------
@dataclass
class WordSpec:
    word: str
    pos: Optional[str]  # NOUN/VERB/ADJ/ADV or None

def parse_words_file(path: Path) -> List[WordSpec]:
    """
    Accepts lines like:
      bank,NOUN
      run,VERB
      star   <no POS>
    Also supports TAB as a separator.
    """
    out: List[WordSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t,]", line)
        if len(parts) == 1:
            w = parts[0].strip()
            pos = None
        else:
            w = parts[0].strip()
            pos = parts[1].strip().upper()
            if pos == "ADJECTIVE":
                pos = "ADJ"
            elif pos == "ADVERB":
                pos = "ADV"
        if not w:
            continue
        out.append(WordSpec(word=w, pos=pos))
    return out

# ------------------ language_config.py loader ------------------
def load_targets_from_config(cfg_path: Path, tier: str) -> List[str]:
    """
    tier ∈ {'high_resource','medium_resource','low_resource'}
    Returns UPPERCASE 2-letter codes (e.g., ['FR','DE',...]) taken from LANGUAGE_CONFIG.
    """
    code = cfg_path.read_text(encoding="utf-8")
    scope: Dict[str, dict] = {}
    exec(code, scope, scope)
    if "LANGUAGE_CONFIG" not in scope:
        raise RuntimeError("LANGUAGE_CONFIG not found in the provided language_config.py")
    cfg = scope["LANGUAGE_CONFIG"]
    if tier not in cfg:
        raise RuntimeError(f"Tier '{tier}' not found in LANGUAGE_CONFIG")
    tier_dict = cfg[tier]
    targets = []
    for k, v in tier_dict.items():
        c = (v.get("code") or k).strip()
        if c:
            targets.append(c.upper())
    return sorted(set(targets))

# ------------------ BabelNet helpers ------------------
def bn_pos_of(pos: Optional[str]) -> Optional[POS]:
    return POS_MAP.get(pos) if pos else None

def synset_id_str(syn) -> str:
    sid = getattr(syn, "id", None)
    if sid is None:
        sid = getattr(syn, "synset_id", None)
    return str(sid) if sid is not None else ""

def get_synsets_for_word(source_word: str, bn_pos: Optional[POS]):
    kwargs = {}
    if bn_pos:
        kwargs["poses"] = {bn_pos}
    return bn.get_synsets(
        source_word,
        from_langs={Language.EN},
        to_langs=None,
        **kwargs
    ) or []

def english_examples(synset) -> Tuple[List[str], str]:
    """
    Returns (examples, origin), where origin is 'examples' or 'glosses'
    """
    out = []
    try:
        for ex in synset.examples(language=Language.EN) or []:
            txt = getattr(ex, "example", None) or str(ex)
            if txt and len(txt.split()) >= 4:
                out.append(clean_space(nfc(txt)))
    except Exception:
        pass
    if out:
        return (dedup_keep_order(out), "examples")

    # Fallback: cut glosses into sentences
    try:
        for gl in synset.glosses(language=Language.EN) or []:
            txt = getattr(gl, "gloss", None) or str(gl)
            if not txt:
                continue
            for sent in re.split(r"(?<=[.!?])\s+", txt):
                if len(sent.split()) >= 4:
                    out.append(clean_space(nfc(sent)))
    except Exception:
        pass
    return (dedup_keep_order(out), "glosses")

def target_lemmas_from_synset(synset, target_lang_upper: str) -> List[str]:
    tgt_enum = getattr(Language, target_lang_upper)
    lemmas = []
    try:
        for bl in synset.lemmas(tgt_enum) or []:
            lemma = getattr(bl, "lemma", None) or str(bl)
            if lemma:
                lemmas.append(clean_space(nfc(lemma)))
    except Exception:
        pass
    return lemmas

# ------------------ per-sense item builder ------------------
def build_items_for_pair(
    task_counter_start: int,
    source_word: str,
    pos: Optional[str],
    target_lang_upper: str,
    max_choices: int,
    require_exact: bool,
    max_senses_per_pair: int,
    rng: random.Random,
) -> List[dict]:
    """
    Return 0..N items: one per *sense* (synset) that has enough data.
    Honor max_senses_per_pair > 0 by limiting items per (word×target_lang).
    Enforce: English context sentence MUST contain the English source word.
    """
    items: List[dict] = []
    bn_pos = bn_pos_of(pos)
    synsets = get_synsets_for_word(source_word, bn_pos)
    if len(synsets) < 2:
        return items  # need ≥2 senses for valid distractors

    rng.shuffle(synsets)
    added = 0

    for gold in synsets:
        # English examples/glosses as context
        examples, example_origin = english_examples(gold)
        if not examples:
            continue

        # STRICT: pick a sentence that actually contains the source word
        context_sentence = None
        for e in examples:
            if re.search(rf"\b{re.escape(source_word)}\b", e, flags=re.IGNORECASE):
                context_sentence = e
                break
        if context_sentence is None:
            continue  # skip if the source word doesn't appear in the sentence

        # target lemmas (answers/distractors)
        gold_lemmas = target_lemmas_from_synset(gold, target_lang_upper)
        gold_lemmas = [x for x in gold_lemmas if valid_for_lang(target_lang_upper, x)]
        gold_lemmas = dedup_keep_order(gold_lemmas)
        if not gold_lemmas:
            continue

        gold_ans = min(gold_lemmas, key=len)

        # distractors from other senses
        distractors_pairs: List[Tuple[str, str]] = []
        for other in synsets:
            if other is gold:
                continue
            other_lemmas = target_lemmas_from_synset(other, target_lang_upper)
            other_lemmas = [x for x in other_lemmas if valid_for_lang(target_lang_upper, x)]
            other_id = synset_id_str(other)
            for lem in other_lemmas:
                distractors_pairs.append((lem, other_id))

        # clean distractors
        def too_close(a: str, b: str) -> bool:
            aL, bL = a.lower(), b.lower()
            if aL == bL:
                return True
            if abs(len(aL) - len(bL)) <= 1 and (aL in bL or bL in aL):
                return True
            return False

        filtered: List[Tuple[str, str]] = []
        seen_norm: Set[str] = set()
        gold_norm = nfc(gold_ans).strip().lower()
        for lem, sid in distractors_pairs:
            if looks_placeholder(lem) or too_close(lem, gold_ans):
                continue
            norm = nfc(lem).strip().lower()
            if norm == gold_norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            filtered.append((lem, sid))

        rng.shuffle(filtered)
        filtered = filtered[: max(0, max_choices - 1)]
        distractor_lemmas = [lem for (lem, _sid) in filtered]
        contrib_ids = dedup_keep_order([_sid for (_lem, _sid) in filtered if _sid])

        # assemble choices
        choices = [gold_ans] + distractor_lemmas
        choices = [c for c in choices if valid_for_lang(target_lang_upper, c)]
        choices = dedup_keep_order([clean_space(nfc(c)) for c in choices if c and c.strip()])

        # enforce caps / exact requirement
        if require_exact:
            if len(choices) != max_choices:
                continue
        else:
            if len(choices) < 2:
                continue
            choices = choices[:max_choices]

        rng.shuffle(choices)
        if gold_ans not in choices:
            if require_exact:
                continue
            if len(choices) >= max_choices:
                choices[-1] = gold_ans
            else:
                choices.append(gold_ans)
            rng.shuffle(choices)
            if gold_ans not in choices:
                continue

        answer_idx = choices.index(gold_ans)
        gold_sid = synset_id_str(gold)

        # ----- EXACT FIELD ORDER -----
        question = f'What is the correct {target_lang_upper} meaning of the word \'{source_word}\' in the sentence: "{context_sentence}"?'
        item = {}
        item["task_id"] = f"clwsd_{source_word}_{target_lang_upper}_{gold_sid}_{task_counter_start + len(items):05d}"
        item["task_type"] = "cross_lingual_wsd"
        item["source_word"] = source_word
        item["source_lang"] = "EN"
        item["target_lang"] = target_lang_upper
        item["context_sentence"] = context_sentence
        item["question"] = question
        item["choices"] = choices
        item["answer"] = gold_ans
        item["meta"] = {
            "gold_synset_id": gold_sid,
            "distractor_synset_ids_used": contrib_ids,
            "pos": (pos or None),
            "example_source": example_origin,
            "answer_idx": answer_idx,
            "generator": {"version": GENERATOR_VERSION, "max_choices": max_choices},
        }
        items.append(item)
        added += 1
        if max_senses_per_pair and added >= max_senses_per_pair:
            break

    return items

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=Path, required=True, help="Path to words.txt (word[,|\\t]POS)")
    ap.add_argument("--targets", type=str, default="", help="Comma-separated target language codes (e.g., FR,DE,IT,JA). If set, overrides --tier.")
    ap.add_argument("--tier", type=str, default="", choices=["high_resource","medium_resource","low_resource"],
                    help="Pick targets from language_config.py by tier (lowercase codes in file → uppercased here).")
    ap.add_argument("--language-config", type=Path, default=Path("language_config.py"),
                    help="Path to language_config.py (default: ./language_config.py)")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    ap.add_argument("--max-choices", type=int, default=50, help="Total options per item (default: 50)")
    ap.add_argument("--max-items", type=int, default=10_000, help="Overall cap across all targets (ignored in --exhaustive)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    ap.add_argument("--log", type=str, default="INFO", help="Logging level")
    # Exhaustive mode + per-lang cap
    ap.add_argument("--exhaustive", action="store_true",
                    help="Try every (word × target_lang) pair; ignore max-items and stop only after a full sweep.")
    ap.add_argument("--max-per-lang", type=int, default=0,
                    help="Optional cap per target language in exhaustive mode (0 = no per-lang cap).")
    # Exact choice count enforcement
    ap.add_argument("--require-exact-choices", action="store_true",
                    help="Only keep items that reach exactly --max-choices options.")
    # Limit items per (word × target language)
    ap.add_argument("--max-senses-per-pair", type=int, default=0,
                    help="Limit items per (word×target) pair; 0 = all senses (default), e.g., 1 or 2 to cap.")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")
    rng = random.Random(args.seed)

    # Resolve targets
    if args.targets.strip():
        targets = [t.strip().upper() for t in args.targets.split(",") if t.strip()]
    elif args.tier:
        try:
            targets = load_targets_from_config(args.language_config, args.tier)
        except Exception as e:
            logging.error(f"Failed to load targets from {args.language_config}: {e}")
            sys.exit(1)
    else:
        logging.error("Provide either --targets or --tier (with --language-config).")
        sys.exit(1)

    if not targets:
        logging.error("No target languages resolved.")
        sys.exit(1)

    words = parse_words_file(args.words)
    if not words:
        logging.error("No words found in --words file.")
        sys.exit(1)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        if args.exhaustive:
            # Full sweep over all target languages and all words
            for tgt in targets:
                per_lang_written = 0
                rng.shuffle(words)  # shuffle once per language for variety
                for ws in words:
                    items = build_items_for_pair(
                        task_counter_start=total_written + 1,
                        source_word=ws.word,
                        pos=ws.pos,
                        target_lang_upper=tgt,
                        max_choices=args.max_choices,
                        require_exact=args.require_exact_choices,
                        max_senses_per_pair=args.max_senses_per_pair,
                        rng=rng,
                    )
                    for it in items:
                        if args.require_exact_choices and len(it["choices"]) != args.max_choices:
                            continue
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")
                        total_written += 1
                        per_lang_written += 1
                        if args.max_per_lang and per_lang_written >= args.max_per_lang:
                            break
                    if args.max_per_lang and per_lang_written >= args.max_per_lang:
                        break
            logging.info(f"Wrote {total_written} items to {out_path} (exhaustive sweep).")
        else:
            # Capped, round-robin mode
            while total_written < args.max_items:
                progressed = False
                for tgt in targets:
                    rng.shuffle(words)
                    for ws in words:
                        items = build_items_for_pair(
                            task_counter_start=total_written + 1,
                            source_word=ws.word,
                            pos=ws.pos,
                            target_lang_upper=tgt,
                            max_choices=args.max_choices,
                            require_exact=args.require_exact_choices,
                            max_senses_per_pair=args.max_senses_per_pair,
                            rng=rng,
                        )
                        wrote_any = False
                        for it in items:
                            if args.require_exact_choices and len(it["choices"]) != args.max_choices:
                                continue
                            f.write(json.dumps(it, ensure_ascii=False) + "\n")
                            total_written += 1
                            wrote_any = True
                            if total_written >= args.max_items:
                                break
                        if wrote_any:
                            progressed = True
                        if total_written >= args.max_items:
                            break
                    if total_written >= args.max_items:
                        break
                if not progressed:
                    break
            logging.info(f"Wrote {total_written} items to {out_path} (capped mode).")

if __name__ == "__main__":
    main()
