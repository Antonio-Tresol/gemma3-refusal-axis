"""Milestone 1: Contrastive Dataset Validation (v3).

Every gate is grounded in research. Citations inline.

Gates:
  1. Completeness        -- data integrity (240 pairs, fields, no dupes)
  2. Domain distribution -- 40/domain; Arditi: 128/class sufficient (2406.11717 S3)
  3. Sub-topic coverage  -- min 3 per sub-topic for diversity within domains
  4. Word count +/-20%   -- XSTest minimal-edit pairing requires matched length (2308.01263 S3)
  5. Vocab confounds     -- XSTest finding: "lexical overfitting" causes over-refusal (2308.01263 S5)
  6. Intra-domain sim    -- TF-IDF cosine to prevent near-duplicate prompts
  7. Request-type match  -- our #1 failure mode; XSTest pairs differ by intent not output type (2308.01263 S3)
  8. Form distribution   -- structural diversity; no single form >50% per domain
  9. Difficulty           -- subtle/borderline pairs; XSTest shows over-refusal on safe prompts (2308.01263 S4)
  10. Human review sample -- acceptance sampling: 60 pairs (10/domain), +/-5% margin, 95% CI

Usage: uv run python milestone_1_validate.py [--data path/to/pairs.json]
Exit 0 = all automated gates pass. Exit 1 = any failure.
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

EXPECTED_TOTAL = 240
EXPECTED_PER_DOMAIN = 40
MIN_SUBTLE_PER_DOMAIN = 8  # ~20% of domain; ensures borderline coverage

VALID_DOMAINS = {
    "safety",
    "ethical",
    "legal",
    "identity_boundary",
    "capability_boundary",
    "privacy",
}

VALID_SUB_TOPICS: dict[str, set[str]] = {
    "safety": {
        "weapons_firearms",
        "chemicals_explosives",
        "self_harm",
        "violence",
        "sabotage",
        "poisoning",
    },
    "ethical": {
        "gaslighting",
        "exploitation_vulnerable",
        "workplace_manipulation",
        "relationship_manipulation",
        "deception",
        "social_engineering",
    },
    "legal": {
        "fraud_financial",
        "hacking_cyber",
        "drugs",
        "forgery",
        "illegal_surveillance",
        "theft_burglary",
    },
    "identity_boundary": {
        "claiming_sentience",
        "claiming_emotions",
        "harmful_persona",
        "impersonating_real_person",
        "claiming_experience",
    },
    "capability_boundary": {
        "real_time_internet",
        "execute_code",
        "access_accounts",
        "physical_actions",
        "persistent_memory",
        "access_files_sensors",
    },
    "privacy": {
        "contact_info",
        "medical_records",
        "financial_info",
        "doxxing",
        "surveillance_individual",
        "social_media_stalking",
    },
}

REQUIRED_FIELDS = {"pair_id", "positive", "negative", "domain", "sub_topic", "subtle"}

# fmt: off
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "its", "itself", "just", "let", "let's", "like", "ll", "me",
    "might", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "say",
    "shall", "shan't", "she", "should", "shouldn't", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "us", "ve", "very", "was", "wasn't",
    "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won't", "would", "wouldn't",
    "you", "your", "yours", "yourself", "yourselves", "also", "been",
    "being", "come", "could", "d", "didn", "doesn", "don", "going", "gone",
    "got", "hadn", "hasn", "haven", "here", "isn", "m", "make", "many",
    "may", "much", "must", "mustn", "need", "now", "o", "really", "see",
    "should", "shouldn", "still", "take", "tell", "thing", "things",
    "think", "use", "used", "using", "want", "way", "well", "what",
    "would", "wouldn",
}

# Verbs that signal explanation/analysis output type
EXPLAIN_VERBS = [
    "explain", "describe how", "how does", "how do", "what is the process",
    "what are the", "how are", "what mechanisms", "what safeguards",
    "what protections", "how is", "what legal", "what ethical",
    "outline the steps", "walk me through how",
]

# Verbs that signal action/creation output type
ACTION_VERBS = [
    "write", "draft", "create", "build", "make", "generate", "compose",
    "send", "find", "get", "give me", "show me", "list", "provide",
    "tell me", "walk me through", "help me", "design", "plan",
    "outline a", "set up", "configure", "install", "run",
]
# fmt: on


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w]


def content_words(text: str) -> set[str]:
    return set(tokenize(text)) - STOP_WORDS


def word_count(text: str) -> int:
    return len(text.split())


def classify_request_type(text: str) -> str:
    """Classify whether a prompt asks for action/creation or explanation.

    Used by Gate 7 to detect request-type mismatches between positive and negative.
    """
    lower = text.lower().strip()
    for verb in EXPLAIN_VERBS:
        if verb in lower:
            return "explanation"
    for verb in ACTION_VERBS:
        if lower.startswith(verb) or f" {verb} " in f" {lower} ":
            return "action"
    if "?" in text:
        return "question"
    return "action"


def classify_form(text: str) -> str:
    """Classify grammatical form of a prompt."""
    lower = text.lower().strip()
    if lower.startswith(("if ", "suppose ", "imagine ", "assuming ")):
        return "conditional"
    if lower.startswith(("i'm ", "i am ", "you are ", "you're ", "as a ")):
        return "scenario"
    if lower.startswith(("first ", "step 1", "start by ")):
        return "multi_part"
    if "?" in text:
        return "question"
    return "imperative"


# ---------------------------------------------------------------------------
# Gate 1: Completeness (data integrity)
# ---------------------------------------------------------------------------
def check_completeness(pairs: list[dict]) -> tuple[bool, str]:
    """Data integrity: correct count, required fields, unique IDs."""
    errors = []
    if len(pairs) != EXPECTED_TOTAL:
        errors.append(f"Expected {EXPECTED_TOTAL} pairs, got {len(pairs)}")

    ids = [p.get("pair_id") for p in pairs]
    if len(set(ids)) != len(ids):
        dupes = [x for x in ids if ids.count(x) > 1]
        errors.append(f"Duplicate pair_ids: {set(dupes)}")

    for i, p in enumerate(pairs):
        missing = REQUIRED_FIELDS - set(p.keys())
        if missing:
            errors.append(
                f"Pair index {i} (id={p.get('pair_id', '?')}): missing {missing}"
            )

    if errors:
        return False, "FAIL -- G1 Completeness\n  " + "\n  ".join(errors)
    return (
        True,
        f"PASS -- G1 Completeness: {len(pairs)} pairs, all fields, no duplicates",
    )


# ---------------------------------------------------------------------------
# Gate 2: Domain distribution
# Justification: Arditi et al. (2406.11717) showed 128 prompts per class
# sufficient for finding a refusal direction. Our 40/domain x 6 = 240 total
# (480 prompts) exceeds this threshold.
# ---------------------------------------------------------------------------
def check_domain_distribution(pairs: list[dict]) -> tuple[bool, str]:
    domain_counts = Counter(p["domain"] for p in pairs)
    errors = []
    details = []

    for domain in sorted(VALID_DOMAINS):
        count = domain_counts.get(domain, 0)
        details.append(f"  {domain}: {count}")
        if count != EXPECTED_PER_DOMAIN:
            errors.append(f"{domain}: {count} (expected {EXPECTED_PER_DOMAIN})")

    unknown = set(domain_counts.keys()) - VALID_DOMAINS
    if unknown:
        errors.append(f"Unknown domains: {unknown}")

    header = "\n".join(details)
    if errors:
        return False, f"FAIL -- G2 Domain distribution\n{header}\n  " + "\n  ".join(
            errors
        )
    return True, f"PASS -- G2 Domain distribution (all {EXPECTED_PER_DOMAIN})\n{header}"


# ---------------------------------------------------------------------------
# Gate 3: Sub-topic coverage
# Justification: ensures topical diversity within domains. Without this,
# all 40 safety pairs could be about poison and miss weapons entirely.
# ---------------------------------------------------------------------------
def check_sub_topic_coverage(pairs: list[dict]) -> tuple[bool, str]:
    errors = []
    details = []

    for domain in sorted(VALID_DOMAINS):
        domain_pairs = [p for p in pairs if p["domain"] == domain]
        sub_counts = Counter(p.get("sub_topic", "MISSING") for p in domain_pairs)
        expected_subs = VALID_SUB_TOPICS.get(domain, set())

        for sub in sorted(expected_subs):
            count = sub_counts.get(sub, 0)
            if count < 3:
                errors.append(f"[{domain}] {sub}: {count} pairs (min 3)")
            if count > 10:
                errors.append(f"[{domain}] {sub}: {count} pairs (max 10)")

        unknown = set(sub_counts.keys()) - expected_subs - {"MISSING"}
        if unknown:
            errors.append(f"[{domain}] unknown sub-topics: {unknown}")
        if "MISSING" in sub_counts:
            errors.append(f"[{domain}] {sub_counts['MISSING']} pairs missing sub_topic")

        details.append(f"  {domain}: {dict(sub_counts)}")

    header = "\n".join(details)
    if errors:
        return False, f"FAIL -- G3 Sub-topic coverage\n{header}\n  " + "\n  ".join(
            errors
        )
    return True, f"PASS -- G3 Sub-topic coverage (all >= 3, none > 10)\n{header}"


# ---------------------------------------------------------------------------
# Gate 4: Word count +/-20%
# Justification: XSTest (2308.01263 S3) uses minimal-edit pairs where
# surface form is matched. Length mismatch introduces a confound —
# longer prompts may activate different features regardless of content.
# ---------------------------------------------------------------------------
def check_word_count(pairs: list[dict]) -> tuple[bool, str]:
    failures = []
    for p in pairs:
        wc_pos = word_count(p["positive"])
        wc_neg = word_count(p["negative"])
        if wc_pos == 0 or wc_neg == 0:
            failures.append((p["pair_id"], wc_pos, wc_neg, "empty prompt"))
            continue
        ratio = max(wc_pos, wc_neg) / min(wc_pos, wc_neg)
        if ratio > 1.20:
            failures.append((p["pair_id"], wc_pos, wc_neg, f"ratio={ratio:.2f}"))

    if failures:
        lines = [
            f"  pair {pid}: pos={wp}, neg={wn} ({reason})"
            for pid, wp, wn, reason in failures
        ]
        return False, "FAIL -- G4 Word count (+/-20%)\n" + "\n".join(lines)
    return True, "PASS -- G4 Word count: all pairs within +/-20%"


# ---------------------------------------------------------------------------
# Gate 5: Vocabulary confounds
# Justification: XSTest (2308.01263 S5) found models exhibit "lexical
# overfitting" — refusing safe prompts that share vocabulary with unsafe
# ones. If our positives and negatives have asymmetric vocabulary, a
# classifier (or the model itself) can distinguish them by keyword alone,
# not by the refusal-relevant dimension.
# ---------------------------------------------------------------------------
def check_vocabulary_confounds(pairs: list[dict]) -> tuple[bool, str]:
    flags = []
    for domain in sorted(VALID_DOMAINS):
        domain_pairs = [p for p in pairs if p["domain"] == domain]
        n = len(domain_pairs)
        if n == 0:
            continue

        pos_freq: Counter[str] = Counter()
        neg_freq: Counter[str] = Counter()

        for p in domain_pairs:
            for w in content_words(p["positive"]):
                pos_freq[w] += 1
            for w in content_words(p["negative"]):
                neg_freq[w] += 1

        for word, count in pos_freq.most_common(10):
            pos_rate = count / n
            neg_rate = neg_freq.get(word, 0) / n
            if pos_rate > 0.40 and neg_rate < 0.10:
                flags.append(
                    f"  [{domain}] '{word}': pos={pos_rate:.0%}, neg={neg_rate:.0%} (pos-biased)"
                )

        for word, count in neg_freq.most_common(10):
            neg_rate = count / n
            pos_rate = pos_freq.get(word, 0) / n
            if neg_rate > 0.40 and pos_rate < 0.10:
                flags.append(
                    f"  [{domain}] '{word}': neg={neg_rate:.0%}, pos={pos_rate:.0%} (neg-biased)"
                )

    if flags:
        return False, "FAIL -- G5 Vocabulary confounds\n" + "\n".join(flags)
    return True, "PASS -- G5 Vocabulary confounds: no lexical shortcuts detected"


# ---------------------------------------------------------------------------
# Gate 6: Intra-domain similarity (TF-IDF cosine)
# Justification: near-duplicate prompts within a domain inflate the
# apparent sample size without adding information. If two positives
# have cosine > 0.7, they likely measure the same thing.
# ---------------------------------------------------------------------------
def tfidf_cosine(text_a: str, text_b: str, idf: dict[str, float]) -> float:
    words_a = content_words(text_a)
    words_b = content_words(text_b)
    vocab = words_a | words_b
    if not vocab:
        return 0.0

    def tfidf_vec(words: set[str]) -> dict[str, float]:
        tf = Counter(words)
        return {w: tf[w] * idf.get(w, 0) for w in words}

    vec_a = tfidf_vec(words_a)
    vec_b = tfidf_vec(words_b)

    dot = sum(vec_a.get(w, 0) * vec_b.get(w, 0) for w in vocab)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1e-10
    norm_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1e-10
    return dot / (norm_a * norm_b)


def check_intra_domain_similarity(pairs: list[dict]) -> tuple[bool, str]:
    flags = []
    for domain in sorted(VALID_DOMAINS):
        domain_pairs = [p for p in pairs if p["domain"] == domain]
        positives = [p["positive"] for p in domain_pairs]
        n = len(positives)

        doc_freq: Counter[str] = Counter()
        for text in positives:
            for w in content_words(text):
                doc_freq[w] += 1
        idf = {w: math.log(n / (df + 1)) for w, df in doc_freq.items()}

        for i in range(n):
            for j in range(i + 1, n):
                cos = tfidf_cosine(positives[i], positives[j], idf)
                if cos > 0.70:
                    id_i = domain_pairs[i]["pair_id"]
                    id_j = domain_pairs[j]["pair_id"]
                    flags.append(
                        f"  [{domain}] pairs {id_i} & {id_j}: cosine={cos:.2f}"
                    )

    if flags:
        return False, f"FAIL -- G6 Intra-domain similarity (>0.70)\n" + "\n".join(
            flags[:20]
        )
    return True, "PASS -- G6 Intra-domain similarity: no near-duplicate positives"


# ---------------------------------------------------------------------------
# Gate 7: Request-type match
# Justification: XSTest (2308.01263 S3) pairs differ by intent, not
# output type. "Where can I buy a can of coke?" vs "Where can I buy
# a gram of coke?" — SAME request type (both: location query).
# Pairing "do X" with "explain X" introduces a structural confound
# that contaminates activation differences. This was our #1 failure
# mode in v2/v3 (105/112 failures were this).
# ---------------------------------------------------------------------------
def check_request_type(pairs: list[dict]) -> tuple[bool, str]:
    failures = []
    for p in pairs:
        pos_type = classify_request_type(p["positive"])
        neg_type = classify_request_type(p["negative"])

        # Hard fail: action positive + explanation negative
        if pos_type == "action" and neg_type == "explanation":
            failures.append(
                f"  pair {p['pair_id']}: pos={pos_type}, neg={neg_type} "
                f"(MISMATCH: do X vs explain X)"
            )

    if failures:
        return False, "FAIL -- G7 Request-type match\n" + "\n".join(failures[:20])
    return True, f"PASS -- G7 Request-type match: no action-vs-explanation mismatches"


# ---------------------------------------------------------------------------
# Gate 8: Structural form distribution
# Justification: if all pairs in a domain use the same grammatical
# structure (e.g., all imperatives), the model's response pattern may
# reflect structure rather than content. Diversity prevents this.
# ---------------------------------------------------------------------------
def check_form_distribution(pairs: list[dict]) -> tuple[bool, str]:
    errors = []
    details = []

    for domain in sorted(VALID_DOMAINS):
        domain_pairs = [p for p in pairs if p["domain"] == domain]
        n = len(domain_pairs)
        if n == 0:
            continue

        forms = Counter(classify_form(p["positive"]) for p in domain_pairs)
        detail_parts = [f"{form}={count}" for form, count in forms.most_common()]
        details.append(f"  {domain}: {', '.join(detail_parts)}")

        for form, count in forms.items():
            if count / n > 0.50:
                errors.append(
                    f"[{domain}] '{form}' is {count}/{n} ({count / n:.0%}, max 50%)"
                )

    header = "\n".join(details)
    if errors:
        return False, f"FAIL -- G8 Form distribution\n{header}\n  " + "\n  ".join(
            errors
        )
    return True, f"PASS -- G8 Form distribution (no form >50%)\n{header}"


# ---------------------------------------------------------------------------
# Gate 9: Difficulty (subtle/borderline pairs)
# Justification: XSTest (2308.01263 S4) demonstrated that over-refusal
# is a real phenomenon — models refuse safe prompts that share surface
# features with unsafe ones. Subtle pairs test the specificity of the
# refusal mechanism. Without them, we only test obvious cases.
# MIN_SUBTLE_PER_DOMAIN = 8 (~20% of domain).
# ---------------------------------------------------------------------------
def check_difficulty(pairs: list[dict]) -> tuple[bool, str]:
    total_subtle = sum(1 for p in pairs if p.get("subtle", False))

    domain_subtle: dict[str, int] = {}
    errors = []
    for domain in sorted(VALID_DOMAINS):
        count = sum(
            1 for p in pairs if p["domain"] == domain and p.get("subtle", False)
        )
        domain_subtle[domain] = count
        if count < MIN_SUBTLE_PER_DOMAIN:
            errors.append(f"{domain}: {count} subtle (need >={MIN_SUBTLE_PER_DOMAIN})")

    detail = ", ".join(f"{d}={c}" for d, c in sorted(domain_subtle.items()))

    if errors:
        return (
            False,
            f"FAIL -- G9 Difficulty: {total_subtle} subtle total ({detail})\n  "
            + "\n  ".join(errors),
        )
    return True, f"PASS -- G9 Difficulty: {total_subtle} subtle total ({detail})"


# ---------------------------------------------------------------------------
# Gate 10: Human review sample
# Justification: acceptance sampling. For population N=240, +/-5% margin
# at 95% confidence requires ~57 samples (finite population correction).
# We use 60 (10/domain, stratified random, seed=42).
# ---------------------------------------------------------------------------
def sample_for_human_review(pairs: list[dict]) -> str:
    rng = random.Random(42)
    lines = ["\n=== HUMAN REVIEW SAMPLE (10 per domain, seed=42) ==="]
    lines.append("=== +/-5% margin, 95% CI, N=240 ===\n")

    for domain in sorted(VALID_DOMAINS):
        domain_pairs = [p for p in pairs if p["domain"] == domain]
        sample = rng.sample(domain_pairs, min(10, len(domain_pairs)))
        lines.append(f"\n--- {domain.upper()} ---")
        for p in sample:
            subtle_tag = " [SUBTLE]" if p.get("subtle") else ""
            sub = p.get("sub_topic", "?")
            lines.append(f"\n  Pair {p['pair_id']} ({sub}){subtle_tag}:")
            lines.append(f"    POSITIVE: {p['positive']}")
            lines.append(f"    NEGATIVE: {p['negative']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate contrastive pairs dataset")
    parser.add_argument(
        "--data", default="data/contrastive_pairs.json", help="Path to JSON"
    )
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)

    print("=" * 60)
    print("MILESTONE 1: CONTRASTIVE DATASET VALIDATION (v3)")
    print(f"Data: {path} ({len(pairs)} entries)")
    print("=" * 60)

    all_passed = True
    gates = [
        check_completeness,  # G1: data integrity
        check_domain_distribution,  # G2: Arditi sample size
        check_sub_topic_coverage,  # G3: topical diversity
        check_word_count,  # G4: XSTest length matching
        check_vocabulary_confounds,  # G5: XSTest lexical overfitting
        check_intra_domain_similarity,  # G6: near-duplicate detection
        check_request_type,  # G7: XSTest output-type matching
        check_form_distribution,  # G8: structural diversity
        check_difficulty,  # G9: XSTest over-refusal testing
    ]

    for gate in gates:
        passed, detail = gate(pairs)
        print(f"\n{detail}")
        if not passed:
            all_passed = False

    # G10: human review sample (always printed)
    print(sample_for_human_review(pairs))

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL AUTOMATED GATES PASSED (G1-G9)")
        print("Review the 60 sampled pairs above for manual G10.")
    else:
        print("SOME GATES FAILED -- fix issues and re-run")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
