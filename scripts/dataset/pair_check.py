"""Pair-level validator for contrastive prompt pairs.

Validates individual pairs and batches against all matching rules.

Usage:
  # Single pair
  uv run python pair_check.py --pos "..." --neg "..." --domain safety --sub_topic violence

  # Validate a domain JSON file (all pairs + batch checks)
  uv run python pair_check.py --file data/domain_safety.json

  # Validate the full assembled dataset
  uv run python pair_check.py --file data/contrastive_pairs.json

  # Interactive mode: paste pairs, get instant feedback
  uv run python pair_check.py --interactive
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_DOMAINS = {
    "safety",
    "ethical",
    "legal",
    "identity_boundary",
    "capability_boundary",
    "privacy",
}

VALID_SUB_TOPICS = {
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

# fmt: off
FRAMING_CRUTCHES = [
    "for a school project", "for educational purposes", "as a researcher",
    "for my research", "so our security team", "so victims can recognize",
    "for training purposes", "in an academic context", "for a class assignment",
    "to raise awareness", "for a documentary",
]

EXPLAIN_VERBS = [
    "explain", "describe how", "how does", "how do", "what is the process",
    "what are the", "how are", "what mechanisms", "what safeguards",
    "what protections", "how is", "what legal", "what ethical",
]

ACTION_VERBS = [
    "write", "draft", "create", "build", "make", "generate", "compose",
    "send", "find", "get", "give me", "show me", "list", "provide",
    "tell me", "walk me through", "help me",
]
# fmt: on


# ---------------------------------------------------------------------------
# Individual pair checks
# ---------------------------------------------------------------------------
def check_word_count(pos: str, neg: str) -> tuple[bool, str]:
    wp = len(pos.split())
    wn = len(neg.split())
    if wp == 0 or wn == 0:
        return False, "Empty prompt"
    ratio = max(wp, wn) / min(wp, wn)
    if ratio > 1.20:
        return False, f"Word count ratio {ratio:.2f} (pos={wp}, neg={wn}, max 1.20)"
    return True, f"Word count OK (pos={wp}, neg={wn}, ratio={ratio:.2f})"


def classify_request_type(text: str) -> str:
    """Classify whether a prompt asks for action/creation or explanation."""
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


def check_request_type(pos: str, neg: str) -> tuple[bool, str]:
    pos_type = classify_request_type(pos)
    neg_type = classify_request_type(neg)

    if pos_type == "action" and neg_type == "explanation":
        return (
            False,
            f"REQUEST-TYPE MISMATCH: pos={pos_type}, neg={neg_type} (do X vs explain X)",
        )

    if pos_type != neg_type:
        return (
            False,
            f"Request type differs: pos={pos_type}, neg={neg_type} (check manually)",
        )

    return True, f"Request type: both={pos_type}"


def classify_form(text: str) -> str:
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


def check_grammatical_structure(pos: str, neg: str) -> tuple[bool, str]:
    pos_form = classify_form(pos)
    neg_form = classify_form(neg)

    if pos_form != neg_form:
        return False, f"Grammar mismatch: pos={pos_form}, neg={neg_form}"
    return True, f"Grammar: both={pos_form}"


def check_framing_crutches(neg: str) -> tuple[bool, str]:
    lower = neg.lower()
    for crutch in FRAMING_CRUTCHES:
        if crutch in lower:
            return False, f"Framing crutch in negative: '{crutch}'"
    return True, "No framing crutches"


def check_single_word_swap(pos: str, neg: str) -> tuple[bool, str]:
    pos_words = pos.lower().split()
    neg_words = neg.lower().split()

    if len(pos_words) != len(neg_words):
        return True, "Different lengths (not a swap)"

    diffs = sum(1 for a, b in zip(pos_words, neg_words) if a != b)
    if diffs <= 1:
        return False, f"Single-word swap detected ({diffs} word differs)"
    if diffs == 2:
        return False, f"Near single-word swap ({diffs} words differ)"
    return True, f"Sufficient divergence ({diffs} words differ)"


def check_domain_sub_topic(domain: str, sub_topic: str) -> tuple[bool, str]:
    if domain not in VALID_DOMAINS:
        return False, f"Invalid domain: {domain}"
    valid = VALID_SUB_TOPICS.get(domain, set())
    if sub_topic not in valid:
        return (
            False,
            f"Invalid sub_topic '{sub_topic}' for domain '{domain}'. Valid: {sorted(valid)}",
        )
    return True, f"Domain/sub_topic valid"


# ---------------------------------------------------------------------------
# Full pair validation
# ---------------------------------------------------------------------------
def validate_pair(pair: dict) -> list[tuple[bool, str]]:
    """Run all checks on a single pair. Returns list of (passed, message)."""
    results = []

    pos = pair.get("positive", "")
    neg = pair.get("negative", "")
    domain = pair.get("domain", "")
    sub_topic = pair.get("sub_topic", "")

    results.append(check_word_count(pos, neg))
    results.append(check_request_type(pos, neg))
    results.append(check_grammatical_structure(pos, neg))
    results.append(check_framing_crutches(neg))
    results.append(check_single_word_swap(pos, neg))
    results.append(check_domain_sub_topic(domain, sub_topic))

    return results


def print_pair_report(pair: dict, results: list[tuple[bool, str]]) -> bool:
    pair_id = pair.get("pair_id", "?")
    domain = pair.get("domain", "?")
    sub_topic = pair.get("sub_topic", "?")
    subtle = pair.get("subtle", False)

    all_pass = all(r[0] for r in results)
    status = "PASS" if all_pass else "FAIL"

    print(
        f"\n  [{status}] Pair {pair_id} | {domain}/{sub_topic}"
        + (" [subtle]" if subtle else "")
    )
    print(f"    POS: {pair.get('positive', '')[:90]}...")
    print(f"    NEG: {pair.get('negative', '')[:90]}...")

    for passed, msg in results:
        marker = "  ok" if passed else "  FAIL"
        print(f"    {marker}: {msg}")

    return all_pass


# ---------------------------------------------------------------------------
# Batch checks (across multiple pairs in same domain)
# ---------------------------------------------------------------------------
def check_batch(pairs: list[dict]) -> list[str]:
    """Run batch-level checks: template repetition, sub-topic coverage, form distribution."""
    issues = []

    # Group by domain
    by_domain: dict[str, list[dict]] = {}
    for p in pairs:
        by_domain.setdefault(p.get("domain", "?"), []).append(p)

    for domain, domain_pairs in sorted(by_domain.items()):
        n = len(domain_pairs)

        # Template repetition (4+ word opening)
        openings: dict[str, list[int]] = {}
        for p in domain_pairs:
            words = p["positive"].split()[:4]
            key = " ".join(w.lower() for w in words)
            openings.setdefault(key, []).append(p.get("pair_id", 0))
        for key, ids in openings.items():
            if len(ids) > 1:
                issues.append(f"[{domain}] Template repetition '{key}...': pairs {ids}")

        # Sub-topic coverage
        from collections import Counter

        sub_counts = Counter(p.get("sub_topic", "MISSING") for p in domain_pairs)
        expected = VALID_SUB_TOPICS.get(domain, set())
        for sub in sorted(expected):
            count = sub_counts.get(sub, 0)
            if count < 3:
                issues.append(f"[{domain}] {sub}: {count} pairs (min 3)")
            if count > 10:
                issues.append(f"[{domain}] {sub}: {count} pairs (max 10)")

        # Form distribution
        forms = Counter(classify_form(p["positive"]) for p in domain_pairs)
        for form, count in forms.items():
            if n > 0 and count / n > 0.50:
                issues.append(
                    f"[{domain}] Form '{form}' is {count}/{n} ({count / n:.0%}, max 50%)"
                )

        # Subtle count
        subtle_count = sum(1 for p in domain_pairs if p.get("subtle", False))
        if subtle_count < 8:
            issues.append(f"[{domain}] Only {subtle_count} subtle pairs (min 8)")

    return issues


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def mode_single(args: argparse.Namespace) -> int:
    pair = {
        "pair_id": 0,
        "positive": args.pos,
        "negative": args.neg,
        "domain": args.domain,
        "sub_topic": args.sub_topic or "",
        "subtle": args.subtle,
    }
    results = validate_pair(pair)
    all_pass = print_pair_report(pair, results)
    return 0 if all_pass else 1


def mode_file(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"Validating {len(pairs)} pairs from {path}")
    print("=" * 60)

    pass_count = 0
    fail_count = 0
    fail_ids = []

    for pair in pairs:
        results = validate_pair(pair)
        all_pass = print_pair_report(pair, results)
        if all_pass:
            pass_count += 1
        else:
            fail_count += 1
            fail_ids.append(pair.get("pair_id", "?"))

    # Batch checks
    print("\n" + "=" * 60)
    print("BATCH CHECKS")
    print("=" * 60)
    batch_issues = check_batch(pairs)
    if batch_issues:
        for issue in batch_issues:
            print(f"  FAIL: {issue}")
    else:
        print("  All batch checks pass")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {pass_count} PASS, {fail_count} FAIL")
    if fail_ids:
        print(f"Failing pair IDs: {fail_ids}")
    print("=" * 60)

    return 0 if fail_count == 0 and not batch_issues else 1


def mode_interactive(_args: argparse.Namespace) -> int:
    print("Pair Validator (interactive)")
    print("Enter pairs as JSON objects, one per line. Ctrl+C to quit.\n")

    while True:
        try:
            line = input("pair> ").strip()
            if not line:
                continue
            pair = json.loads(line)
            results = validate_pair(pair)
            print_pair_report(pair, results)
            print()
        except json.JSONDecodeError as e:
            print(f"  Invalid JSON: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nDone.")
            return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate contrastive prompt pairs")
    sub = parser.add_subparsers(dest="mode")

    # Single pair
    single = sub.add_parser("single", help="Validate one pair")
    single.add_argument("--pos", required=True, help="Positive prompt")
    single.add_argument("--neg", required=True, help="Negative prompt")
    single.add_argument("--domain", required=True)
    single.add_argument("--sub_topic", default="")
    single.add_argument("--subtle", action="store_true")

    # File
    file_p = sub.add_parser("file", help="Validate a JSON file")
    file_p.add_argument("path", help="Path to JSON file")

    # Interactive
    sub.add_parser("interactive", help="Interactive mode")

    args = parser.parse_args()

    if args.mode == "single":
        return mode_single(args)
    elif args.mode == "file":
        args.file = args.path
        return mode_file(args)
    elif args.mode == "interactive":
        return mode_interactive(args)
    else:
        # Default: if --file is given as positional
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
