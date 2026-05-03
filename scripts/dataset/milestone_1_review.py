"""Milestone 1: Human Review TUI.

Stratified random sample of contrastive pairs for blind human review.
102 pairs (17 per domain), shuffled. Progress saves automatically.

Usage: uv run python milestone_1_review.py [--data data/contrastive_pairs.json] [--all]
"""

import argparse
import difflib
import json
import os
import random
import sys
from pathlib import Path

VALID_DOMAINS = [
    "capability_boundary",
    "ethical",
    "identity_boundary",
    "legal",
    "privacy",
    "safety",
]

SAMPLE_PER_DOMAIN = 10  # 10 per domain x 6 = 60 ~= 57 for +/-5% margin, 95% CI
BREAK_EVERY = 15
REVIEW_OUTPUT = Path("findings/milestone_1_human_review.json")

# --- ANSI colors ---
DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Word-level diff
# ---------------------------------------------------------------------------
def highlight_diff(pos: str, neg: str) -> tuple[str, str]:
    """Highlight words that differ between pos and neg."""
    pos_words = pos.split()
    neg_words = neg.split()

    matcher = difflib.SequenceMatcher(None, pos_words, neg_words)
    pos_highlighted = []
    neg_highlighted = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            pos_highlighted.extend(pos_words[i1:i2])
            neg_highlighted.extend(neg_words[j1:j2])
        elif op == "replace":
            for w in pos_words[i1:i2]:
                pos_highlighted.append(f"{RED}{BOLD}{w}{RESET}")
            for w in neg_words[j1:j2]:
                neg_highlighted.append(f"{GREEN}{BOLD}{w}{RESET}")
        elif op == "delete":
            for w in pos_words[i1:i2]:
                pos_highlighted.append(f"{RED}{BOLD}{w}{RESET}")
        elif op == "insert":
            for w in neg_words[j1:j2]:
                neg_highlighted.append(f"{GREEN}{BOLD}{w}{RESET}")

    return " ".join(pos_highlighted), " ".join(neg_highlighted)


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
def progress_bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = done / total * 100 if total > 0 else 0
    return f"[{bar}] {pct:.0f}%"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def get_review_pairs(pairs: list[dict], review_all: bool) -> list[dict]:
    """Stratified random sample: 17 per domain = ~102 total. Blind."""
    if review_all:
        return pairs

    rng = random.Random(42)
    sample = []

    for domain in sorted(VALID_DOMAINS):
        domain_pairs = [p for p in pairs if p["domain"] == domain]
        n = min(SAMPLE_PER_DOMAIN, len(domain_pairs))
        sample.extend(rng.sample(domain_pairs, n))

    rng.shuffle(sample)
    return sample


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_pair(pair: dict, pos_hl: str, neg_hl: str) -> str:
    wc_pos = word_count(pair["positive"])
    wc_neg = word_count(pair["negative"])
    ratio = max(wc_pos, wc_neg) / min(wc_pos, wc_neg) if min(wc_pos, wc_neg) > 0 else 0
    subtle_tag = f" {YELLOW}[SUBTLE]{RESET}" if pair.get("subtle") else ""
    sub = pair.get("sub_topic", "?")

    lines = [
        f"  {DIM}Domain:{RESET} {CYAN}{pair['domain']}{RESET}/{sub}{subtle_tag}",
        f"  {DIM}Words:{RESET} pos={wc_pos}, neg={wc_neg}, ratio={ratio:.2f}",
        "",
        f"  {RED}POSITIVE{RESET} {DIM}(should be REFUSED):{RESET}",
        f"    {pos_hl}",
        "",
        f"  {GREEN}NEGATIVE{RESET} {DIM}(should be ANSWERED):{RESET}",
        f"    {neg_hl}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------
def run_review(pairs: list[dict], review_all: bool) -> None:
    to_review = get_review_pairs(pairs, review_all)
    total = len(to_review)
    results: list[dict] = []

    # Load existing progress
    if REVIEW_OUTPUT.exists():
        with open(REVIEW_OUTPUT, encoding="utf-8") as f:
            existing = json.load(f)
        reviewed_ids = {r["pair_id"] for r in existing}
        results = existing
    else:
        reviewed_ids = set()

    remaining = [p for p in to_review if p["pair_id"] not in reviewed_ids]
    if not remaining:
        print("  All pairs already reviewed!")
        show_summary(results, total)
        return

    passed_so_far = sum(1 for r in results if r["verdict"] == "PASS")
    failed_so_far = sum(1 for r in results if r["verdict"] == "FAIL")
    reviewed_count = len(results)

    for i, pair in enumerate(remaining):
        # Break reminder
        if i > 0 and i % BREAK_EVERY == 0:
            clear()
            print()
            print(f"  {YELLOW}--- BREAK TIME ---{RESET}")
            print(f"  You've reviewed {BREAK_EVERY} pairs. Take a moment.")
            print(f"  Look away from the screen for 20 seconds.")
            print()
            print(
                f"  {DIM}Progress so far:{RESET} {passed_so_far} pass, {failed_so_far} fail"
            )
            print(f"  {DIM}Remaining:{RESET} {len(remaining) - i} pairs")
            print()
            input(f"  {DIM}Press Enter when ready to continue...{RESET}")

        pos_hl, neg_hl = highlight_diff(pair["positive"], pair["negative"])

        clear()
        # Header
        print()
        print(f"  {BOLD}MILESTONE 1 — HUMAN REVIEW{RESET}")
        bar = progress_bar(reviewed_count, total)
        stats = f"{DIM}{passed_so_far}p {failed_so_far}f{RESET}"
        print(f"  {bar}  {reviewed_count}/{total}  {stats}")
        print()

        # Pair content
        print(render_pair(pair, pos_hl, neg_hl))
        print()

        # Checks (dimmed — reference, not the focus)
        print(f"  {DIM}(a) Positive should trigger refusal?{RESET}")
        print(f"  {DIM}(b) Negative is same topic but benign?{RESET}")
        print(f"  {DIM}(c) Only difference is refusal dimension?{RESET}")
        print()
        print(
            f"  {GREEN}p{RESET} pass    {RED}f{RESET} fail    {DIM}s{RESET} skip    {DIM}q{RESET} quit"
        )
        print()

        while True:
            choice = input("  > ").strip().lower()
            if choice in ("p", "f", "s", "q"):
                break
            print(f"  {DIM}Enter p, f, s, or q{RESET}")

        if choice == "q":
            save_results(results)
            print(f"\n  Saved {len(results)} reviews. Run again to resume.")
            return

        if choice == "s":
            continue

        note = ""
        if choice == "f":
            note = input(f"  {RED}Reason:{RESET} ").strip()

        results.append(
            {
                "pair_id": pair["pair_id"],
                "domain": pair["domain"],
                "sub_topic": pair.get("sub_topic", ""),
                "verdict": "PASS" if choice == "p" else "FAIL",
                "note": note,
            }
        )
        save_results(results)

        reviewed_count += 1
        if choice == "p":
            passed_so_far += 1
        else:
            failed_so_far += 1

    clear()
    show_summary(results, total)


def save_results(results: list[dict]) -> None:
    REVIEW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def show_summary(results: list[dict], total: int) -> None:
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    failed = sum(1 for r in results if r["verdict"] == "FAIL")

    clear()
    print()
    print(f"  {BOLD}REVIEW COMPLETE{RESET}")
    print(f"  {progress_bar(len(results), total)}")
    print()
    print(f"  Reviewed: {len(results)}/{total}")
    print(f"  {GREEN}PASS: {passed}{RESET}")
    print(f"  {RED}FAIL: {failed}{RESET}")
    print()

    if failed > 0:
        print(f"  {RED}Failed pairs:{RESET}")
        for r in results:
            if r["verdict"] == "FAIL":
                note = f" -- {r['note']}" if r["note"] else ""
                print(f"    Pair {r['pair_id']} [{r['domain']}]{note}")
        print()

    by_domain: dict[str, list[dict]] = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)

    print(f"  {BOLD}By domain:{RESET}")
    for domain in sorted(by_domain):
        dr = by_domain[domain]
        dp = sum(1 for r in dr if r["verdict"] == "PASS")
        df = len(dr) - dp
        bar = f"{GREEN}{'█' * dp}{RESET}{RED}{'█' * df}{RESET}"
        print(f"    {domain:<25s} {bar} {dp}/{len(dr)}")

    print()
    print(f"  {DIM}Results: {REVIEW_OUTPUT}{RESET}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Human review TUI")
    parser.add_argument("--data", default="data/contrastive_pairs.json")
    parser.add_argument("--all", action="store_true", help="Review all pairs")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)

    run_review(pairs, args.all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
