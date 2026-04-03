"""
Autonomous experiment loop for DP audit tightness research.

Adapted from Karpathy's autoresearch pattern. An LLM proposes changes to
experiment.py, the loop runs them, keeps improvements, reverts failures,
and feeds compressed history back so the LLM never repeats itself.

Can be run as CLI or imported from a notebook:

    # CLI
    OPENROUTER_API_KEY=sk-or-... python autoresearch/agent_loop.py

    # Notebook / Python
    from autoresearch.agent_loop import run_loop
    run_loop(api_key="sk-or-...", max_experiments=150)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from urllib import request, error as urlerror

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so imports work from anywhere)
# ---------------------------------------------------------------------------
AUTORESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUTORESEARCH_DIR.parent
EXPERIMENT_PY = AUTORESEARCH_DIR / "experiment.py"
PROGRAM_MD = AUTORESEARCH_DIR / "program.md"
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"
APPROACHES_LOG = AUTORESEARCH_DIR / "approaches_log.jsonl"
RUN_LOG = AUTORESEARCH_DIR / "run.log"

# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "qwen/qwen3-coder",
    "meta-llama/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemma-3-27b-it",
]

# Max prompt size in chars before switching to compressed history.
# Free models typically have 32-128K context; we stay well under.
MAX_PROMPT_CHARS = 48_000


def call_llm(
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> str | None:
    """Call OpenRouter chat completions. Returns response text or None."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dp-audit-tightness",
        "X-Title": "DP Audit Autoresearch",
    }
    req = request.Request(OPENROUTER_URL, data=payload, headers=headers, method="POST")

    for attempt in range(3):
        try:
            with request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode())
                return body["choices"][0]["message"]["content"]
        except urlerror.HTTPError as exc:
            if exc.code == 429:
                wait = min(60 * (2 ** attempt), 300)
                print(f"  [rate-limited] waiting {wait}s (retry {attempt+1}/3)")
                time.sleep(wait)
            else:
                print(f"  [HTTP {exc.code}] {str(exc)[:100]}")
                return None
        except (urlerror.URLError, OSError) as exc:
            print(f"  [network error] {str(exc)[:80]}, retrying in 15s ...")
            time.sleep(15)
    return None


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------
def syntax_check(code: str) -> str | None:
    """Return None if code compiles, else the error message."""
    try:
        compile(code, "<experiment.py>", "exec")
        return None
    except SyntaxError as exc:
        return f"Line {exc.lineno}: {exc.msg}"


def run_experiment(timeout: int) -> dict:
    """Run experiment.py, return parsed metrics dict."""
    try:
        result = subprocess.run(
            [sys.executable, str(EXPERIMENT_PY)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Exceeded {timeout}s"}

    RUN_LOG.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

    if result.returncode != 0:
        return {"status": "crash", "error": (result.stderr or result.stdout)[-500:]}

    metrics = {}
    for line in result.stdout.splitlines():
        for key in ("tightness_ratio", "epsilon_lower", "epsilon_upper",
                     "privacy_loss_gap", "score_gap", "member_favoring", "wall_seconds"):
            if line.startswith(f"{key}:"):
                val = line.split(":", 1)[1].strip()
                metrics[key] = (val == "True") if key == "member_favoring" else _float(val)

    if "tightness_ratio" not in metrics:
        return {"status": "crash", "error": "No tightness_ratio in output"}

    metrics["status"] = "ok"
    return metrics


def _float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Logging: results.tsv (flat) + approaches_log.jsonl (structured memory)
# ---------------------------------------------------------------------------
def init_results_log():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "exp_num\ttightness_ratio\twall_seconds\tstatus\tdescription\n",
            encoding="utf-8",
        )


def append_result(exp_num: int, tr: float, wall: float, status: str, desc: str):
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        f.write(f"{exp_num}\t{tr:.6f}\t{wall:.1f}\t{status}\t{desc}\n")


def log_approach(exp_num: int, desc: str, status: str, tr: float,
                 code_summary: str, error: str | None = None):
    entry = {"exp": exp_num, "desc": desc, "status": status,
             "tr": round(tr, 6), "code_summary": code_summary}
    if error:
        entry["error"] = error[:200]
    with APPROACHES_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_approaches() -> list[dict]:
    if not APPROACHES_LOG.exists():
        return []
    entries = []
    for line in APPROACHES_LOG.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def get_best_and_count(approaches: list[dict]) -> tuple[float, int]:
    """Return (best_tightness, next_exp_num) from approaches log."""
    best = 0.0
    for a in approaches:
        if a["status"] == "keep" and a["tr"] > best:
            best = a["tr"]
    next_num = max((a["exp"] for a in approaches), default=-1) + 1
    return best, next_num


# ---------------------------------------------------------------------------
# Code-change summarizer
# ---------------------------------------------------------------------------
_SCORING_KEYWORDS = {
    "cross_entropy": "negative_loss", "negative_loss": "negative_loss",
    "max_probability": "max_prob", "logit_margin": "logit_margin",
    "entropy": "entropy", "zscore": "z-score", "percentile": "percentile",
    "temperature": "temperature", "shadow": "shadow_model",
    "reference": "reference_model", "augment": "augmentation",
    "per_class": "per_class_norm", "stratif": "stratified",
    "lira": "LiRA", "neighborhood": "neighborhood",
}


def summarize_code_change(old_code: str, new_code: str) -> str:
    """Auto-detect what changed: constants, functions, scoring approach."""
    old_lines = set(old_code.strip().splitlines())
    new_lines = set(new_code.strip().splitlines())
    added = new_lines - old_lines
    parts = []

    # Constant changes
    for line in added:
        s = line.strip()
        if "=" in s and not s.startswith(("#", "def ", "return ", "if ", "for ")):
            for kw in ("QUERY_BUDGET", "SEED", "TEMPERATURE", "NUM_SEEDS",
                        "K_SHADOW", "NUM_AUGMENT", "BUDGET"):
                if kw in s:
                    parts.append(s.split("#")[0].strip())
                    break

    # New functions
    for line in added:
        s = line.strip()
        if s.startswith("def ") and s not in {l.strip() for l in old_lines}:
            parts.append(f"new:{s.split('(')[0].replace('def ', '')}()")

    # Scoring keywords
    for line in added:
        low = line.lower()
        for kw, label in _SCORING_KEYWORDS.items():
            if kw in low and label not in parts:
                parts.append(label)

    return "; ".join(parts[:6]) if parts else f"{len(added)} lines changed"


# ---------------------------------------------------------------------------
# Prompt construction — with compression to stay under context limits
# ---------------------------------------------------------------------------
def _compress_discards(discards: list[dict]) -> str:
    """Group discarded approaches by category instead of listing each one."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for a in discards:
        # Use code_summary as the category key (first keyword)
        cat = a.get("code_summary", "unknown").split(";")[0].strip()
        if not cat or cat == "no change from current":
            cat = "misc"
        groups[cat].append(a)

    lines = []
    for cat, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        best_tr = max(i["tr"] for i in items)
        descs = "; ".join(dict.fromkeys(i["desc"][:60] for i in items[:3]))
        lines.append(f"- **{cat}** ({len(items)}x, best {best_tr:.4f}): {descs}")
    return "\n".join(lines)


def build_history_prompt(approaches: list[dict], best_tr: float) -> str:
    """Build history section. Compresses discards when list gets long."""
    if not approaches:
        return ""

    keeps = [a for a in approaches if a["status"] == "keep"]
    discards = [a for a in approaches if a["status"] == "discard"]
    crashes = [a for a in approaches if a["status"] in ("crash", "timeout")]

    parts = [f"## Experiment history ({len(approaches)} total, best: {best_tr:.6f})\n"]

    # Keeps — always list individually (there are few of these)
    if keeps:
        parts.append(f"### Improvements ({len(keeps)}):")
        for a in keeps:
            parts.append(f"- #{a['exp']}: **{a['tr']:.6f}** — {a['desc']} [{a['code_summary']}]")
        parts.append("")

    # Discards — compress if >15, list individually if <=15
    if discards:
        if len(discards) <= 15:
            parts.append(f"### Failed — DO NOT REPEAT ({len(discards)}):")
            for a in discards:
                parts.append(f"- #{a['exp']}: {a['tr']:.4f} — {a['desc']} [{a['code_summary']}]")
        else:
            parts.append(f"### Failed — DO NOT REPEAT ({len(discards)}, grouped by category):")
            parts.append(_compress_discards(discards))
        parts.append("")

    # Crashes — last 3 only
    if crashes:
        parts.append(f"### Crashes ({len(crashes)} total, last 3):")
        for a in crashes[-3:]:
            parts.append(f"- #{a['exp']}: {a['desc']} — {a.get('error', '')[:80]}")
        parts.append("")

    # Auto-lessons after 5+ experiments
    if len(approaches) >= 5:
        parts.append("### Lessons:")
        if keeps:
            bk = max(keeps, key=lambda a: a["tr"])
            parts.append(f"- Best so far: #{bk['exp']} — {bk['desc']}")
        parts.append(f"- {len(discards)} failed, {len(crashes)} crashed")
        parts.append(f"- You MUST try something DIFFERENT from all failed approaches")

    return "\n".join(parts)


def build_messages(
    program_md: str,
    current_code: str,
    approaches: list[dict],
    best_tr: float,
    exp_num: int,
) -> list[dict]:
    """Build LLM messages. Trims program.md if prompt gets too long."""

    history = build_history_prompt(approaches, best_tr)

    system = dedent(f"""\
        You are an autonomous research agent. Modify experiment.py to maximize
        tightness_ratio. Output the COMPLETE file in a single ```python block.

        Rules:
        - FULL experiment.py in one ```python block — must be runnable standalone
        - Must import prepare, call prepare.evaluate_audit() and prepare.print_results()
        - Current best: {best_tr:.6f}. Beat it.
        - Only stdlib + torch/numpy/opacus/dp_accounting allowed
        - Must finish in <5 min (shadow model training: <10 min)
        - Start with "Hypothesis: <one line>" then the code block
        - DO NOT repeat any failed approach from the history below
    """)

    # If prompt is getting long, trim program.md to just the strategy sections
    context = program_md
    estimated_size = len(system) + len(history) + len(current_code) + len(context)
    if estimated_size > MAX_PROMPT_CHARS:
        # Keep only the actionable sections
        trimmed = []
        keep = False
        for line in program_md.splitlines():
            if line.startswith("## What you can change") or line.startswith("## Experiment progression"):
                keep = True
            elif line.startswith("## Setup") or line.startswith("## The three files") or line.startswith("## Output format"):
                keep = False
            if keep:
                trimmed.append(line)
        context = "\n".join(trimmed) if trimmed else program_md[:8000]

    user = f"""{history}

## Research context
{context}

## Current experiment.py (best so far)
```python
{current_code}
```

Propose experiment #{exp_num}. ONE hypothesis, COMPLETE code."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------
def extract_code(response: str) -> str | None:
    blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        c = max(blocks, key=len).strip()
        if "import" in c and "def " in c:
            return c
    return None


def extract_description(response: str) -> str:
    for pat in (r"[Hh]ypothesis:\s*(.+)", r"\*\*[Hh]ypothesis\*\*:\s*(.+)",
                r"[Cc]hange:\s*(.+)"):
        m = re.search(pat, response)
        if m:
            return m.group(1).strip()[:120]
    for line in response.splitlines():
        s = line.strip()
        if s and not s.startswith(("```", "import", "#")):
            return s[:120]
    return "no description"


# ---------------------------------------------------------------------------
# The loop — callable from CLI or notebook
# ---------------------------------------------------------------------------
def run_loop(
    api_key: str,
    model: str = "qwen/qwen3-coder",
    max_experiments: int = 0,
    timeout: int = 300,
    dry_run: bool = False,
):
    """Run the autonomous experiment loop.

    Parameters
    ----------
    api_key : str
        OpenRouter API key.
    model : str
        Primary model ID. Falls back through FREE_MODELS on failure.
    max_experiments : int
        Stop after this many. 0 = unlimited.
    timeout : int
        Seconds per experiment run.
    dry_run : bool
        Print prompt and exit without running.
    """
    print(f"=== DP Audit Autoresearch Loop ===")
    print(f"Model:       {model}")
    print(f"Timeout:     {timeout}s")
    print(f"Max exps:    {max_experiments or 'unlimited'}")
    print(f"State files: {AUTORESEARCH_DIR}\n")

    program_md = PROGRAM_MD.read_text(encoding="utf-8")
    init_results_log()

    # ---- Load state (supports resume) ----
    approaches = load_approaches()
    best_tr, exp_num = get_best_and_count(approaches)

    # ---- Baseline on fresh start ----
    if exp_num == 0:
        print("[exp 0] Running baseline ...")
        bl = run_experiment(timeout)
        if bl["status"] == "ok":
            best_tr = bl["tightness_ratio"]
            wall = bl.get("wall_seconds", 0)
            desc = "baseline: logit_margin budget=128"
            append_result(0, best_tr, wall, "keep", desc)
            log_approach(0, desc, "keep", best_tr, "original logit_margin scoring")
            print(f"  Baseline: {best_tr:.6f}")
            exp_num = 1
        else:
            print(f"  Baseline FAILED: {bl.get('error', '?')[:200]}")
            raise RuntimeError("Baseline experiment failed. Fix experiment.py first.")

    print(f"  Resuming at exp #{exp_num}, best={best_tr:.6f}, "
          f"{len(approaches)} approaches in memory.\n")

    # ---- Main loop ----
    experiments_run = 0
    while max_experiments == 0 or experiments_run < max_experiments:
        current_code = EXPERIMENT_PY.read_text(encoding="utf-8")

        print(f"\n{'='*55}")
        print(f"[exp {exp_num}] best={best_tr:.6f} | history={len(approaches)} | calling LLM ...")

        messages = build_messages(program_md, current_code, approaches, best_tr, exp_num)
        prompt_chars = sum(len(m["content"]) for m in messages)
        print(f"  Prompt: {prompt_chars:,} chars")

        if dry_run:
            print("\n--- SYSTEM ---")
            print(messages[0]["content"][:600])
            print("\n--- USER (truncated) ---")
            print(messages[1]["content"][:1500])
            break

        # Call LLM with fallback chain
        response = None
        models_to_try = [model] + [m for m in FREE_MODELS if m != model]
        for m in models_to_try:
            response = call_llm(messages, m, api_key)
            if response:
                if m != model:
                    print(f"  (fell back to {m})")
                break

        if not response:
            print("  All models failed. Sleeping 5 min ...")
            time.sleep(300)
            continue

        # Extract code + description
        new_code = extract_code(response)
        desc = extract_description(response)

        if not new_code:
            print(f"  No code block in response. Skipping.")
            append_result(exp_num, 0, 0, "crash", f"no code: {desc}")
            log_approach(exp_num, desc, "crash", 0, "no code extracted",
                         error="LLM returned no python block")
            exp_num += 1
            experiments_run += 1
            continue

        if new_code.strip() == current_code.strip():
            print(f"  Identical code returned. Skipping.")
            append_result(exp_num, 0, 0, "discard", "identical code")
            log_approach(exp_num, "identical code", "discard", 0, "no change")
            exp_num += 1
            experiments_run += 1
            continue

        # Syntax pre-check (instant, avoids wasting a subprocess)
        syn_err = syntax_check(new_code)
        if syn_err:
            print(f"  Syntax error: {syn_err}. Skipping.")
            code_summary = summarize_code_change(current_code, new_code)
            append_result(exp_num, 0, 0, "crash", f"syntax: {desc}")
            log_approach(exp_num, desc, "crash", 0, code_summary,
                         error=f"SyntaxError: {syn_err}")
            exp_num += 1
            experiments_run += 1
            continue

        code_summary = summarize_code_change(current_code, new_code)
        print(f"  Hypothesis: {desc}")
        print(f"  Changes:    {code_summary}")

        # Write, run, evaluate
        EXPERIMENT_PY.write_text(new_code, encoding="utf-8")
        t0 = time.time()
        result = run_experiment(timeout)
        elapsed = time.time() - t0

        if result["status"] != "ok":
            err = result.get("error", "")
            print(f"  FAILED ({result['status']}): {err[:150]}")
            EXPERIMENT_PY.write_text(current_code, encoding="utf-8")
            append_result(exp_num, 0, elapsed, result["status"], desc)
            log_approach(exp_num, desc, result["status"], 0, code_summary, error=err)
            exp_num += 1
            experiments_run += 1
            continue

        new_tr = result["tightness_ratio"]
        wall = result.get("wall_seconds", elapsed)
        member_fav = result.get("member_favoring", False)

        if not member_fav:
            print(f"  member_favoring=False. Reverting.")
            EXPERIMENT_PY.write_text(current_code, encoding="utf-8")
            append_result(exp_num, new_tr, wall, "discard", f"[!mf] {desc}")
            log_approach(exp_num, f"[!mf] {desc}", "discard", new_tr, code_summary)
        elif new_tr > best_tr:
            imp = new_tr - best_tr
            print(f"  IMPROVED: {best_tr:.6f} -> {new_tr:.6f} (+{imp:.6f})")
            best_tr = new_tr
            append_result(exp_num, new_tr, wall, "keep", desc)
            log_approach(exp_num, desc, "keep", new_tr, code_summary)
        else:
            print(f"  No improvement: {new_tr:.6f} <= {best_tr:.6f}. Reverting.")
            EXPERIMENT_PY.write_text(current_code, encoding="utf-8")
            append_result(exp_num, new_tr, wall, "discard", desc)
            log_approach(exp_num, desc, "discard", new_tr, code_summary)

        # Refresh approaches from disk (source of truth for resume)
        approaches = load_approaches()
        exp_num += 1
        experiments_run += 1
        time.sleep(2)

    print(f"\n=== Done. Best: {best_tr:.6f} | Experiments: {experiments_run} ===")
    print(f"Results:    {RESULTS_TSV}")
    print(f"Approaches: {APPROACHES_LOG}")
    print(f"Best code:  {EXPERIMENT_PY}")
    return best_tr


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Autonomous DP audit experiment loop")
    p.add_argument("--model", default="qwen/qwen3-coder")
    p.add_argument("--max-experiments", type=int, default=0)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Set OPENROUTER_API_KEY env var. Get a free key: https://openrouter.ai/keys")
        sys.exit(1)

    run_loop(api_key=api_key, model=args.model,
             max_experiments=args.max_experiments,
             timeout=args.timeout, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. State saved in results.tsv + approaches_log.jsonl.")
