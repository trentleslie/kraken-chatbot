"""Blinded link-evidence judge, audit packet, and judge certification (Unit 3).

The judge sees ONLY the claim, the paper title and the normalized abstract -- never the source
that returned the paper (R2). A missing or unparseable verdict is ``not_run`` and blocks verdicts
downstream (never silently counted as not-hit). Certification is against Trent's verdict-blind
audit (R3), not LLM-to-LLM agreement.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

JUDGE_MODEL = "claude-sonnet-5"
JUDGE_CONCURRENCY = 4

JUDGE_PROMPT = """You are judging whether a scientific paper reports evidence on ONE specific claimed relationship.

Claim: {claim}

Paper title: {title}
Paper abstract: {abstract}

A paper is a HIT only if it reports evidence on the specific relationship the claim makes, whether
that evidence supports it or refutes it. A paper that discusses both entities, or the general topic,
without reporting evidence on this link is NOT a hit. Judge from the title and abstract only.

Return ONLY a JSON object, no prose:
{{"hit": true|false, "direction": "supports"|"refutes"|"na", "reason": "<one line>"}}
Use direction "na" when hit is false."""

DIRECTIONS = {"supports", "refutes", "na"}

ModelFn = Callable[[str], Awaitable[str]]


def prompt_hash() -> str:
    return hashlib.sha256((JUDGE_MODEL + "\n" + JUDGE_PROMPT).encode()).hexdigest()[:16]


def build_judge_prompt(claim: str, title: str, abstract: str) -> str:
    """Source-blind prompt: only claim, title and abstract are interpolated."""
    return JUDGE_PROMPT.format(claim=claim.strip(), title=title.strip(), abstract=abstract.strip())


def parse_verdict(text: str) -> dict[str, Any] | None:
    """Parse the judge's JSON verdict; anything malformed or inconsistent returns None."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    hit, direction = data.get("hit"), data.get("direction")
    if not isinstance(hit, bool) or direction not in DIRECTIONS:
        return None
    if hit and direction == "na":
        return None
    if not hit and direction != "na":
        return None
    return {"hit": hit, "direction": direction, "reason": str(data.get("reason", ""))[:300]}


async def default_call_model(prompt: str) -> str:  # pragma: no cover - live, Unit 4 only
    """One judge call via the Agent SDK on the local Claude login, pinned to JUDGE_MODEL."""
    from ...graph.sdk_utils import HAS_SDK, ClaudeAgentOptions, _apply_byok_env, query

    if not HAS_SDK:
        raise RuntimeError("claude_agent_sdk not installed; judge cannot run")
    options = ClaudeAgentOptions(
        system_prompt="You are a careful biomedical evidence judge. Return only the JSON object.",
        allowed_tools=[],
        max_turns=1,
        permission_mode="bypassPermissions",
        model=JUDGE_MODEL,
    )
    options = _apply_byok_env(options)
    parts: list[str] = []
    async for event in query(prompt=prompt, options=options):
        for block in getattr(event, "content", None) or []:
            if hasattr(block, "text"):
                parts.append(block.text)
    return "".join(parts)


async def judge_items(
    items: list[dict[str, Any]],
    call_model: ModelFn,
    *,
    concurrency: int = JUDGE_CONCURRENCY,
    attempts: int = 2,
    retry_delay: float = 0.5,
    checkpoint: Any = None,
) -> list[dict[str, Any]]:
    """Judge each item (``item_id, hyp_id, claim, title, abstract``) under a fixed semaphore.

    Items without an abstract are ``no_abstract`` (excluded from precision, counted). A call that
    raises or returns an unparseable verdict after ``attempts`` is ``not_run``. With a checkpoint,
    already-judged item_ids are skipped and new rows are appended as they finish.
    """
    sem = asyncio.Semaphore(concurrency)
    done = checkpoint.done_keys() if checkpoint is not None else set()

    async def one(item: dict[str, Any]) -> dict[str, Any]:
        base = {k: item[k] for k in ("item_id", "hyp_id") if k in item}
        if not item.get("abstract"):
            return {**base, "status": "no_abstract"}
        prompt = build_judge_prompt(item["claim"], item.get("title") or "", item["abstract"])
        last_error = None
        for attempt in range(attempts):
            try:
                async with sem:
                    raw = await call_model(prompt)
            except Exception as e:  # noqa: BLE001 - every failure becomes an explicit not_run
                last_error = f"{type(e).__name__}: {e}"
                if attempt + 1 < attempts:
                    await asyncio.sleep(retry_delay)
                continue
            verdict = parse_verdict(raw)
            if verdict is not None:
                return {**base, "status": "judged", **verdict, "raw": raw[:2000]}
            last_error = f"unparseable verdict: {raw[:200]!r}"
        return {**base, "status": "not_run", "error": last_error}

    todo = [it for it in items if (it["item_id"],) not in done]
    results = await asyncio.gather(*(one(it) for it in todo))
    if checkpoint is not None:
        for r in results:
            checkpoint.append(r)
    return results


# ---------------------------------------------------------------------------------------------
# Audit packet
# ---------------------------------------------------------------------------------------------

_PACKET_FIELDS = ("item_id", "claim", "title", "abstract", "consensus_title", "consensus_year",
                  "consensus_first_author", "resolved_title", "resolved_year", "resolved_first_author")


def draw_audit_sample(
    judgments: list[dict[str, Any]],
    *,
    false_support_events: list[dict[str, Any]],
    consensus_checks: list[dict[str, Any]],
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    """Seeded, stratified (source x judge verdict) audit sample + false-support + identity checks.

    The rule (seed, sizes, caps) comes from the frozen decision_rule.json, so the sample is fixed
    before any verdict is seen. Hard cap: ``rule['max_readings']``.
    """
    rng = random.Random(rule["seed"])
    judged = [j for j in judgments if j.get("status") == "judged"]
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for j in sorted(judged, key=lambda r: r["item_id"]):
        key = ((j.get("sources") or ["?"])[0], bool(j["hit"]))
        strata.setdefault(key, []).append(j)
    order = sorted(strata)
    for key in order:
        rng.shuffle(strata[key])

    sample: list[dict[str, Any]] = []
    cursor = {k: 0 for k in order}
    while len(sample) < rule["sample_size"] and any(cursor[k] < len(strata[k]) for k in order):
        for k in order:
            if len(sample) >= rule["sample_size"]:
                break
            if cursor[k] < len(strata[k]):
                j = strata[k][cursor[k]]
                cursor[k] += 1
                sample.append({**j, "judge_hit": bool(j["hit"]), "audit_role": "sample"})

    seen: set[tuple[str, str]] = set()
    fs_unique = []
    for ev in sorted(false_support_events, key=lambda r: r["item_id"]):
        key = (ev["hyp_id"], ev["paper_key"])
        if key not in seen:
            seen.add(key)
            fs_unique.append(ev)
    rng.shuffle(fs_unique)
    fs = [{**ev, "judge_hit": True, "audit_role": "false_support"}
          for ev in fs_unique[: rule["false_support_cap"]]]

    cc = sorted(consensus_checks, key=lambda r: r["item_id"])
    rng.shuffle(cc)
    cc = [{**c, "audit_role": "consensus_identity"} for c in cc[: rule["consensus_identity_checks"]]]

    packet = sample + fs + cc
    if len(packet) > rule["max_readings"]:
        raise ValueError(f"audit packet {len(packet)} exceeds hard cap {rule['max_readings']}")
    return packet


def export_audit_packet(sample: list[dict[str, Any]], path: Path, seed: int = 0) -> Path:
    """Write the verdict-blind, source-blind packet (shuffled so role order leaks nothing)."""
    rows = [{k: s[k] for k in _PACKET_FIELDS if k in s} for s in sample]
    random.Random(seed).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def ingest_audit_labels(path: Path) -> dict[str, bool]:
    """Read Trent's labels: one JSON object per line with ``item_id`` and boolean ``hit``."""
    labels: dict[str, bool] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("hit"), bool):
            raise ValueError(f"audit label for {row.get('item_id')} is not a boolean hit")
        labels[row["item_id"]] = row["hit"]
    return labels


def agreement_stats(pairs: list[tuple[bool, bool, str]]) -> dict[str, Any]:
    """(judge_hit, human_hit, source) pairs -> pooled agreement, Cohen's kappa, PPV, NPV, per source."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None, "ppv": None, "npv": None, "per_source": {}}
    agree = sum(1 for j, h, _ in pairs if j == h)
    po = agree / n
    pj = sum(1 for j, _, _ in pairs if j) / n
    ph = sum(1 for _, h, _ in pairs if h) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else 0.0)
    jpos = [(j, h) for j, h, _ in pairs if j]
    jneg = [(j, h) for j, h, _ in pairs if not j]
    ppv = sum(1 for _, h in jpos if h) / len(jpos) if jpos else None
    npv = sum(1 for _, h in jneg if not h) / len(jneg) if jneg else None
    per: dict[str, dict[str, Any]] = {}
    for src in sorted({s for _, _, s in pairs}):
        sp = [(j, h) for j, h, s in pairs if s == src]
        per[src] = {"n": len(sp), "agreement": sum(1 for j, h in sp if j == h) / len(sp)}
    return {"n": n, "agreement": po, "kappa": kappa, "ppv": ppv, "npv": npv, "per_source": per}


def certify(stats: dict[str, Any], rule: dict[str, Any]) -> tuple[bool, list[str]]:
    """Pooled agreement >= min AND kappa >= min AND PPV >= min (frozen thresholds)."""
    reasons = []
    for key, threshold in (("agreement", rule["min_agreement"]), ("kappa", rule["min_kappa"]),
                           ("ppv", rule["min_ppv"])):
        val = stats.get(key)
        if val is None or val < threshold:
            reasons.append(f"{key}={val} below {threshold}")
    return (not reasons), reasons


def corrected_precision(p_judge: float, *, ppv: float, npv: float) -> float:
    """Judge-positive rate corrected to the human-label scale: P(hit) = PPV*p + (1-NPV)*(1-p)."""
    return ppv * p_judge + (1 - npv) * (1 - p_judge)
