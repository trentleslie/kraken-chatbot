"""Frozen pre-registration config for the code-on-graph spike (Phase 0).

This is the pre-registration artifact (P1-P5): all thresholds and the random seed
are fixed BEFORE any results are seen. The git commit timestamp on this file is the
freeze proof (Phase 0 uses a plain committed file; the hash-enforcing manifest is a
Phase-1 addition). Do not tune any value here in response to observed results.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SpikeConfig(BaseModel):
    model_config = {"frozen": True}

    # --- recall gate (P1) ---
    recall_lift_abs: float = Field(default=0.15, description="Absolute recall-lift threshold to proceed")
    recall_lift_recover_frac: float = Field(default=0.50, description="Relative form: recover >=this frac of static's misses")
    r0_relative_switch: float = Field(default=0.85, description="If baseline recall R0 > this, the absolute form is unattainable -> use relative form")

    # --- significance (P2) ---
    alpha: float = Field(default=0.05, description="McNemar two-sided alpha")
    mcnemar_exact_primary: bool = Field(default=True, description="Exact binomial McNemar is primary; asymptotic uncorrected is sensitivity")
    pi_d_prior: float = Field(default=0.25, description="Conservative discordance prior for powered-N before the iterate arm exists")
    n_floor: int = Field(default=30, description="Hard floor; below the powered-N -> INCONCLUSIVE, never a kill")
    n_target: int = Field(default=50, description="Target N (20 anchors + 30 random)")

    # --- cost (P3) ---
    turn_cap: int = Field(default=5, description="Hard self-correction turn cap for the iterate loop")
    per_turn_kestrel_call_cap: int = Field(default=8, description="Frozen from principle (not pilot): max Kestrel calls per loop turn")
    cost_ceiling_mult: float = Field(default=3.0, description="Worst-case loop cost must be <= this x baseline")

    # --- evidence budget (finding #2) ---
    multi_hop_limit: int = Field(default=100, description="FROZEN: /multi-hop limit, applied identically to BOTH arms")
    baseline_max_path_length: int = Field(default=5, description="Static baseline = ONE-SHOT query at full depth (== iterate reach), so the comparison isolates iteration, not search depth; not a shallow 2-hop")
    max_path_length: int = Field(default=5, description="Executor cap; also the reachability-filter hop bound")
    aggregate_path_budget: int = Field(default=100, description="Loop's cumulative distinct-path budget == baseline's")

    # --- variance band (no temperature control) ---
    k_reruns: int = Field(default=3, ge=1, description="Per-item reruns; hit = majority of K")

    # --- precision (Phase 1, P5) ---
    kappa_floor: float = Field(default=0.6, description="Inter-rater kappa floor; below -> precision INCONCLUSIVE")

    # --- gold set reproducibility (P5) ---
    drugmechdb_sample_seed: int = Field(default=20260603, description="Frozen seed for the random-30 DrugMechDB sample")
    drugmechdb_commit_sha: str = Field(default="UNPINNED", description="Pin DrugMechDB indication_paths.yaml commit SHA in Unit 0.2")


CONFIG = SpikeConfig()
