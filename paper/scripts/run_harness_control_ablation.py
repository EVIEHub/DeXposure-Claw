#!/usr/bin/env python3
"""Run paired DeXposure-Claw harness-control ablations.

The historical M6 artifact supplies 29 immutable weekly prompts, three cached
Claude Opus 4.7 outputs per prompt, and future stress labels.  This script
uses that artifact as the full-control condition and creates two one-control
input ablations:

* without_scenario_evidence removes only the stress-scenario block.
* without_output_schema removes only the JSON-only instruction and schema.

New LLM calls reuse the historical decision model, temperature, and number of
runs.  Every request records source, rendered-prompt, and raw-response hashes.
The future label is never included in a model request.

The script also replays two post-output controls on the exact cached full
outputs.  self_score_action_gate is the gate implemented by llm_eval_b5.py.
numeric_trace_gate is an explicitly weak proxy that demotes interventions
whose protocol or cited number cannot be traced to the rendered prompt.  It
is not a formal provenance validator because the historical ticket schema
contains no evidence IDs.

Data-health and monitor-confidence gates are intentionally not synthesized.
Their required per-week fields are absent from the frozen M6 artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_ROOT = ROOT / "paper"
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))
from experiments.b5_decision_quality import (  # noqa: E402
    CandidateSet,
    EvidenceBundle,
    evidence_hash as canonical_evidence_hash,
    require_exact_k_targets,
    ranked_unique_targets as canonical_ranked_unique_targets,
)
from experiments.llm_eval_b5 import (  # noqa: E402
    apply_action_gate as _canonical_action_gate,
    response_usage,
)
DEFAULT_OUTPUT_DIR = ROOT / "paper/results/harness_control_ablation"
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_BUDGETS = (1, 3, 5, 7)
INTERVENTION_ACTIONS = {"Recommend-Reduce", "Contingency"}
ALLOWED_RISK_LEVELS = {"low", "moderate", "elevated", "critical"}
ALLOWED_ACTIONS = {"Monitor", "Investigate", *INTERVENTION_ACTIONS}
MAX_OUTPUT_TOKENS = 4096
_SCENARIO_HEADER = "== STRESS SCENARIO ANALYSIS (applied to predicted graph) =="
_FINAL_QUESTION = "Based on these FM model predictions, metrics, and stress analysis,"
_SCHEMA_START = "You MUST respond with valid JSON only"
_RULES_START = "Rules:"
_CATEGORY_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
_CANDIDATE_SCENARIO_FIELDS_RE = re.compile(
    r";\s*scenario_max_loss:\s*[-+]?\d+(?:\.\d+)?"
    r";\s*scenario_sum_loss:\s*[-+]?\d+(?:\.\d+)?"
)
_WRITE_LOCK = threading.Lock()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_protocol(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _CATEGORY_SUFFIX_RE.sub("", value.strip()).strip()


def evidence_hash(
    system_prompt: str,
    user_prompt: str,
    candidate_set: CandidateSet | None = None,
) -> str:
    return canonical_evidence_hash(system_prompt, user_prompt, candidate_set)


def request_config(
    *, api_url: str, model: str, temperature: float, timeout: int
) -> dict[str, Any]:
    return {
        "api_url": api_url,
        "provider_model": api_model_slug(model),
        "temperature": temperature,
        "timeout_seconds": timeout,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def request_identity(
    variant: dict[str, Any], config: dict[str, Any]
) -> str:
    payload = {
        "request_config": config,
        "system_prompt_sha256": variant["system_prompt_sha256"],
        "user_prompt_sha256": variant["user_prompt_sha256"],
    }
    return sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


def remove_scenario_evidence(user_prompt: str) -> str:
    if _SCENARIO_HEADER not in user_prompt or _FINAL_QUESTION not in user_prompt:
        raise ValueError("prompt does not contain the expected scenario block")
    prefix, remainder = user_prompt.split(_SCENARIO_HEADER, 1)
    _, question_tail = remainder.split(_FINAL_QUESTION, 1)
    question = (
        "Based on these FM model predictions and metrics,"
        + question_tail
    )
    prefix = _CANDIDATE_SCENARIO_FIELDS_RE.sub("", prefix)
    return prefix.rstrip() + "\n\n" + question


def remove_output_schema(system_prompt: str) -> str:
    if _SCHEMA_START not in system_prompt or _RULES_START not in system_prompt:
        raise ValueError("system prompt does not contain the expected output schema")
    prefix, remainder = system_prompt.split(_SCHEMA_START, 1)
    _, rules = remainder.split(_RULES_START, 1)
    return prefix.rstrip() + "\n\n" + _RULES_START + rules


def render_condition(
    system_prompt: str, user_prompt: str, condition: str
) -> tuple[str, str]:
    if condition in {"full_harness", "full_harness_current"}:
        return system_prompt, user_prompt
    if condition == "without_scenario_evidence":
        return system_prompt, remove_scenario_evidence(user_prompt)
    if condition == "without_output_schema":
        return remove_output_schema(system_prompt), user_prompt
    raise ValueError(f"unknown condition: {condition}")


def parse_json_response(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def load_source(raw_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(raw_path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("source artifact must be a non-empty list")
    dates = [str(row.get("date", "")) for row in raw]
    if len(set(dates)) != len(dates):
        raise ValueError("source artifact contains duplicate dates")
    for row in raw:
        outputs = row.get("raw_llm_outputs", row.get("llm_outputs", []))
        if len(outputs) < 1:
            raise ValueError(f"{row.get('date')}: missing cached output")
    return raw


def build_manifest(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for row in raw:
        source_system = str(row["system_prompt"])
        source_user = str(row["user_prompt"])
        candidate_set = None
        evidence_payload = row.get("evidence_bundle")
        if isinstance(evidence_payload, dict):
            candidate_rows = evidence_payload.get("candidate_set", [])
            if not isinstance(candidate_rows, list):
                raise ValueError(f"{row.get('date')}: candidate_set must be a list")
            candidate_set = CandidateSet.from_rows(candidate_rows)
            if evidence_payload.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
                raise ValueError(f"{row.get('date')}: candidate_set hash mismatch")
            recorded_membership = evidence_payload.get("candidate_membership_sha256")
            if (
                recorded_membership is not None
                and recorded_membership != candidate_set.candidate_membership_sha256
            ):
                raise ValueError(f"{row.get('date')}: candidate membership hash mismatch")
            recorded = str(evidence_payload.get("evidence_sha256", ""))
            if recorded and recorded != evidence_hash(source_system, source_user, candidate_set):
                raise ValueError(f"{row.get('date')}: EvidenceBundle hash mismatch")
        source_bundle = EvidenceBundle(
            date=str(row["date"]),
            system_prompt=source_system,
            user_prompt=source_user,
            candidate_set=candidate_set,
        )
        source_hash = source_bundle.evidence_sha256
        variants: dict[str, dict[str, Any]] = {}
        for condition in (
            "full_harness",
            "full_harness_current",
            "without_scenario_evidence",
            "without_output_schema",
        ):
            system_prompt, user_prompt = render_condition(
                source_system, source_user, condition
            )
            variants[condition] = {
                "rendered_evidence_sha256": evidence_hash(
                    system_prompt, user_prompt, candidate_set
                ),
                "system_prompt_sha256": sha256_text(system_prompt),
                "user_prompt_sha256": sha256_text(user_prompt),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "candidate_set_sha256": (
                    candidate_set.candidate_set_sha256 if candidate_set else None
                ),
                "candidate_membership_sha256": (
                    candidate_set.candidate_membership_sha256 if candidate_set else None
                ),
            }
        manifest.append(
            {
                "date": str(row["date"]),
                "source_evidence_sha256": source_hash,
                "candidate_set_sha256": (
                    candidate_set.candidate_set_sha256 if candidate_set else None
                ),
                "candidate_membership_sha256": (
                    candidate_set.candidate_membership_sha256 if candidate_set else None
                ),
                "variants": variants,
            }
        )
    return manifest


def api_model_slug(model: str) -> str:
    normalized = re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)
    return normalized if "/" in normalized else f"anthropic/{normalized}"


def call_llm(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout: int,
    seed: int | None = None,
    provider: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": api_model_slug(model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    if seed is not None:
        payload["seed"] = seed
    if provider is not None:
        payload["provider"] = provider
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dexposure-agent",
        },
    )
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    last_error = ""
    attempt_records: list[dict[str, Any]] = []
    for attempt in range(5):
        attempt_started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read())
            content = result["choices"][0]["message"]["content"]
            input_tokens, output_tokens, usage_known = response_usage(result)
            attempt_records.append(
                {
                    "attempt_index": attempt,
                    "status": "ok",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "usage_known": usage_known,
                    "latency_ms": round((time.monotonic() - attempt_started) * 1000, 1),
                }
            )
            return {
                "status": "ok",
                "raw_response": content,
                "raw_response_sha256": sha256_text(content),
                "parsed": parse_json_response(content),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "usage_known": usage_known,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "provider_model": payload["model"],
                "response_model": result.get("model"),
                "response_provider": result.get("provider"),
                "response_id": result.get("id"),
                "usage": result.get("usage"),
                "request_started_at": started_at,
                "attempts": attempt_records,
            }
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {error.code}: {body}"
            attempt_records.append(
                {
                    "attempt_index": attempt,
                    "status": "error",
                    "error": last_error,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usage_known": False,
                    "latency_ms": round((time.monotonic() - attempt_started) * 1000, 1),
                }
            )
            retryable = error.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == 4:
                break
        except Exception as error:  # network failures need checkpointed evidence
            last_error = f"{type(error).__name__}: {error}"
            attempt_records.append(
                {
                    "attempt_index": attempt,
                    "status": "error",
                    "error": last_error,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usage_known": False,
                    "latency_ms": round((time.monotonic() - attempt_started) * 1000, 1),
                }
            )
            if attempt == 4:
                break
        time.sleep(5 * (attempt + 1))
    return {
        "status": "error",
        "error": last_error,
        "raw_response": "",
        "raw_response_sha256": sha256_text(""),
        "parsed": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "usage_known": False,
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "provider_model": payload["model"],
        "request_started_at": started_at,
        "attempts": attempt_records,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def run_input_calls(
    *,
    manifest: list[dict[str, Any]],
    output_dir: Path,
    conditions: tuple[str, ...],
    model: str,
    runs: int,
    workers: int,
    api_url: str,
    api_key: str,
    temperature: float,
    timeout: int,
    seed: int,
) -> None:
    tasks: list[tuple[str, int, dict[str, Any]]] = []
    config = request_config(
        api_url=api_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    for condition in conditions:
        path = output_dir / f"raw_{condition}.json"
        existing = json.loads(path.read_text()) if path.exists() else []
        completed = {
            (
                str(row["date"]),
                int(row["run_index"]),
                str(row.get("request_identity_sha256", "")),
            )
            for row in existing
            if row.get("status") == "ok"
        }
        by_date = {str(row["date"]): row for row in manifest}
        for date, row in by_date.items():
            for run_index in range(runs):
                identity = request_identity(row["variants"][condition], config)
                if (date, run_index, identity) not in completed:
                    tasks.append((condition, run_index, row))

    # Interleave conditions and dates so condition-level latency is not tied to
    # provider load at one contiguous point in the run.
    random.Random(seed).shuffle(tasks)

    logging.info("scheduled %d missing LLM calls", len(tasks))

    def execute(task: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
        condition, run_index, row = task
        variant = row["variants"][condition]
        identity = request_identity(variant, config)
        response = call_llm(
            api_url=api_url,
            api_key=api_key,
            model=model,
            system_prompt=variant["system_prompt"],
            user_prompt=variant["user_prompt"],
            temperature=temperature,
            timeout=timeout,
        )
        return {
            "condition": condition,
            "date": row["date"],
            "run_index": run_index,
            "source_evidence_sha256": row["source_evidence_sha256"],
            "rendered_evidence_sha256": variant["rendered_evidence_sha256"],
            "candidate_set_sha256": variant.get("candidate_set_sha256"),
            "candidate_membership_sha256": variant.get(
                "candidate_membership_sha256"
            ),
            "system_prompt_sha256": variant["system_prompt_sha256"],
            "user_prompt_sha256": variant["user_prompt_sha256"],
            "request_config": config,
            "request_identity_sha256": identity,
            **response,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            condition = str(result["condition"])
            path = output_dir / f"raw_{condition}.json"
            with _WRITE_LOCK:
                existing = json.loads(path.read_text()) if path.exists() else []
                key = (str(result["date"]), int(result["run_index"]))
                existing = [
                    row
                    for row in existing
                    if (str(row["date"]), int(row["run_index"])) != key
                ]
                existing.append(result)
                existing.sort(key=lambda row: (row["date"], row["run_index"]))
                atomic_write_json(path, existing)
            logging.info(
                "%s %s run=%d status=%s latency_ms=%s",
                condition,
                result["date"],
                result["run_index"],
                result["status"],
                result["latency_ms"],
            )


def target_rows(output: dict[str, Any] | None) -> list[dict[str, Any]]:
    normalized_output = deepcopy(output)
    if isinstance(normalized_output, dict) and isinstance(
        normalized_output.get("target_protocols"), list
    ):
        normalized_output["target_protocols"] = [
            {
                **row,
                "protocol": normalize_protocol(row.get("protocol", "")),
            }
            for row in normalized_output["target_protocols"]
            if isinstance(row, dict)
        ]
    result = canonical_ranked_unique_targets(normalized_output)
    for row in result:
        row["protocol"] = normalize_protocol(row.get("protocol"))
    return result


def candidate_set_for_source(source: dict[str, Any]) -> CandidateSet | None:
    """Read and verify the per-date typed CandidateSet from frozen evidence."""
    evidence = source.get("evidence_bundle")
    if not isinstance(evidence, dict):
        return None
    rows = evidence.get("candidate_set")
    if not isinstance(rows, list):
        raise ValueError("typed evidence_bundle candidate_set must be a list")
    candidate_set = CandidateSet.from_rows(rows)
    if evidence.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
        raise ValueError("typed evidence_bundle candidate_set hash mismatch")
    recorded_membership = evidence.get("candidate_membership_sha256")
    if (
        recorded_membership is not None
        and recorded_membership != candidate_set.candidate_membership_sha256
    ):
        raise ValueError("typed evidence_bundle candidate membership hash mismatch")
    return candidate_set


def satisfies_ticket_contract(
    output: dict[str, Any] | None,
    candidate_set: CandidateSet | None = None,
    expected_count: int = 7,
) -> bool:
    if not isinstance(output, dict):
        return False
    if output.get("risk_level") not in ALLOWED_RISK_LEVELS:
        return False
    if (
        not isinstance(output.get("rationale"), str)
        or not output["rationale"].strip()
    ):
        return False
    targets = output.get("target_protocols")
    if not isinstance(targets, list):
        return False
    if len(targets) != expected_count:
        return False
    seen: set[str] = set()
    previous_score = float("inf")
    for target in targets:
        if not isinstance(target, dict):
            return False
        protocol = normalize_protocol(target.get("protocol"))
        score = target.get("risk_score")
        action = target.get("action")
        reason = target.get("reason")
        if not protocol:
            return False
        if protocol in seen:
            return False
        if candidate_set is not None and protocol not in {
            candidate for candidate, _, _, _ in candidate_set.rows
        }:
            return False
        seen.add(protocol)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not np.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or float(score) > previous_score
        ):
            return False
        previous_score = float(score)
        if action not in ALLOWED_ACTIONS:
            return False
        if not isinstance(reason, str) or not reason.strip():
            return False
    return True


def trace_status(target: dict[str, Any], prompt: str) -> dict[str, bool]:
    protocol = normalize_protocol(target.get("protocol"))
    protocol_visible = bool(protocol) and bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(protocol)}(?![A-Za-z0-9_])",
            prompt,
        )
    )
    prompt_numbers = set(_NUMBER_RE.findall(prompt))
    reason_numbers = set(_NUMBER_RE.findall(str(target.get("reason", ""))))
    return {
        "protocol_visible": protocol_visible,
        "numeric_trace": bool(reason_numbers & prompt_numbers),
    }


def self_score_action_gate(output: dict[str, Any]) -> tuple[dict[str, Any], int]:
    raw_actions = [
        target.get("action")
        for target in output.get("target_protocols", [])
        if isinstance(target, dict)
    ]
    gated = _canonical_action_gate(output)
    gated["target_protocols"] = target_rows(gated)
    gated_actions = [target.get("action") for target in gated["target_protocols"]]
    demotions = sum(
        before in INTERVENTION_ACTIONS and after == "Investigate"
        for before, after in zip(raw_actions, gated_actions)
    )
    return gated, demotions


def numeric_trace_gate(
    output: dict[str, Any], prompt: str
) -> tuple[dict[str, Any], int]:
    gated = deepcopy(output)
    demotions = 0
    targets = target_rows(gated)
    for target in targets:
        status = trace_status(target, prompt)
        if (
            target.get("action") in INTERVENTION_ACTIONS
            and not all(status.values())
        ):
            target["action"] = "Investigate"
            demotions += 1
    gated["target_protocols"] = targets
    return gated, demotions


def evaluate_outputs(
    *,
    condition: str,
    outputs: list[dict[str, Any]],
    source_by_date: dict[str, dict[str, Any]],
    prompt_by_date: dict[str, str],
    budgets: tuple[int, ...],
    rates: dict[str, float] | None = None,
    post_gate: str | None = None,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    coverage_by_budget: dict[int, list[float]] = {budget: [] for budget in budgets}
    for budget in budgets:
        selected_count = hit_count = intervention_count = false_count = 0
        non_loss_false_count = 0
        stressed_count = 0
        intervention_weeks = demotions = parse_successes = contract_successes = 0
        n_valid_exact_k = 0
        active_week_firs: list[float] = []
        trace_protocol_count = trace_numeric_count = trace_target_count = 0
        latency_values: list[float] = []
        input_tokens = output_tokens = 0
        unknown_usage_attempts = 0
        for record in outputs:
            if record.get("status") != "ok":
                raise ValueError(
                    f"{condition}: failed request for "
                    f"{record.get('date')} run={record.get('run_index')}"
                )
            date = str(record["date"])
            source = source_by_date[date]
            candidate_set = candidate_set_for_source(source)
            stressed = {
                normalize_protocol(value)
                for value in source.get("truly_stressed", [])
                if normalize_protocol(value)
            }
            if not stressed:
                raise ValueError(f"{condition} {date}: empty stressed target set")
            parsed = record.get("parsed")
            if isinstance(parsed, dict):
                parse_successes += 1
                contract_valid = satisfies_ticket_contract(parsed, candidate_set)
                output = deepcopy(parsed) if contract_valid else {}
            else:
                contract_valid = False
                output = {}
            contract_successes += int(contract_valid)
            if contract_valid and post_gate == "self_score_action_gate":
                output, count = self_score_action_gate(output)
                demotions += count
            elif contract_valid and post_gate == "numeric_trace_gate":
                output, count = numeric_trace_gate(output, prompt_by_date[date])
                demotions += count

            # Invalid or contract-drifting responses fail closed.  They do not
            # become a smaller valid budget by slicing whatever rows happen to
            # be present.  Their end-to-end coverage is therefore zero, while
            # conditional financial metrics below use only valid exact-K runs.
            selected: list[dict[str, Any]] = []
            if contract_valid:
                try:
                    selected = require_exact_k_targets(target_rows(output), budget)
                except ValueError:
                    selected = []
                else:
                    n_valid_exact_k += 1
            hits = sum(target["protocol"] in stressed for target in selected)
            selected_count += len(selected)
            hit_count += hits
            stressed_count += len(stressed)
            coverage_by_budget[budget].append(hits / len(stressed))
            interventions = [
                target
                for target in selected
                if target.get("action") in INTERVENTION_ACTIONS
            ]
            false = [
                target
                for target in interventions
                if target["protocol"] not in stressed
            ]
            positive_loss = {
                normalize_protocol(value)
                for value in source.get(
                    "positive_loss_protocols",
                    [
                        name
                        for name, loss in source.get("actual_losses", {}).items()
                        if float(loss or 0.0) > 0.0
                    ],
                )
                if normalize_protocol(value)
            }
            non_loss_false = [
                target
                for target in interventions
                if target["protocol"] not in positive_loss
            ]
            intervention_count += len(interventions)
            false_count += len(false)
            non_loss_false_count += len(non_loss_false)
            intervention_weeks += int(bool(interventions))
            if interventions:
                active_week_firs.append(len(false) / len(interventions))
            for target in selected:
                status = trace_status(target, prompt_by_date[date])
                trace_target_count += 1
                trace_protocol_count += int(status["protocol_visible"])
                trace_numeric_count += int(status["numeric_trace"])
            attempts = record.get("attempts")
            if isinstance(attempts, list) and attempts:
                unknown_usage_attempts += sum(
                    attempt.get("usage_known") is not True
                    for attempt in attempts
                )
                input_tokens += sum(
                    int(attempt.get("input_tokens", 0) or 0)
                    for attempt in attempts
                )
                output_tokens += sum(
                    int(attempt.get("output_tokens", 0) or 0)
                    for attempt in attempts
                )
            else:
                input_tokens += int(record.get("input_tokens", 0) or 0)
                output_tokens += int(record.get("output_tokens", 0) or 0)
            if record.get("latency_ms") is not None:
                latency_values.append(float(record["latency_ms"]))

        n = len(outputs)
        cost = None
        if rates:
            cost = (
                input_tokens * rates["input"]
                + output_tokens * rates["output"]
            ) / 1_000_000
        known_cost = cost
        if unknown_usage_attempts:
            cost = None
        label = condition if post_gate is None else f"{condition}+{post_gate}"
        precision_micro = hit_count / selected_count if selected_count else None
        recall_micro = hit_count / stressed_count if stressed_count else None
        f1_micro = None
        if precision_micro is not None and recall_micro is not None:
            denominator = precision_micro + recall_micro
            f1_micro = (
                2 * precision_micro * recall_micro / denominator
                if denominator
                else 0.0
            )
        rows.append(
            {
                "condition": label,
                "budget": budget,
                "n_week_runs": n,
                "coverage_at_k": float(np.mean(coverage_by_budget[budget])),
                "precision_micro_at_k": precision_micro,
                "recall_micro_at_k": recall_micro,
                "f1_micro_at_k": f1_micro,
                "budget_utilization": selected_count / (n * budget),
                "intervention_targets_per_week_run": intervention_count / n,
                "false_interventions_per_week_run": false_count / n,
                "non_loss_false_interventions_per_week_run": non_loss_false_count / n,
                "fir_micro": false_count / intervention_count if intervention_count else None,
                "tail_fir_micro": false_count / intervention_count if intervention_count else None,
                "non_loss_fir_micro": (
                    non_loss_false_count / intervention_count
                    if intervention_count
                    else None
                ),
                "fir_active_week_macro": (
                    float(np.mean(active_week_firs)) if active_week_firs else None
                ),
                "intervention_week_rate": intervention_weeks / n,
                "parse_success_rate": parse_successes / n,
                "contract_success_rate": contract_successes / n,
                "ticket_contract_success_rate": contract_successes / n,
                "n_valid_exact_k": n_valid_exact_k,
                "valid_exact_k_rate": n_valid_exact_k / n,
                "protocol_trace_rate": (
                    trace_protocol_count / trace_target_count
                    if trace_target_count
                    else None
                ),
                "numeric_trace_rate": (
                    trace_numeric_count / trace_target_count
                    if trace_target_count
                    else None
                ),
                "gate_demotions": demotions,
                "mean_latency_ms": (
                    float(np.mean(latency_values)) if latency_values else None
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "unknown_usage_attempts": unknown_usage_attempts,
                "api_cost_usd": cost,
                "api_cost_usd_per_week_run": cost / n if cost is not None else None,
                "known_api_cost_usd": known_cost,
                # Financial/action metrics conditioned on a contract-valid,
                # exact-K response. End-to-end fields above retain invalid
                # outputs as zero-target runs.
                "conditional_intervention_targets_per_valid_output": (
                    intervention_count / n_valid_exact_k
                    if n_valid_exact_k
                    else None
                ),
                "conditional_false_interventions_per_valid_output": (
                    false_count / n_valid_exact_k if n_valid_exact_k else None
                ),
                "conditional_non_loss_false_interventions_per_valid_output": (
                    non_loss_false_count / n_valid_exact_k
                    if n_valid_exact_k
                    else None
                ),
                "conditional_fir_micro": (
                    false_count / intervention_count
                    if intervention_count
                    else None
                ),
                "conditional_non_loss_fir_micro": (
                    non_loss_false_count / intervention_count
                    if intervention_count
                    else None
                ),
            }
        )
    return rows, {
        budget: np.asarray(values, dtype=float)
        for budget, values in coverage_by_budget.items()
    }


def bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    delta = a - b
    n = len(delta)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    for index in range(n_bootstrap):
        starts = rng.integers(0, n, size=blocks)
        chosen = np.concatenate(
            [((start + offsets) % n) for start in starts]
        )[:n]
        samples[index] = float(np.mean(delta[chosen]))
    observed = float(np.mean(delta))
    # Block sign-flip randomization tests the paired null while preserving
    # within-block temporal dependence. Bootstrap samples above are used only
    # for the confidence interval.
    null_samples = np.empty(n_bootstrap, dtype=float)
    n_blocks = int(np.ceil(n / block_length))
    for index in range(n_bootstrap):
        offset = int(rng.integers(0, n))
        rotated = np.roll(delta, -offset)
        signs = rng.choice((-1.0, 1.0), size=n_blocks)
        signed = rotated.copy()
        for block in range(n_blocks):
            start = block * block_length
            stop = min(n, start + block_length)
            signed[start:stop] *= signs[block]
        null_samples[index] = float(np.mean(signed))
    p_value = (
        np.count_nonzero(np.abs(null_samples) >= abs(observed)) + 1
    ) / (n_bootstrap + 1)
    return {
        "mean_delta": observed,
        "ci_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "p_two_sided": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def historical_records(raw: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in raw:
        for run_index, output in enumerate(
            row.get("raw_llm_outputs", row.get("llm_outputs", []))
        ):
            records.append(
                {
                    "condition": "full_harness",
                    "date": str(row["date"]),
                    "run_index": run_index,
                    "status": "ok",
                    "parsed": deepcopy(output),
                    "input_tokens": int(output.get("input_tokens", 0) or 0),
                    "output_tokens": int(output.get("output_tokens", 0) or 0),
                    "latency_ms": output.get("latency_ms"),
                    "provider_model": api_model_slug(model),
                    "attempts": output.get(
                        "attempts",
                        [
                            {
                                "attempt_index": 0,
                                "status": "ok",
                                "input_tokens": int(output.get("input_tokens", 0) or 0),
                                "output_tokens": int(output.get("output_tokens", 0) or 0),
                                "latency_ms": output.get("latency_ms"),
                            }
                        ],
                    ),
                }
            )
    return records


def validate_new_records(
    *,
    condition: str,
    records: list[dict[str, Any]],
    manifest_by_date: dict[str, dict[str, Any]],
    runs: int,
) -> dict[str, Any]:
    expected = len(manifest_by_date) * runs
    keys = [(str(row["date"]), int(row["run_index"])) for row in records]
    if len(records) != expected:
        raise ValueError(
            f"{condition}: expected {expected} records, found {len(records)}"
        )
    if len(set(keys)) != len(keys):
        raise ValueError(f"{condition}: duplicate date/run records")
    expected_keys = {
        (date, run_index)
        for date in manifest_by_date
        for run_index in range(runs)
    }
    if set(keys) != expected_keys:
        raise ValueError(f"{condition}: date/run grid is incomplete")
    hash_matches = 0
    request_identities: set[str] = set()
    for row in records:
        date = str(row["date"])
        manifest_row = manifest_by_date[date]
        variant = manifest_row["variants"][condition]
        if (
            row.get("source_evidence_sha256")
            != manifest_row["source_evidence_sha256"]
            or row.get("rendered_evidence_sha256")
            != variant["rendered_evidence_sha256"]
            or row.get("candidate_set_sha256")
            != variant.get("candidate_set_sha256")
            or row.get("candidate_membership_sha256")
            != variant.get("candidate_membership_sha256")
            or row.get("system_prompt_sha256")
            != variant["system_prompt_sha256"]
            or row.get("user_prompt_sha256")
            != variant["user_prompt_sha256"]
        ):
            raise ValueError(f"{condition} {date}: prompt hash mismatch")
        config = row.get("request_config")
        if not isinstance(config, dict):
            raise ValueError(f"{condition} {date}: missing request config")
        expected_identity = request_identity(variant, config)
        if row.get("request_identity_sha256") != expected_identity:
            raise ValueError(f"{condition} {date}: request identity mismatch")
        request_identities.add(expected_identity)
        if row.get("status") != "ok":
            raise ValueError(
                f"{condition} {date} run={row.get('run_index')}: "
                f"request failed: {row.get('error', 'unknown error')}"
            )
        if not isinstance(row.get("attempts"), list) or not row["attempts"]:
            raise ValueError(f"{condition} {date}: missing per-attempt usage records")
        hash_matches += 1
    request_configs = {
        json.dumps(row["request_config"], sort_keys=True)
        for row in records
    }
    if len(request_configs) != 1:
        raise ValueError(f"{condition}: mixed request configurations")
    return {
        "expected_records": expected,
        "actual_records": len(records),
        "unique_date_run_keys": len(set(keys)),
        "successful_requests": sum(row.get("status") == "ok" for row in records),
        "prompt_hash_matches": hash_matches,
        "unique_request_identities": len(request_identities),
        "request_config": json.loads(next(iter(request_configs))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    *,
    raw: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    rates: dict[str, float] | None,
    budgets: tuple[int, ...],
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if tuple(sorted(budgets)) != DEFAULT_BUDGETS:
        raise ValueError(
            "harness ablation requires exactly K={1,3,5,7}; K=5 is primary"
        )
    source_by_date = {str(row["date"]): row for row in raw}
    manifest_by_date = {str(row["date"]): row for row in manifest}
    for date, manifest_row in manifest_by_date.items():
        source_candidate_hash = manifest_row.get("candidate_set_sha256")
        source_membership_hash = manifest_row.get("candidate_membership_sha256")
        for condition, variant in manifest_row["variants"].items():
            if variant.get("candidate_set_sha256") != source_candidate_hash:
                raise ValueError(
                    f"{condition} {date}: matched policy candidate-set hash mismatch"
                )
            if variant.get("candidate_membership_sha256") != source_membership_hash:
                raise ValueError(
                    f"{condition} {date}: matched policy candidate membership mismatch"
                )
    prompt_by_condition: dict[str, dict[str, str]] = {}
    records_by_condition: dict[str, list[dict[str, Any]]] = {
        "full_harness_historical": historical_records(raw, model)
    }
    integrity: dict[str, dict[str, Any]] = {}
    for condition in (
        "full_harness",
        "full_harness_current",
        "without_scenario_evidence",
        "without_output_schema",
    ):
        prompt_by_condition[condition] = {
            date: row["variants"][condition]["user_prompt"]
            for date, row in manifest_by_date.items()
        }
        if condition not in {"full_harness", "full_harness_historical"}:
            path = output_dir / f"raw_{condition}.json"
            if path.exists():
                records_by_condition[condition] = json.loads(path.read_text())
    prompt_by_condition["full_harness_historical"] = prompt_by_condition[
        "full_harness"
    ]

    # The historical full condition has three cached runs per week. Current
    # calls use the same run count and are the valid paired reference for new
    # input ablations, avoiding a historical-versus-current model confound.
    runs = len(records_by_condition["full_harness_historical"]) // len(raw)
    for condition in (
        "full_harness_current",
        "without_scenario_evidence",
        "without_output_schema",
    ):
        if condition in records_by_condition:
            integrity[condition] = validate_new_records(
                condition=condition,
                records=records_by_condition[condition],
                manifest_by_date=manifest_by_date,
                runs=runs,
            )

    # Matched ablations are one request-configuration comparison. A
    # condition-local check cannot detect a full-vs-ablated model or sampling
    # drift, so compare every newly generated condition together.
    global_request_configs = {
        json.dumps(record["request_config"], sort_keys=True)
        for condition, records in records_by_condition.items()
        if condition != "full_harness_historical"
        for record in records
        if isinstance(record.get("request_config"), dict)
    }
    if len(global_request_configs) > 1:
        raise ValueError(
            "matched harness conditions use different request configurations"
        )

    rows: list[dict[str, Any]] = []
    coverage: dict[str, dict[int, np.ndarray]] = {}
    for condition, records in records_by_condition.items():
        condition_rows, condition_coverage = evaluate_outputs(
            condition=condition,
            outputs=records,
            source_by_date=source_by_date,
            prompt_by_date=prompt_by_condition[condition],
            budgets=budgets,
            rates=rates,
        )
        rows.extend(condition_rows)
        coverage[condition] = condition_coverage

    gate_reference = (
        "full_harness_current"
        if "full_harness_current" in records_by_condition
        else "full_harness_historical"
    )
    full_records = records_by_condition[gate_reference]
    for gate in ("self_score_action_gate", "numeric_trace_gate"):
        gate_rows, _ = evaluate_outputs(
            condition=gate_reference,
            outputs=full_records,
            source_by_date=source_by_date,
            prompt_by_date=prompt_by_condition[
                "full_harness_current"
                if gate_reference == "full_harness_current"
                else "full_harness"
            ],
            budgets=budgets,
            rates=rates,
            post_gate=gate,
        )
        rows.extend(gate_rows)

    comparisons: list[dict[str, Any]] = []
    comparison_reference = "full_harness_current"
    if comparison_reference not in coverage:
        comparison_reference = "full_harness_historical"
    full_coverage = coverage[comparison_reference]
    for condition in ("without_scenario_evidence", "without_output_schema"):
        if condition not in coverage:
            continue
        condition_records = records_by_condition[condition]
        if len(condition_records) != len(full_records):
            continue
        family: list[dict[str, Any]] = []
        for budget in budgets:
            # Arrays are date-major and run-minor. Average repeated calls within
            # each week before the dependence-aware weekly block bootstrap.
            ablated_weekly = coverage[condition][budget].reshape(len(raw), runs).mean(axis=1)
            full_weekly = full_coverage[budget].reshape(len(raw), runs).mean(axis=1)
            test = bootstrap_delta(
                ablated_weekly,
                full_weekly,
                block_length=block_length,
                n_bootstrap=n_bootstrap,
                seed=seed + budget,
            )
            family.append(
                {
                    "comparison": f"{condition}_minus_full_harness",
                    "budget": budget,
                    "paired_unit": "week after averaging repeated LLM calls",
                    "week_runs": runs,
                    "n_weeks": len(raw),
                    "block_length_weeks": block_length,
                    **test,
                }
            )
        comparisons.extend(family)

    corrected = holm_adjust([test["p_two_sided"] for test in comparisons])
    for test, p_holm in zip(comparisons, corrected, strict=True):
        test["p_holm_across_all_input_controls_and_budgets"] = p_holm

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "weeks": len(raw),
            "runs_per_week": runs,
            "budgets": list(budgets),
            "budget_unit": "exactly K unique protocol targets per week-run",
            "decision_model": model,
            "rate_usd_per_million_tokens": rates,
            "temperature": 0.0,
            "input_comparison_reference": comparison_reference,
            "input_controls": {
                "without_scenario_evidence": "removes only the scenario block",
                "without_output_schema": (
                    "removes JSON-only instruction and explicit output schema"
                ),
            },
            "post_output_controls": {
                "self_score_action_gate": "exact current gate replay",
                "numeric_trace_gate": (
                    "weak proxy, not formal provenance validation"
                ),
            },
            "not_identified": {
                "data_health_gate": (
                    "per-week data_health.safe_mode is absent from frozen M6 artifact"
                ),
                "confidence_gate": (
                    "per-week independent monitor confidence is absent from frozen M6 artifact"
                ),
                "formal_provenance_validator": (
                    "historical ticket schema contains no evidence IDs"
                ),
            },
            "bootstrap": {
                "type": (
                    "paired circular moving-block over weekly means after "
                    "averaging repeated LLM calls"
                ),
                "block_length_weeks": block_length,
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            },
            "integrity": integrity,
            "typed_candidate_set_hashes": {
                date: manifest_row.get("candidate_set_sha256")
                for date, manifest_row in manifest_by_date.items()
            },
            "candidate_membership_hashes": {
                date: manifest_row.get("candidate_membership_sha256")
                for date, manifest_row in manifest_by_date.items()
            },
        },
        "rows": rows,
        "comparisons": comparisons,
    }
    atomic_write_json(output_dir / "results.json", result)
    write_csv(output_dir / "frontier.csv", rows)
    return result


def print_summary(result: dict[str, Any]) -> None:
    print(
        f"{'condition':46s} {'K':>2s} {'coverage':>10s} {'Pmicro':>8s} "
        f"{'int/wr':>7s} {'false/wr':>8s} {'FIR':>7s} {'parse':>7s}"
    )
    for row in result["rows"]:
        precision = row["precision_micro_at_k"]
        fir = row["fir_micro"]
        print(
            f"{row['condition']:46s} {row['budget']:2d} "
            f"{row['coverage_at_k']:10.6f} "
            f"{('n/a' if precision is None else f'{precision:.3f}'):>8s} "
            f"{row['intervention_targets_per_week_run']:7.2f} "
            f"{row['false_interventions_per_week_run']:8.2f} "
            f"{('n/a' if fir is None else f'{fir:.3f}'):>7s} "
            f"{row['parse_success_rate']:7.3f}"
        )
    if result["comparisons"]:
        print("\nPaired input-control comparisons:")
        for test in result["comparisons"]:
            print(
                f"  {test['comparison']} K={test['budget']}: "
                f"delta={test['mean_delta']:+.6f}, "
                f"CI=[{test['ci_95'][0]:+.6f},{test['ci_95'][1]:+.6f}], "
                f"p={test['p_two_sided']:.4f}"
                ", p_holm_all="
                f"{test['p_holm_across_all_input_controls_and_budgets']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Corrected frozen raw_m6_fm_llm.json artifact")
    parser.add_argument("--summary", type=Path, required=True,
                        help="Summary artifact generated by the same corrected run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--budgets", default="1,3,5,7")
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.raw.resolve()
    summary_path = args.summary.resolve()
    if not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("--raw and --summary must name existing files")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log"),
        ],
    )
    raw = load_source(raw_path)
    manifest = build_manifest(raw)
    atomic_write_json(
        output_dir / "prompt_manifest.json",
        {
            "schema_version": 1,
            "source_raw": str(raw_path),
            "source_raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "source_summary": str(summary_path),
            "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "labels_excluded": True,
            "cases": manifest,
        },
    )
    summary = json.loads(summary_path.read_text())
    model = str(summary["model"])
    rates = summary.get("decision_rate_usd_per_million_tokens")
    if not (
        isinstance(rates, dict)
        and set(rates) == {"input", "output"}
        and all(
            isinstance(rates[key], (int, float))
            and not isinstance(rates[key], bool)
            and np.isfinite(float(rates[key]))
            and float(rates[key]) >= 0.0
            for key in ("input", "output")
        )
    ):
        rates = None
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    if not args.prepare_only:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is unset; use --prepare-only or provide it"
            )
        run_input_calls(
            manifest=manifest,
            output_dir=output_dir,
            conditions=(
                "full_harness_current",
                "without_scenario_evidence",
                "without_output_schema",
            ),
            model=model,
            runs=args.runs,
            workers=args.workers,
            api_url=os.environ.get("LLM_API_URL", DEFAULT_API_URL),
            api_key=api_key,
            temperature=0.0,
            timeout=args.timeout,
            seed=args.seed,
        )
    result = analyze(
        raw=raw,
        manifest=manifest,
        output_dir=output_dir,
        model=model,
        rates=rates,
        budgets=budgets,
        block_length=args.block_length,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print_summary(result)
    print(f"\nWrote {output_dir}")


if __name__ == "__main__":
    main()
