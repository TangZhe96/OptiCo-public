# -*- coding: utf-8 -*-
"""Hard checks for a proposed training configuration.

OptiCo-public only asked the model, in the prompt, not to repeat a historical
parameter set. Meta-OptiCo rejects that in code (validate_configuration). This
module is that check, reduced to the public search space and to the values the
public train scripts actually launch.

Several train scripts overwrite the agent's CP / SP / DO / TP_COMM_OVERLAP and
ignore AC_MODULES. Two proposals that differ only in those overwritten fields
launch the same run, so they count as the same configuration.
"""

from __future__ import annotations

import csv
import os
from typing import Mapping, Optional

# Train scripts that assign CP=1, SP=true, DO=true and set
# TP_COMM_OVERLAP from TP, and that never pass --recompute-modules.
# gpt3 does not pass --context-parallel-size at all, so CP is not a real knob.
STANDARD_MODELS = {"llama2", "llama3", "qwen", "qwen2.5", "qwen3-moe", "gpt3"}
MOE_MODELS = {"qwen3-moe", "deepseek_v2", "021"}

SIGNATURE_ORDER = (
    "MICRO_BATCH_SIZE",
    "TP",
    "PP",
    "CP",
    "EP",
    "SP",
    "DO",
    "AC",
    "AC_METHOD",
    "AC_LAYERS",
    "AC_MODULES",
    "OVERLAP_GRAD_REDUCE",
    "OVERLAP_PARAM_GATHER",
    "TP_COMM_OVERLAP",
    "FUSED_CROSS_ENTROPY",
    "FIRST_PP_NUM_LAYERS",
    "LAST_PP_NUM_LAYERS",
)

_MISSING = {"", "none", "null", "nan", "nil"}


def model_name(model: str) -> str:
    text = (model or "").strip().lower()
    if text in ("qwen", "qwen2.5"):
        return "qwen2.5"
    if text.startswith("deepseek"):
        return "deepseek_v2"
    return text


def _missing(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _MISSING


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if _missing(value):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value, default: int) -> int:
    if _missing(value):
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _as_ac(value) -> str:
    if _missing(value):
        return "none"
    text = str(value).strip().lower()
    if text == "selective":
        return "sel"
    if text in {"none", "full", "sel", "offload"}:
        return text
    return text


def _pp_endpoint(value) -> str:
    if _missing(value):
        return "none"
    try:
        return str(int(float(str(value).strip())))
    except (TypeError, ValueError):
        return "none"


def _flag(value: bool) -> str:
    return "true" if value else "false"


def is_moe(model: str, params: Mapping) -> bool:
    if model_name(model) in MOE_MODELS:
        return True
    return "EP" in params


def canonicalize(params: Mapping, model: str = "") -> dict:
    """Values the public train script and monitor will record."""
    model = model_name(model)
    standard = model in STANDARD_MODELS
    src = params or {}

    micro = max(1, _as_int(src.get("MICRO_BATCH_SIZE"), 1))
    tp = max(1, _as_int(src.get("TP"), 1))
    pp = max(1, _as_int(src.get("PP"), 1))
    cp = 1 if standard else max(1, _as_int(src.get("CP"), 1))
    ep = max(1, _as_int(src.get("EP"), 1))
    ac = _as_ac(src.get("AC"))
    if ac not in {"none", "full", "sel", "offload"}:
        ac = "none"

    if standard or tp <= 1:
        sp = tp > 1
    else:
        sp = _as_bool(src.get("SP"), False) and tp > 1

    do = True if standard else _as_bool(src.get("DO"), True)

    if ac == "full":
        if standard:
            ac_method = "uniform"
        else:
            ac_method = str(src.get("AC_METHOD") or "").strip().lower()
            if ac_method not in {"uniform", "block"}:
                ac_method = "uniform"
    else:
        ac_method = "none"

    ac_layers = max(1, _as_int(src.get("AC_LAYERS"), 1))
    if standard:
        ac_modules = "none"
    else:
        modules = src.get("AC_MODULES")
        ac_modules = "none" if _missing(modules) else str(modules).strip()

    overlap_grad = _as_bool(src.get("OVERLAP_GRAD_REDUCE"), False)
    overlap_param = _as_bool(src.get("OVERLAP_PARAM_GATHER"), False)
    if standard and ac == "offload":
        overlap_grad = False
        overlap_param = False

    if standard:
        tp_comm = tp > 1
    else:
        tp_comm = _as_bool(src.get("TP_COMM_OVERLAP"), False) and tp > 1

    fused = _as_bool(src.get("FUSED_CROSS_ENTROPY"), False)

    if pp <= 1:
        first = "none"
        last = "none"
    else:
        first = _pp_endpoint(src.get("FIRST_PP_NUM_LAYERS"))
        last = _pp_endpoint(src.get("LAST_PP_NUM_LAYERS"))

    return {
        "MICRO_BATCH_SIZE": str(micro),
        "TP": str(tp),
        "PP": str(pp),
        "CP": str(cp),
        "EP": str(ep),
        "SP": _flag(sp),
        "DO": _flag(do),
        "AC": ac,
        "AC_METHOD": ac_method,
        "AC_LAYERS": str(ac_layers),
        "AC_MODULES": ac_modules,
        "OVERLAP_GRAD_REDUCE": _flag(overlap_grad),
        "OVERLAP_PARAM_GATHER": _flag(overlap_param),
        "TP_COMM_OVERLAP": _flag(tp_comm),
        "FUSED_CROSS_ENTROPY": _flag(fused),
        "FIRST_PP_NUM_LAYERS": first,
        "LAST_PP_NUM_LAYERS": last,
    }


def effective_signature(canon: Mapping, model: str = "", moe: bool = False) -> dict:
    """Fields that change the launched command. No-op fields are left out."""
    ac = canon.get("AC", "none")
    keys = [
        "MICRO_BATCH_SIZE",
        "TP",
        "PP",
        "CP",
        "SP",
        "DO",
        "AC",
        "OVERLAP_GRAD_REDUCE",
        "OVERLAP_PARAM_GATHER",
        "TP_COMM_OVERLAP",
        "FUSED_CROSS_ENTROPY",
        "FIRST_PP_NUM_LAYERS",
        "LAST_PP_NUM_LAYERS",
    ]
    if ac == "full":
        keys.extend(["AC_METHOD", "AC_LAYERS"])
    elif ac == "offload":
        keys.append("AC_LAYERS")
    if model_name(model) not in STANDARD_MODELS:
        keys.append("AC_MODULES")
    if moe:
        keys.append("EP")
    return {key: canon.get(key) for key in keys}


def format_parameters(params: Mapping) -> str:
    lines = []
    for key in SIGNATURE_ORDER:
        if key in params:
            lines.append(f"{key}: {params[key]}")
    return "\n".join(lines)


def apply_canonical(original: Mapping, canon: Mapping) -> dict:
    """Keep the parameter file's keys and order; store launched values."""
    merged = {}
    for key, value in original.items():
        merged[key] = canon[key] if key in canon else value
    return merged


def load_fixed(path: str) -> dict:
    found = {}
    if not path or not os.path.isfile(path):
        return found
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            found[key.strip().lower()] = value.strip()
    return found


def _fixed_int(fixed: Mapping, *names: str) -> Optional[int]:
    for name in names:
        raw = fixed.get(name)
        if _missing(raw):
            continue
        try:
            return int(float(str(raw).strip()))
        except (TypeError, ValueError):
            continue
    return None


def load_history(report_path: str) -> list:
    if not report_path or not os.path.isfile(report_path):
        return []
    rows = []
    with open(report_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            cleaned = {}
            for key, value in row.items():
                if not key:
                    continue
                cleaned[key.strip()] = "" if value is None else str(value).strip()
            if cleaned:
                rows.append(cleaned)
    return rows


def check_constraints(canon: Mapping, fixed: Mapping, model: str = "") -> tuple:
    """Return (ok, reason). Missing fixed facts skip only the checks that need them."""
    model = model_name(model or fixed.get("model type", ""))
    try:
        micro = int(canon["MICRO_BATCH_SIZE"])
        tp = int(canon["TP"])
        pp = int(canon["PP"])
        cp = int(canon["CP"])
        ep = int(canon["EP"])
        ac_layers = int(canon["AC_LAYERS"])
    except (KeyError, TypeError, ValueError):
        return False, "MICRO_BATCH_SIZE/TP/PP/CP/EP/AC_LAYERS must be integers"

    if min(micro, tp, pp, cp, ep, ac_layers) < 1:
        return False, "MICRO_BATCH_SIZE/TP/PP/CP/EP/AC_LAYERS must be >= 1"
    if canon["AC"] not in {"none", "full", "sel", "offload"}:
        return False, "AC must be one of none/full/sel/offload"
    if canon["AC"] == "full" and canon["AC_METHOD"] not in {"uniform", "block"}:
        return False, "AC=full requires AC_METHOD uniform or block"

    overlap_param = canon["OVERLAP_PARAM_GATHER"] == "true"
    if overlap_param and (
        canon["DO"] != "true" or canon["OVERLAP_GRAD_REDUCE"] != "true"
    ):
        return False, "OVERLAP_PARAM_GATHER requires DO=true and OVERLAP_GRAD_REDUCE=true"

    first = canon["FIRST_PP_NUM_LAYERS"]
    last = canon["LAST_PP_NUM_LAYERS"]
    if pp <= 1 and (first != "none" or last != "none"):
        return False, "PP=1 cannot set FIRST_PP_NUM_LAYERS or LAST_PP_NUM_LAYERS"
    if (first == "none") != (last == "none"):
        return False, "FIRST_PP_NUM_LAYERS and LAST_PP_NUM_LAYERS must both be set or both be none"

    num_gpus = _fixed_int(fixed, "number of gpus", "num_gpus", "nproc")
    global_batch = _fixed_int(fixed, "global batch size", "global_batch_size", "gbs")
    num_layers = _fixed_int(fixed, "num_layers", "num layers")
    num_experts = _fixed_int(fixed, "num_experts", "num experts")

    if num_gpus is not None:
        mesh = tp * pp * cp
        if mesh <= 0 or num_gpus % mesh != 0:
            return False, f"Number of GPUs ({num_gpus}) must be divisible by TP*PP*CP ({mesh})"
        if is_moe(model, canon) and ep > 1:
            expert_mesh = ep * pp
            if num_gpus % expert_mesh != 0:
                return False, (
                    f"Number of GPUs ({num_gpus}) must be divisible by EP*PP ({expert_mesh})"
                )
        if global_batch is not None:
            dp = num_gpus // mesh
            unit = micro * dp
            if unit <= 0 or global_batch % unit != 0:
                return False, (
                    f"Global Batch Size ({global_batch}) must be divisible by "
                    f"MICRO_BATCH_SIZE*DP ({micro}*{dp})"
                )

    if is_moe(model, canon) and num_experts is not None and num_experts > 0:
        if num_experts % ep != 0:
            return False, f"NUM_EXPERTS ({num_experts}) must be divisible by EP ({ep})"

    stage_counts = None
    if num_layers is not None and num_layers > 0:
        if first == "none":
            if num_layers % pp != 0:
                return False, f"NUM_LAYERS ({num_layers}) must be divisible by PP ({pp})"
            stage_counts = [num_layers // pp] * pp
        else:
            first_n = int(first)
            last_n = int(last)
            if first_n < 1 or last_n < 1:
                return False, "FIRST_PP_NUM_LAYERS and LAST_PP_NUM_LAYERS must be >= 1"
            middle = num_layers - first_n - last_n
            if pp == 2:
                if middle != 0:
                    return False, "When PP=2, FIRST_PP_NUM_LAYERS + LAST_PP_NUM_LAYERS must equal NUM_LAYERS"
                stage_counts = [first_n, last_n]
            else:
                if middle <= 0:
                    return False, "Middle layer count must be > 0 when PP>2"
                if middle % (pp - 2) != 0:
                    return False, "Middle layer count must be divisible by (PP-2)"
                mid = middle // (pp - 2)
                stage_counts = [first_n] + [mid] * (pp - 2) + [last_n]

    if canon["AC"] == "full" and stage_counts is not None:
        for count in stage_counts:
            if count % ac_layers != 0:
                return False, "Each pipeline stage's layer count must be divisible by AC_LAYERS when AC=full"

    return True, "Configuration passes all checks"


def find_duplicate(signature: Mapping, history: list, model: str, moe: bool) -> Optional[int]:
    for index, row in enumerate(history, start=1):
        past = effective_signature(canonicalize(row, model), model, moe)
        if past == signature:
            return index
    return None


def validate_proposal(params: Mapping, report_path: str, fixed_path: str, model: str = "") -> dict:
    fixed = load_fixed(fixed_path)
    resolved_model = model_name(model or fixed.get("model type", ""))
    canon = canonicalize(params, resolved_model)
    moe = is_moe(resolved_model, params)
    ok, reason = check_constraints(canon, fixed, resolved_model)
    shown = effective_signature(canon, resolved_model, moe)
    if not ok:
        return {
            "validation_result": "INVALID",
            "reason": reason,
            "parameters": shown,
            "canonical": canon,
        }
    duplicate_index = find_duplicate(shown, load_history(report_path), resolved_model, moe)
    if duplicate_index is not None:
        return {
            "validation_result": "DUPLICATE",
            "reason": f"Matches historical configuration #{duplicate_index}",
            "duplicate_index": duplicate_index,
            "parameters": shown,
            "canonical": canon,
        }
    return {
        "validation_result": "VALID",
        "reason": "Configuration passes all checks",
        "parameters": shown,
        "canonical": canon,
    }


def self_check() -> None:
    fixed = {
        "model type": "llama2",
        "num_layers": "32",
        "global batch size": "32",
        "number of gpus": "8",
    }
    base = {
        "MICRO_BATCH_SIZE": "1",
        "TP": "2",
        "PP": "2",
        "CP": "1",
        "SP": "true",
        "DO": "true",
        "AC": "none",
        "AC_METHOD": "none",
        "AC_LAYERS": "1",
        "AC_MODULES": "none",
        "OVERLAP_GRAD_REDUCE": "false",
        "OVERLAP_PARAM_GATHER": "false",
        "TP_COMM_OVERLAP": "true",
        "FUSED_CROSS_ENTROPY": "false",
        "FIRST_PP_NUM_LAYERS": "none",
        "LAST_PP_NUM_LAYERS": "none",
    }
    # DO/CP/AC_MODULES differ on paper, but llama2 launches the same run.
    tweaked = dict(base)
    tweaked["DO"] = "false"
    tweaked["CP"] = "4"
    tweaked["AC_MODULES"] = "core_attn"
    tweaked["AC_LAYERS"] = "8"
    assert effective_signature(canonicalize(base, "llama2"), "llama2") == effective_signature(
        canonicalize(tweaked, "llama2"), "llama2"
    )

    history = [base]
    again = find_duplicate(
        effective_signature(canonicalize(tweaked, "llama2"), "llama2"),
        history,
        "llama2",
        False,
    )
    assert again == 1, again

    changed = dict(base)
    changed["MICRO_BATCH_SIZE"] = "2"
    assert find_duplicate(
        effective_signature(canonicalize(changed, "llama2"), "llama2"),
        history,
        "llama2",
        False,
    ) is None

    bad = dict(base)
    bad["MICRO_BATCH_SIZE"] = "3"
    ok, reason = check_constraints(canonicalize(bad, "llama2"), fixed, "llama2")
    assert not ok and "divisible" in reason, reason

    uneven = dict(base)
    uneven["PP"] = "1"
    uneven["TP"] = "1"
    uneven["FIRST_PP_NUM_LAYERS"] = "4"
    uneven["LAST_PP_NUM_LAYERS"] = "4"
    canon = canonicalize(uneven, "llama2")
    assert canon["FIRST_PP_NUM_LAYERS"] == "none"
    assert canon["TP_COMM_OVERLAP"] == "false"
    ok, reason = check_constraints(canon, fixed, "llama2")
    assert ok, reason


if __name__ == "__main__":
    self_check()
    print("param_check ok")
