#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write / refresh best_parameters.json with TGS, LLM cost, and split search timings.

OptiCo-public adaptations vs Meta-OptiCo:
- parameters.txt (Key: Value) as well as parameters.json
- infer_time: X s in analyzer logs (also accepts Meta's API infer duration)
- fixed_parameters.txt fallback for GBS / SEQ / NPROC
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


TS_RE = re.compile(r"^\[(\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2})-")
TIMEOUT_DURATION_RE = re.compile(r"API timeout duration:\s*([\d.]+)\s*s", re.I)
TIMEOUT_FAIL_RE = re.compile(r"API call failed:.*timed out", re.I)
RETRY_WAIT_RE = re.compile(r"Retry\s*\[\d+\].*waiting\s+(\d+)\s+sec", re.I)
# Meta: "API infer duration: X s"; public: "infer_time: X s"
INFER_DURATION_RE = re.compile(
    r"(?:API infer duration|infer_time):\s*([\d.]+)\s*s", re.I
)
RESPONSE_IS_RE = re.compile(r"\]\s*\[[^\]]+\]\s*response is:", re.I)
HTTP_OK_RE = re.compile(r"HTTP Request:.*\b200\b", re.I)
HTTP_REQ_RE = re.compile(r"HTTP Request:", re.I)
TGS_LOG_RE = re.compile(r"throughput \(token/sec/GPU\)\s*:\s*(\d+\.\d+)")
FIXED_TXT_RE = re.compile(
    r"^(Global Batch Size|Sequence Length|Number of GPUs)\s*:\s*(.+)$",
    re.I,
)


def _to_float(value, default=0.0) -> float:
    try:
        if value is None or value == "" or str(value).upper() == "NULL":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _line_ts(line: str) -> Optional[datetime]:
    match = TS_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y.%m.%d-%H.%M.%S")


def _coerce_param_value(raw: str):
    text = raw.strip()
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    try:
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if re.fullmatch(r"-?\d+\.\d+", text):
            return float(text)
    except ValueError:
        pass
    return text


def load_parameters(para_file: str) -> dict:
    """Load parameters.json or parameters.txt (Key: Value)."""
    path = Path(para_file)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(text)
    params = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        params[key.strip()] = _coerce_param_value(value)
    return params


def load_fixed_from_txt(output_path: str) -> dict:
    """Parse OptiCo-public fixed_parameters.txt into gbs/seq/nproc."""
    path = Path(output_path) / "fixed_parameters.txt"
    out = {"gbs": 0.0, "seq": 0.0, "nproc": 0.0}
    if not path.is_file():
        return out
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return out
    for line in lines:
        match = FIXED_TXT_RE.match(line.strip())
        if not match:
            continue
        label, value = match.group(1).lower(), match.group(2).strip()
        if "batch" in label:
            out["gbs"] = _to_float(value)
        elif "sequence" in label:
            out["seq"] = _to_float(value)
        elif "gpu" in label:
            out["nproc"] = _to_float(value)
    return out


def _sum_duration_logs(lines: list[str]) -> float:
    total = 0.0
    for idx, line in enumerate(lines):
        match = TIMEOUT_DURATION_RE.search(line)
        if not match:
            continue
        total += float(match.group(1))
        if idx + 1 < len(lines):
            wait = RETRY_WAIT_RE.search(lines[idx + 1])
            if wait:
                total += float(wait.group(1))
    return total


def _sum_timeout_gaps(lines: list[str]) -> float:
    total = 0.0
    prev_ts = None
    idx = 0
    n = len(lines)
    while idx < n:
        line = lines[idx]
        ts = _line_ts(line)
        if TIMEOUT_FAIL_RE.search(line) and ts is not None:
            if prev_ts is not None:
                total += max(0.0, (ts - prev_ts).total_seconds())
            wait = 0.0
            if idx + 1 < n:
                wait_match = RETRY_WAIT_RE.search(lines[idx + 1])
                if wait_match:
                    wait = float(wait_match.group(1))
            next_is_timeout = False
            look = idx + 1
            while look < n:
                if TIMEOUT_FAIL_RE.search(lines[look]):
                    next_is_timeout = True
                    break
                look_ts = _line_ts(lines[look])
                if look_ts is not None and not RETRY_WAIT_RE.search(lines[look]):
                    break
                look += 1
            if not next_is_timeout:
                total += wait
            prev_ts = ts
            idx += 1
            continue
        if ts is not None:
            prev_ts = ts
        idx += 1
    return total


def _iter_analyzer_logs(output_path: str, max_iter: int):
    root = Path(output_path)
    for it in range(1, int(max_iter) + 1):
        iter_dir = root / str(it)
        if not iter_dir.is_dir():
            continue
        for log_path in sorted(iter_dir.glob("analyzer_log_*.log")):
            try:
                yield log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue


def parse_api_timeout_seconds(output_path: str, max_iter: int) -> float:
    """Sum API timeout (+ post-timeout retry wait) from analyzer logs through max_iter."""
    total = 0.0
    for lines in _iter_analyzer_logs(output_path, max_iter):
        if any(TIMEOUT_DURATION_RE.search(line) for line in lines):
            total += _sum_duration_logs(lines)
        else:
            total += _sum_timeout_gaps(lines)
    return round(total, 3)


def _sum_infer_duration_logs(lines: list[str]) -> float:
    total = 0.0
    for line in lines:
        match = INFER_DURATION_RE.search(line)
        if match:
            total += float(match.group(1))
    return total


def _sum_infer_gaps(lines: list[str]) -> float:
    """Estimate successful API wait from timestamps when infer-duration logs are absent."""
    total = 0.0
    for idx, line in enumerate(lines):
        if HTTP_OK_RE.search(line):
            ts_end = _line_ts(line)
        elif RESPONSE_IS_RE.search(line):
            if idx > 0 and HTTP_OK_RE.search(lines[idx - 1]):
                continue
            ts_end = _line_ts(line)
        else:
            continue
        if ts_end is None:
            continue
        ts_start = None
        for look in range(idx - 1, -1, -1):
            prev = lines[look]
            if HTTP_REQ_RE.search(prev):
                continue
            if (
                TIMEOUT_DURATION_RE.search(prev)
                or TIMEOUT_FAIL_RE.search(prev)
                or RETRY_WAIT_RE.search(prev)
            ):
                break
            ts = _line_ts(prev)
            if ts is None:
                continue
            ts_start = ts
            break
        if ts_start is not None:
            total += max(0.0, (ts_end - ts_start).total_seconds())
    return total


def parse_api_infer_seconds(output_path: str, max_iter: int) -> float:
    """Sum successful LLM request duration from analyzer logs through max_iter."""
    total = 0.0
    for lines in _iter_analyzer_logs(output_path, max_iter):
        if any(INFER_DURATION_RE.search(line) for line in lines):
            total += _sum_infer_duration_logs(lines)
        else:
            total += _sum_infer_gaps(lines)
    return round(total, 3)


def _wall_elapsed_from_payload(current: dict) -> float:
    elapsed = _to_float(current.get("ElapsedTime"))
    timeout = _to_float(current.get("ApiTimeoutTime"))
    if "TuningTime" in current:
        return elapsed
    return elapsed + timeout


def load_llm_stats(output_path: str, max_iter: int) -> dict:
    stats = {
        "LLMCalls": 0,
        "PromptTokens": 0,
        "CompletionTokens": 0,
        "TotalTokens": 0,
    }
    csv_path = Path(output_path) / "llm_stats.csv"
    if not csv_path.is_file():
        return stats
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            # tolerate BOM / whitespace on header keys
            norm = {(k or "").lstrip("\ufeff").strip(): v for k, v in row.items()}
            try:
                it = int(norm.get("Iter") or 0)
            except (TypeError, ValueError):
                continue
            if it < 1 or it > int(max_iter):
                continue
            stats["LLMCalls"] += 1
            stats["PromptTokens"] += int(_to_float(norm.get("PromptTokens")))
            stats["CompletionTokens"] += int(_to_float(norm.get("CompletionTokens")))
            stats["TotalTokens"] += int(_to_float(norm.get("TotalTokens")))
    return stats


def compute_tgs_from_log(log_file: str) -> float:
    path = Path(log_file)
    if not path.is_file():
        return 0.0
    values = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0.0
    for match in TGS_LOG_RE.finditer(text):
        values.append(float(match.group(1)))
    usable = values[2:] if len(values) > 2 else values
    if not usable:
        return 0.0
    return round(sum(usable) / len(usable), 1)


def compute_tgs_from_formula(iter_time: float, gbs: float, seq: float, nproc: float) -> float:
    if iter_time <= 0 or nproc <= 0 or gbs <= 0 or seq <= 0:
        return 0.0
    return round(gbs * seq / (iter_time * nproc), 1)


def resolve_tgs(
    tgs: Optional[float],
    output_path: str,
    it: int,
    iter_time: float,
    gbs: Optional[float] = None,
    seq: Optional[float] = None,
    nproc: Optional[float] = None,
) -> float:
    if tgs is not None and tgs > 0:
        return round(float(tgs), 1)
    root = Path(output_path)
    fixed_path = root / "fixed_args.json"
    if fixed_path.is_file():
        with fixed_path.open("r", encoding="utf-8") as f:
            fixed = json.load(f)
        if not gbs:
            gbs = _to_float(fixed.get("global_batch_size"))
        if not seq:
            seq = _to_float(fixed.get("seq_length"))
        if not nproc:
            nproc = _to_float(fixed.get("num_gpus"))
    if not gbs or not seq or not nproc:
        fixed_txt = load_fixed_from_txt(output_path)
        if not gbs:
            gbs = fixed_txt["gbs"]
        if not seq:
            seq = fixed_txt["seq"]
        if not nproc:
            nproc = fixed_txt["nproc"]
    iter_log_dir = root / str(it) / "megatron_log"
    if iter_log_dir.is_dir():
        numeric = [int(p.stem) for p in iter_log_dir.glob("*.txt") if p.stem.isdigit()]
        if numeric:
            from_log = compute_tgs_from_log(str(iter_log_dir / f"{max(numeric)}.txt"))
            if from_log > 0:
                return from_log
    return compute_tgs_from_formula(
        iter_time,
        _to_float(gbs),
        _to_float(seq),
        _to_float(nproc),
    )


def build_payload(
    params: dict,
    mfu: float,
    tflops: float,
    tgs: float,
    iter_time: float,
    it: int,
    elapsed: float,
    output_path: str,
) -> dict:
    llm = load_llm_stats(output_path, it)
    infer = parse_api_infer_seconds(output_path, it)
    timeout = parse_api_timeout_seconds(output_path, it)
    wall = float(elapsed)
    tuning = max(0.0, wall - infer - timeout)
    return {
        "MFU": float(mfu),
        "TFLOPS": float(tflops),
        "TGS": float(tgs),
        "IterTime": float(iter_time),
        "ElapsedTime": round(wall, 3),
        "ApiInferTime": float(infer),
        "ApiTimeoutTime": float(timeout),
        "TuningTime": round(tuning, 3),
        "LLMCalls": int(llm["LLMCalls"]),
        "PromptTokens": int(llm["PromptTokens"]),
        "CompletionTokens": int(llm["CompletionTokens"]),
        "TotalTokens": int(llm["TotalTokens"]),
        "iter": int(it),
        "parameters": params,
    }


def write_best_parameters(
    para_file: str,
    output_file: str,
    output_path: str,
    mfu: float,
    tflops: float,
    iter_time: float,
    it: int,
    elapsed: float,
    tgs: Optional[float] = None,
    gbs: Optional[float] = None,
    seq: Optional[float] = None,
    nproc: Optional[float] = None,
) -> dict:
    params = load_parameters(para_file)
    resolved_tgs = resolve_tgs(tgs, output_path, it, iter_time, gbs, seq, nproc)
    payload = build_payload(
        params, mfu, tflops, resolved_tgs, iter_time, it, elapsed, output_path
    )
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return payload


def backfill(output_path: str) -> dict:
    root = Path(output_path)
    best_path = root / "best_parameters.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"best_parameters.json not found: {best_path}")
    with best_path.open("r", encoding="utf-8") as f:
        current = json.load(f)
    it = int(current.get("iter") or 0)
    if it < 1:
        raise ValueError(f"best_parameters.json missing valid iter: {best_path}")
    para_json = root / str(it) / "parameters.json"
    para_txt = root / str(it) / "parameters.txt"
    if para_json.is_file():
        para_file = para_json
    elif para_txt.is_file():
        para_file = para_txt
    else:
        raise FileNotFoundError(
            f"parameters.json/txt not found for iter {it} under {root / str(it)}"
        )
    elapsed = _wall_elapsed_from_payload(current)
    payload = write_best_parameters(
        para_file=str(para_file),
        output_file=str(best_path),
        output_path=str(root),
        mfu=_to_float(current.get("MFU")),
        tflops=_to_float(current.get("TFLOPS")),
        iter_time=_to_float(current.get("IterTime")),
        it=it,
        elapsed=elapsed,
        tgs=_to_float(current.get("TGS")) or None,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write best_parameters.json extras.")
    parser.add_argument("--backfill", type=str, default="", help="Refresh an existing output dir.")
    parser.add_argument("--para-file", type=str, default="")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--mfu", type=float, default=0.0)
    parser.add_argument("--tflops", type=float, default=0.0)
    parser.add_argument("--tgs", type=float, default=0.0)
    parser.add_argument("--iter-time", type=float, default=0.0)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--elapsed", type=float, default=0.0)
    parser.add_argument("--gbs", type=float, default=0.0)
    parser.add_argument("--seq", type=float, default=0.0)
    parser.add_argument("--nproc", type=float, default=0.0)
    args = parser.parse_args()

    if args.backfill:
        payload = backfill(args.backfill)
    else:
        if not args.para_file or not args.output_file or not args.output_path:
            raise SystemExit("--para-file, --output-file and --output-path are required")
        payload = write_best_parameters(
            para_file=args.para_file,
            output_file=args.output_file,
            output_path=args.output_path,
            mfu=args.mfu,
            tflops=args.tflops,
            iter_time=args.iter_time,
            it=args.iter,
            elapsed=args.elapsed,
            tgs=args.tgs or None,
            gbs=args.gbs or None,
            seq=args.seq or None,
            nproc=args.nproc or None,
        )
    print(
        "update best_parameters.json "
        f"MFU={payload['MFU']} TFLOPS={payload['TFLOPS']} TGS={payload['TGS']} "
        f"IterTime={payload['IterTime']} ElapsedTime={payload['ElapsedTime']}s "
        f"ApiInferTime={payload['ApiInferTime']}s "
        f"ApiTimeoutTime={payload['ApiTimeoutTime']}s "
        f"TuningTime={payload['TuningTime']}s iter={payload['iter']} "
        f"LLMCalls={payload['LLMCalls']} "
        f"tokens={payload['PromptTokens']}/{payload['CompletionTokens']}/{payload['TotalTokens']}"
    )


if __name__ == "__main__":
    main()
