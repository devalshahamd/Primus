"""Token-step serving simulator (vLLM V1 unified-batch policy).

Replays a workload of requests through a continuous-batching scheduler where
each step is one forward pass (no wall-clock model) and records, for every
request scheduled in every step, its query length, KV length and phase. This
exposes the distribution of how requests get packed together at each step.

See README.md for the design and Further_improvements.md for the knobs that are
intentionally fixed or deferred in this starting-point version.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Config:
    # --- Workload ---
    num_prompts: int = 256
    request_rate: float = float("inf")  # req/s; inf => all arrive at step 0
    burstiness: float = 1.0             # gamma shape; 1.0 => Poisson
    ISL: int = 1024                     # MAX input length; actual ~ U[R*ISL, ISL]
    OSL: int = 128                      # MAX output length; actual ~ U[R*OSL, OSL]
    range_ratio: float = 1.0            # lower-bound fraction for the uniform sampling
    seed: int = 0
    workload_file: Optional[str] = None  # if set, load (arrival, isl, osl) instead of sampling

    # --- Capacity / admission ---
    CONC: int = 64                      # max concurrent requests in the system
    max_num_batched_tokens: int = 8192  # per-step token budget across the whole batch
    long_prefill_token_threshold: int = 0  # per-request prefill chunk cap (0 = off)
    max_model_len: int = 32768          # per-request context cap (prompt + output)

    # --- KV cache ---
    kv_cache_tokens: int = 1_000_000    # total KV token-slot pool (flat budget)

    # Fixed in v1 (see Further_improvements.md): enable_chunked_prefill=True,
    # scheduler_reserve_full_isl=True, FCFS queue policy.


@dataclass
class Request:
    rid: int
    arrival_step: int
    prompt_len: int
    output_len: int
    num_computed_tokens: int = 0        # tokens already computed / in KV
    num_tokens: int = field(init=False)  # known token ids so far = prompt + generated
    generated: int = 0
    status: str = "WAITING"             # WAITING | RUNNING | FINISHED

    def __post_init__(self):
        self.num_tokens = self.prompt_len

    @property
    def reserved_kv(self) -> int:
        # Full-ISL reservation: a request reserves its whole sequence up front.
        return self.prompt_len + self.output_len

    @property
    def in_prefill(self) -> bool:
        return self.num_computed_tokens < self.prompt_len


def build_workload(cfg: Config) -> List[Request]:
    """Construct the request list, either from a file or by synthetic sampling."""
    if cfg.workload_file:
        return _load_workload_file(cfg.workload_file)

    rng = np.random.RandomState(cfg.seed)

    def sample_len(max_len: int) -> np.ndarray:
        lower = max(1, int(max_len * cfg.range_ratio))
        return rng.randint(lower, max_len + 1, size=cfg.num_prompts)

    isl = sample_len(cfg.ISL)
    osl = sample_len(cfg.OSL)

    # Arrival steps: inf rate => everything available at step 0 (saturation).
    # Finite rate => gamma inter-arrival times, floored to integer step indices
    # (1 time-unit == 1 step; documented simplification).
    if math.isinf(cfg.request_rate):
        arrivals = [0] * cfg.num_prompts
    else:
        theta = 1.0 / (cfg.request_rate * cfg.burstiness)
        intervals = rng.gamma(shape=cfg.burstiness, scale=theta, size=cfg.num_prompts)
        times = np.cumsum(intervals)
        arrivals = [int(t) for t in times]

    return [
        Request(rid=i, arrival_step=int(arrivals[i]),
                prompt_len=int(isl[i]), output_len=int(osl[i]))
        for i in range(cfg.num_prompts)
    ]


def _load_workload_file(path: str) -> List[Request]:
    """Load requests from JSON (list of dicts) or CSV. Columns/keys (case-insensitive):
    arrival (step id), isl (prompt len), osl (output len)."""
    rows: List[dict] = []
    if path.endswith(".json"):
        with open(path) as f:
            rows = json.load(f)
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

    def get(row, *names, default=0):
        lower = {k.lower(): v for k, v in row.items()}
        for n in names:
            if n in lower and lower[n] not in (None, ""):
                return int(float(lower[n]))
        return default

    reqs = []
    for i, row in enumerate(rows):
        reqs.append(Request(
            rid=i,
            arrival_step=get(row, "arrival", "arrival_step", "step", default=0),
            prompt_len=get(row, "isl", "input_len", "prompt_len", default=1),
            output_len=get(row, "osl", "output_len", default=1),
        ))
    return reqs


def simulate(cfg: Config, requests: List[Request]) -> List[dict]:
    """Run the unified-batch scheduler and return one record per execution step."""
    pending = sorted(requests, key=lambda r: (r.arrival_step, r.rid))  # FCFS
    waiting: List[Request] = []
    running: List[Request] = []
    kv_used = 0
    records: List[dict] = []

    step = 0
    next_arrival = 0
    n = len(pending)

    while next_arrival < n or waiting or running:
        # 1. Admit arrivals due at this step into the FCFS waiting queue.
        while next_arrival < n and pending[next_arrival].arrival_step <= step:
            waiting.append(pending[next_arrival])
            next_arrival += 1

        token_budget = cfg.max_num_batched_tokens
        scheduled: List[dict] = []

        # 2a. Phase 1 - already-running requests (decodes + in-progress prefills).
        for req in running:
            q = _schedule_tokens(cfg, req, token_budget)
            if q == 0:
                continue
            scheduled.append(_record_req(req, q))
            token_budget -= q
            _advance(req, q)

        # 2b. Phase 2 - admit new waiting requests (full-ISL reservation gate).
        while waiting and len(running) < cfg.CONC and token_budget > 0:
            req = waiting[0]
            if kv_used + req.reserved_kv > cfg.kv_cache_tokens:
                break  # cannot reserve full sequence; head-of-line wait
            q = _schedule_tokens(cfg, req, token_budget)
            if q == 0:
                break  # no budget left for even one token
            waiting.pop(0)
            req.status = "RUNNING"
            running.append(req)
            kv_used += req.reserved_kv
            scheduled.append(_record_req(req, q))
            token_budget -= q
            _advance(req, q)

        # 3. Retire finished requests and free their KV reservations.
        still_running = []
        for req in running:
            if req.status == "FINISHED":
                kv_used -= req.reserved_kv
            else:
                still_running.append(req)
        running = still_running

        if scheduled:
            records.append(_step_record(step, scheduled, kv_used))
        step += 1

        # Safety valve against accidental infinite loops.
        if step > 10_000_000:
            raise RuntimeError("step limit exceeded; check workload/config")

    return records


def _schedule_tokens(cfg: Config, req: Request, token_budget: int) -> int:
    """Number of tokens to compute for `req` this step (0 if it can't run)."""
    need = req.num_tokens - req.num_computed_tokens
    if need <= 0:
        return 0
    if cfg.long_prefill_token_threshold > 0:
        need = min(need, cfg.long_prefill_token_threshold)
    need = min(need, token_budget)
    need = min(need, cfg.max_model_len - 1 - req.num_computed_tokens)
    return max(need, 0)


def _record_req(req: Request, q: int) -> dict:
    return {
        "request_id": req.rid,
        "phase": "prefill" if req.in_prefill else "decode",
        "query_len": q,
        "kv_len": req.num_computed_tokens,  # context length attended to this step
    }


def _advance(req: Request, q: int) -> None:
    """Apply a forward pass: advance computed tokens, sample a token if caught up."""
    req.num_computed_tokens += q
    if req.num_computed_tokens == req.num_tokens:
        # Model emits one new token whenever computation catches up to the
        # known token ids (last prefill chunk or a decode step).
        req.generated += 1
        req.num_tokens += 1
        if req.generated >= req.output_len:
            req.status = "FINISHED"


def _step_record(step: int, scheduled: List[dict], kv_used: int) -> dict:
    n_pref = sum(1 for r in scheduled if r["phase"] == "prefill")
    n_dec = len(scheduled) - n_pref
    return {
        "step": step,
        "batch_size": len(scheduled),
        "num_prefill_reqs": n_pref,
        "num_decode_reqs": n_dec,
        "total_query_tokens": sum(r["query_len"] for r in scheduled),
        "kv_tokens_in_use": kv_used,
        "requests": scheduled,
    }


def _selftest() -> None:
    # Single request: ISL prefill chunks then OSL decode steps, all under budget.
    cfg = Config(num_prompts=1, ISL=100, OSL=5, range_ratio=1.0,
                 max_num_batched_tokens=40, CONC=4, kv_cache_tokens=10_000)
    recs = simulate(cfg, build_workload(cfg))
    # Prefill 100 tokens in chunks of <=40 => 3 prefill steps (40,40,20). The
    # last prefill step also samples the 1st output token, so decode steps = OSL-1.
    pref_steps = [r for r in recs if r["num_prefill_reqs"] > 0]
    dec_steps = [r for r in recs if r["num_decode_reqs"] > 0]
    assert len(pref_steps) == 3, pref_steps
    assert len(dec_steps) == 4, len(dec_steps)
    # Budget never exceeded.
    assert all(r["total_query_tokens"] <= cfg.max_num_batched_tokens for r in recs)

    # CONC cap respected with many requests.
    cfg2 = Config(num_prompts=50, ISL=8, OSL=8, range_ratio=1.0,
                  max_num_batched_tokens=100000, CONC=4, kv_cache_tokens=10_000)
    recs2 = simulate(cfg2, build_workload(cfg2))
    assert all(r["batch_size"] <= cfg2.CONC for r in recs2), \
        max(r["batch_size"] for r in recs2)

    # KV budget gates admission (no overflow).
    cfg3 = Config(num_prompts=20, ISL=100, OSL=100, range_ratio=1.0,
                  max_num_batched_tokens=100000, CONC=100, kv_cache_tokens=600)
    recs3 = simulate(cfg3, build_workload(cfg3))
    assert all(r["kv_tokens_in_use"] <= cfg3.kv_cache_tokens for r in recs3)
    print("selftest OK")


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    for f in cfg.__dataclass_fields__:
        if hasattr(args, f) and getattr(args, f) is not None:
            setattr(cfg, f, getattr(args, f))
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-prompts", dest="num_prompts", type=int)
    p.add_argument("--request-rate", dest="request_rate", type=float)
    p.add_argument("--burstiness", dest="burstiness", type=float)
    p.add_argument("--isl", dest="ISL", type=int, help="max input length")
    p.add_argument("--osl", dest="OSL", type=int, help="max output length")
    p.add_argument("--range-ratio", dest="range_ratio", type=float)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--workload-file", dest="workload_file", type=str)
    p.add_argument("--conc", dest="CONC", type=int, help="max concurrent requests")
    p.add_argument("--max-num-batched-tokens", dest="max_num_batched_tokens", type=int)
    p.add_argument("--long-prefill-token-threshold", dest="long_prefill_token_threshold", type=int)
    p.add_argument("--max-model-len", dest="max_model_len", type=int)
    p.add_argument("--kv-cache-tokens", dest="kv_cache_tokens", type=int)
    p.add_argument("--out", default="sim_steps.json", help="output JSON path")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    cfg = _build_config(args)
    requests = build_workload(cfg)
    records = simulate(cfg, requests)

    with open(args.out, "w") as f:
        json.dump({"config": cfg.__dict__, "steps": records}, f)

    n_steps = len(records)
    if n_steps:
        avg_bs = sum(r["batch_size"] for r in records) / n_steps
        max_bs = max(r["batch_size"] for r in records)
        print(f"{len(requests)} requests -> {n_steps} steps; "
              f"avg batch {avg_bs:.1f}, max batch {max_bs}; wrote {args.out}")
    else:
        print("no steps produced")


if __name__ == "__main__":
    main()
