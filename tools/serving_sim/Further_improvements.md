# Further improvements

This starting-point simulator fixes several knobs to a single value and omits
others to stay small. This file records every such decision, the logic that the
deferred option would require, and the SGLang scheduling nuances that a future
version could model.

## Fixed-default knobs in v1

### `enable_chunked_prefill = True` (fixed)
We always allow a prompt to be split across steps: a request gets
`min(remaining, token_budget, ...)` tokens per step. To support
`enable_chunked_prefill = False`:
- A waiting request whose remaining prompt exceeds the remaining token budget
  must **not** be partially scheduled. In vLLM the waiting loop simply `break`s
  (head-of-line blocking) - it does not skip to the next waiter. Implement by,
  in Phase 2, breaking out of the admission loop when
  `prompt_len > token_budget` instead of clamping `query_len`.

### `scheduler_reserve_full_isl = True` (fixed)
A request is admitted only if its whole sequence (`prompt + output`) fits in the
remaining KV pool, and that full amount is reserved up front. Consequently a
running request can never hit a KV shortage, so **preemption never triggers**.
To support `scheduler_reserve_full_isl = False`:
- Admit on incremental availability (only enough KV for the next chunk), letting
  KV grow as tokens are computed.
- Add a per-step KV check for running requests; on shortage, preempt (see FCFS
  preemption below). This produces more aggressive packing but introduces
  preemption/recompute dynamics.

### FCFS queue policy (fixed)
The waiting queue is strict first-come-first-served. To add a `priority` policy:
- Replace the FCFS `waiting` list with a heap keyed on
  `(priority, arrival_step, rid)` (vLLM `PriorityRequestQueue`).
- On KV-OOM preemption, evict the lowest-priority running request
  (`max((priority, arrival_time))`) rather than the most-recently-added one.

### FCFS preemption (not implemented, because reserve_full_isl=True)
Under full-ISL reservation there is no running-time KV shortage, so there is
nothing to preempt. When `scheduler_reserve_full_isl = False` is added, model
vLLM preemption: on `allocate_slots` failure for a running request, pop the
**last** running request (FCFS), reset its `num_computed_tokens = 0`, free its
KV, and re-queue it at the front of `waiting`. Any preemption in a step also
skips that step's new-admission (Phase 2).

## Removed / simplified knobs

### `CONC` vs `max_num_seqs`
We expose a single `CONC` that caps concurrent requests in the system. In a real
deployment these are two distinct limits:
- **CONC** = InferenceX `--max-concurrency`, a *client-side* cap on in-flight
  requests (an `asyncio.Semaphore` in `benchmark_serving.py`). Requests beyond
  CONC are never sent.
- **`max_num_seqs`** = the *server-side* scheduler cap on RUNNING sequences.
In InferenceX sweeps the server is configured with `max_num_seqs >= conc`, so
CONC binds. To model both, add a separate `max_num_seqs` and an admission window
that only lets CONC requests exist in `waiting + running` at once.

### Dropped `max_num_scheduled_tokens` alias
vLLM has both `max_num_batched_tokens` (the budget) and `max_num_scheduled_tokens`
(which defaults to it). They only differ when speculative decoding reserves room
(`max_num_batched_tokens - num_speculative_tokens * max_num_seqs`). We keep only
`max_num_batched_tokens`. Reintroduce the split when spec decode is added.

### Prefix caching (omitted; assume default off)
Requests start with `num_computed_tokens = 0`. To model prefix reuse (radix cache
in SGLang, block-hash prefix cache in vLLM), reduce `num_computed_tokens` at
admission by the number of cached prompt tokens (e.g. a shared system prompt),
which shortens prefill.

### Speculative decoding (`num_speculative_tokens`) (removed)
Each decode would propose K draft tokens (`num_tokens_with_spec`), changing
`query_len` per decode step and the effective budget, plus an acceptance model to
decide how many are kept. Out of scope for v1.

### Block/page-granular KV (removed `block_size`)
KV is a flat token budget (`kv_cache_tokens`). Real engines allocate in pages
(e.g. 16 tokens), which rounds each request's footprint up to a block multiple
and affects fragmentation/admission. Add `block_size` and
`ceil(tokens / block_size) * block_size` accounting to model this.

### Output recording knobs (removed)
We always store everything (every step, every scheduled request's
`query_len`/`kv_len`/`phase`) to one JSON. If files get large, add sampling or a
columnar format.

### Arrival-to-step mapping (simplified)
For synthetic finite `request_rate`, gamma inter-arrival times (in time units)
are floored to integer step indices (1 time-unit == 1 step). This is a rough
mapping since steps have no real duration. For file workloads the `arrival`
column is taken directly as the step id. A faithful mapping would require a
step-duration / latency model (explicitly out of scope - this is a token-step
simulator).

## SGLang scheduling nuances (for an alternative scheduler backend)

The current scheduler models vLLM V1. SGLang differs substantially and would be a
second `simulate`-style function:

- **Phase-based, prefill-prioritized.** Each step runs *either* a prefill
  (`EXTEND`) batch *or* a decode (`DECODE`) batch, not both. If any prefill can be
  admitted, the step is prefill-only; decode runs only when no prefill batch
  forms. Batch composition is therefore homogeneous per step (a separate
  prefill-batch-size distribution and decode-batch-size distribution, plus the
  interleaving pattern between them).
- **Mixed chunk mode.** Only when `chunked_prefill_size` is set *and*
  `enable_mixed_chunk = True` does SGLang produce a `ForwardMode.MIXED` step that
  mixes prefill tokens with one decode token per running request.
- **Token budgets.** `max_prefill_tokens` (default 16384) bounds new input tokens
  per prefill batch; `chunked_prefill_size` is the rolling chunk budget
  (`-1` disables chunking). There is no `max_num_batched_tokens` in the scheduler.
- **Decode-headroom reservation.** Admission reserves KV for estimated future
  decode via `new_token_ratio * max_new_tokens` (conservative); SGLang adjusts
  `new_token_ratio` adaptively after retractions.
- **Decode retraction on KV-OOM** (instead of vLLM-style preemption): sort running
  requests by most-decoded-first, release their KV, and re-queue them for a
  **full re-prefill**.
- **Cache-aware queue policies.** Besides `fcfs`, SGLang offers `lpm`
  (longest-prefix-match against the radix cache), `lof` (longest-output-first),
  `random`, and `dfs-weight`; LPM falls back to FCFS when the queue exceeds 128.
- **`prefill_max_requests`** hard-caps the number of requests in one prefill batch.
- **Priority scheduling** with preemption (`enable_priority_scheduling`,
  `priority_scheduling_preemption_threshold`).
