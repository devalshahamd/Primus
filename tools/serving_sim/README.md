# serving_sim

A small token-step simulator for LLM serving request packing. It replays a
workload of requests through a continuous-batching scheduler (vLLM V1
unified-batch policy) and records, for **every execution step**, which requests
are packed together and their attention shapes. The goal is to study the
**distribution of requests packed together at each step** (prefill vs decode
mix, batch size, query/KV lengths, token-budget utilization).

This is a starting point: logic is deliberately minimal. Knobs that are fixed or
deferred are catalogued in [Further_improvements.md](Further_improvements.md).

## What "step" means

Each step is one forward pass of the model. There is no wall-clock / latency
model: a prefill chunk and a decode each advance a request by the tokens it
computes that step. This is enough to reproduce batch composition.

## How vLLM V1 packs requests (what we model)

vLLM V1 has **no separate prefill/decode phases**. Each request tracks
`num_computed_tokens` vs `num_tokens` (prompt + generated tokens). Every step the
scheduler assigns tokens so each request catches up. One batch can therefore
contain prefill chunks and 1-token decodes together.

Per step the simulator:

1. Admits arrivals due at this step into a FCFS waiting queue.
2. **Phase 1** - schedules already-running requests in order, each getting
   `num_new_tokens = num_tokens - num_computed_tokens`, capped by the per-step
   token budget (`max_num_batched_tokens`), `long_prefill_token_threshold`, and
   `max_model_len`. Chunked prefill emerges naturally from the budget cap.
3. **Phase 2** - admits new waiting requests while `len(running) < CONC` and
   budget remains. A request is admitted only if its full sequence
   (`prompt + output`) fits in the remaining KV pool (full-ISL reservation).
4. Applies the forward pass: advances `num_computed_tokens`; when a request's
   computation catches up to its known tokens, the model emits one new token
   (so the **first output token is sampled at the end of prefill**, then decode
   steps follow).
5. Retires finished requests and frees their KV reservation.

## Knobs (`Config` in `simulator.py`)

Workload:
- `num_prompts` - number of requests.
- `request_rate` (req/s, `inf` = all at step 0), `burstiness` (gamma shape; 1 = Poisson).
- `ISL` / `OSL` - **maximum** input/output length; actual length sampled uniformly
  from `[range_ratio*ISL, ISL]` and `[range_ratio*OSL, OSL]`.
- `range_ratio`, `seed`.
- `workload_file` - load `(arrival, isl, osl)` from JSON/CSV instead of sampling;
  the `arrival` column is the execution step id at which the request enters.

Capacity / admission:
- `CONC` - max concurrent requests in the system (InferenceX `--max-concurrency`).
- `max_num_batched_tokens` - per-step token budget across the whole batch.
- `long_prefill_token_threshold` - per-request prefill chunk cap (0 = off).
- `max_model_len` - per-request context cap (prompt + output).

KV cache:
- `kv_cache_tokens` - total KV token-slot pool (flat budget, no block granularity).

Fixed in v1: `enable_chunked_prefill=True`, `scheduler_reserve_full_isl=True`,
FCFS queue policy. See [Further_improvements.md](Further_improvements.md).

## Usage

```bash
# Synthetic saturation run (all requests at step 0)
python3 simulator.py --num-prompts 64 --isl 512 --osl 64 \
    --conc 8 --max-num-batched-tokens 2048 --out sim_steps.json

# From a workload file (CSV columns: arrival,isl,osl  -- or JSON list of dicts)
python3 simulator.py --workload-file my_workload.csv --conc 16 --out sim_steps.json

# Invariant self-check
python3 simulator.py --selftest
```

## Output

A single JSON: `{"config": {...}, "steps": [ ... ]}`. Each step record:

```json
{
  "step": 1,
  "batch_size": 8,
  "num_prefill_reqs": 4,
  "num_decode_reqs": 4,
  "total_query_tokens": 2048,
  "kv_tokens_in_use": 4608,
  "requests": [
    {"request_id": 0, "phase": "decode",  "query_len": 1,    "kv_len": 512},
    {"request_id": 5, "phase": "prefill", "query_len": 512,  "kv_len": 0}
  ]
}
```

`query_len` is the number of tokens the request computes this step (prefill chunk
size, or 1 for decode); `kv_len` is the context length it attends to
(`num_computed_tokens` at the start of the step). Together these are the
attention q/kv shapes per request per step. Build any distribution you want by
aggregating over `steps[*].requests`.

## SGLang vs vLLM (why we picked vLLM first)

| Dimension | SGLang (default) | vLLM V1 (modeled here) |
|---|---|---|
| Step composition | Phase-based: a step is all-prefill or all-decode | Unified: prefill chunks + decodes mix in one step |
| Prioritization | Prefill-prioritized | Running first, then new admissions |
| Per-step token budget | `max_prefill_tokens` + `chunked_prefill_size` | `max_num_batched_tokens` |
| Concurrency cap | `max_running_requests` | `max_num_seqs` (here: `CONC`) |
| KV-OOM handling | Decode retraction (re-prefill later) | Preemption (not triggered under full-ISL reservation) |
| Queue policy | fcfs / lpm / lof / dfs-weight | fcfs / priority |

SGLang's phase-based scheduling and the other deferred behaviors are described in
[Further_improvements.md](Further_improvements.md).
