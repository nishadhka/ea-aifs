# Icechunk commit cadence — per-member vs per-72h vs per-6h (evaluated & scoped)

**Question (scoped before running):** the present Path A design makes **one Icechunk
commit per member** after its rollout (see `ICECHUNK_PATH_A.md` §2.2/§2.4 and the bring-up
transcript `../2026-06-28-aifs-ensv2-infere-gpu-icechunk-plan.txt`). Can the store instead
be committed **every 72 h**, or **every 6 h** (each model step) if wanted? Is that a good
idea, and what has to change?

**Short answer.** Yes — committing every 72 h or 6 h is feasible and cheap **on its own**.
But commit *frequency* is the easy half. The half that actually matters is the **time-axis
chunk shape**, and the currently-specified chunking (`(1, n_steps, values)` — the whole time
axis in one chunk, `ICECHUNK_PATH_A.md` §1.1) is **the wrong shape for a step-by-step writer**
and causes large write amplification regardless of commit frequency. Fixing the chunking is
what unlocks cheap fine-grained commits.

---

## 1. The finding: chunk shape governs write cost, not commit count

The rollout writes **one timestep at a time** (`root[var][m, n, :] = value` for
`n = 0..n_steps-1`, per `IcechunkMemberWriter.write_step` / anemoi's `ZarrOutput`). A Zarr
**chunk is the atomic I/O unit** — writing a single-step sub-region of a chunk that spans
many steps forces a **full read-modify-write of the entire chunk on every step**. So a chunk
holding all `n_steps` is rewritten `n_steps` times per member.

Measured on this box (local Icechunk store, `icechunk 2.1.0`, N320 width `V=542080`,
`S=24` steps, `NV=3` vars, `f4`, incompressible random data → 156 MB logical; readback
verified on the last step in every case):

| Config | time-chunk | commit every | #commits | store on disk | objects | wall | **store / logical** |
|---|---|---|---|---|---|---|---|
| **P1** per-member, 1 big chunk *(current design)* | 24 steps | 24 (once) | 1 | **1804 MB** | 84 | 16.0 s | **11.6×** |
| **P2** per-72h, chunk aligned | 12 steps | 12 | 2 | 938 MB | 90 | 7.7 s | 6.0× |
| **P3** per-6h, chunk aligned | 1 step | 1 | 24 | **144 MB** | 222 | 1.3 s | **0.92×** |
| **P4** per-72h, chunk **mis**aligned | 24 steps | 12 | 2 | 1804 MB | 90 | 14.3 s | 11.6× |

Read this carefully:

- **Amplification tracks the time-chunk span, ≈ `time_chunk / 2`** (24→11.6×, 12→6.0×,
  1→~1×). It is **independent of how often you commit**: P1 (1 commit) and P4 (2 commits)
  produce the *identical* 1804 MB because both use the big all-time chunk.
- **Finer time-chunks are smaller *and* faster** (P3: 0.92×, 1.3 s vs P1: 11.6×, 16 s) —
  the opposite of the intuition that "more commits = more overhead."
- The current `(1, n_steps, values)` design (P1) already pays **11.6×** at just 24 steps.

### At production scale this is severe
The real rollout is **`n_steps = 160`** (960 h / 6 h). A `(1, 160, values)` chunk is
`160 × 542080 × 4 ≈ 347 MB` **per variable**, rewritten 160 times → tens-of-× amplification
per member. Per member the logical store is ~`160 × 542080 × ~120 vars × 4 B ≈ 42 GB` (`f4`)
or ~21 GB (`f2`); at ~tens-of-× that is **hundreds of GB–TB of writes per member**. On the
**GCS** target that is real egress/ingest cost and wall-clock, not just transient local disk
(which Icechunk GC would later reclaim, but the *I/O* is already spent).

---

## 2. The two clean designs (and how commit cadence falls out)

To write step-by-step **without** amplification, the time-chunk must be **small** (≤ the
amount you write between flushes). Two ways to get there:

### Design S — small time-chunks, stream + commit at cadence
Chunk the time axis to the commit window and write/commit per window:

| Want | time-chunk | write pattern | commit | amplification | objects/var/member |
|---|---|---|---|---|---|
| **commit every 6 h** | `(1, 1, V)` | write each step, commit each step | per step (160×) | ~1.0× | 160 |
| **commit every 72 h** | `(1, 12, V)` | write 12 steps, commit the window | per 72 h (13×) | ~6× *(or ~1× if you buffer — below)* | 13 |

### Design S+ — buffer the window in RAM, write the whole chunk once (best for 72 h)
For 72 h commits, **buffer one 72 h window (12 steps) in RAM**, then write the whole
`(1, 12, V)` chunk in a *single* assignment and commit. One write per chunk → **~1×
amplification**, durable every 72 h, only ~13 chunks/var/member. Buffer cost is trivial:
`12 × 542080 × ~120 × 4 ≈ 3.1 GB` RAM (this box has 220 GB). **This mirrors the current GRIB
cadence exactly** (one 72 h GRIB file ↔ one 72 h Icechunk commit) and is the recommended
target if you want intra-member durability.

### Design B — keep big chunks, but buffer the *whole member* and write once
If you specifically want the big `(1, n_steps, V)` chunk (fewest objects, best bulk-read),
you must **not** write it step-wise. Buffer the member's full array (~21–42 GB RAM) and write
each variable's chunk in one shot, then a single per-member commit. ~1× amplification, but
**no intra-member commits** — you only have data at member end. This is the amplification-free
version of the *current* per-member design.

> **Commit frequency itself is cheap** — extra commits add only small snapshot/manifest
> metadata (P1 vs P4 identical chunk bytes). Its real costs are (a) a longer snapshot history
> to garbage-collect (`expire_snapshots` + `garbage_collect`, already flagged in
> `ICECHUNK_PATH_A.md` §7) and (b) each commit is a durability/visibility point for readers.

---

## 3. Recommendation

1. **Do not write the `(1, n_steps, values)` chunk step-by-step.** That is the one
   combination to avoid (it is what §1.1 currently specifies + how the writer writes).
2. **If you want per-72 h commits (recommended, matches the GRIB cadence):** Design **S+** —
   chunk `(1, 12, values)`, buffer each 72 h window, write the window chunk once, commit.
   ~1× amplification, durable every 72 h, ~13 chunks/var/member.
3. **If you want per-6 h commits:** Design **S** — chunk `(1, 1, values)`, write+commit each
   step. ~1× amplification, finest durability, most objects (160/var/member).
4. **If you want the current per-member commit only:** Design **B** — keep big chunks but
   buffer the whole member and write once (not step-wise).
5. Whatever the cadence, schedule Icechunk **GC** (`expire_snapshots` + `garbage_collect`)
   so history/objects don't grow unbounded.

Downstream is unaffected: `xr.open_zarr(session.store)` reads **any** chunking transparently,
so the N320→1.5° regrid and AI-WQ steps (`ICECHUNK_PATH_A.md` §4) don't care which cadence
was used — this is purely a write-path decision. (Larger time-chunks read marginally faster
for the "all steps of tp/msl/2t" access pattern, which is another reason to prefer the 72 h
window over per-6 h if reads dominate.)

---

## 4. Implementation delta (small)

`icechunk_output.py` (the local writer already added this session) and its Path A/GCS twin
need two parameters:

- **`init_schema(..., time_chunk=…)`** — replace the hard-coded `(1, n_steps, n_values)` chunk
  with `(1, time_chunk, n_values)` (`time_chunk = 1` for 6 h, `12` for 72 h, `n_steps` for
  per-member).
- **`IcechunkMemberWriter(..., commit_every=…, buffer=True)`** — accumulate `commit_every`
  steps, write the whole window slab in one assignment (avoids amplification), then
  `session.commit()` and reopen the session for the next window.

`run_local_icechunk_v2.py` gains `--commit-every {1,12,160}` (6 h / 72 h / per-member) and
passes `time_chunk` through to `init_schema`. The runner loop is otherwise unchanged. This is
a ~30-line change; the GRIB path stays the untouched fallback.

*Evidence: `/tank/projects` probe, 2026-07-06, `icechunk 2.1.0` / `zarr 3.2.1`, RTX 5000 Ada
box. Numbers reproduce with the 4-config probe in this folder's scratch.*
