# Writing N320 inference straight to a source.coop Icechunk store

Attempt to run AIFS-ENS-2.0 for cycle **20260212** and write the forecast into
`s3://e4drr-project/forecasts/fp16FahamuAIFSv2_n320/20260212` (Icechunk over
`https://data.source.coop`), instead of a local store as in `LOCAL_GPU_RUN.md`.

**Halted before the production run.** The path works — a member was written and read
back correctly — but at the default commit cadence the *store write* costs ~12x the
inference it is storing, which is the wrong shape for a 50-member ensemble. Halting was
the right call at the numbers that were visible at the time; the measurements below then
found where the cost actually comes from, and it is fixable. Nothing was left in the
target prefix: it is empty.

Measured 2026-08-31 on an L4 (22 GB, driver 570 / CUDA 12.8), Python 3.12 env built by
`setup_gpu_env_v2.sh --with-icechunk`, against the real 926 MiB member-001 pkl from
`gs://ea_aifs_w1/20260212_0000/input_v2_pre50r1/`.

---

## 1. The shape of the problem

One stored step is **120 variables x 542,080 N320 cells x 4 B = 248 MiB**, written as
120 *independent* single-chunk PUTs (one per variable — that is what the
`(member, time, values)` schema chunked `(1, time_chunk, n_values)` means).

Inference produces that step in **4.4 s** (FP16, 16 chunks — the figure from
`LOAD_TEST_RESULTS.md` §3, re-confirmed here). So the question is only ever: how fast
can 248 MiB leave the box as ~120 objects?

| Config | s/step | of which inference | of which store |
|---|---:|---:|---:|
| First working version (sequential writes, commit every step) | **215** | 4.4 | ~211 |
| + 16 concurrent PUTs, commit every step (`--commit-every 1`) | **54.1** | 4.4 | ~50 |
| + `--commit-every 4` (chunk = 4 steps) | **19.3** | 4.4 | ~15 |
| + `--commit-every 12` (chunk = 12 steps = 72 h) | **10.0** | 4.4 | ~5.6 |

Each row is an end-to-end run of member 001 into a throwaway prefix (4, 4, 8 and 12
steps respectively), not a projection.

## 2. Why: the store is latency-bound, not bandwidth-bound

Two independent measurements, both against `data.source.coop` with the same credential.

**Concurrency** — 32 x 2.1 MiB objects, varying threads:

| threads | throughput |
|---:|---:|
| 1 | 1.69 MiB/s |
| 8 | 12.85 MiB/s |
| 24 | 16.49 MiB/s |

**Object size** — 40 objects at 16 threads, varying bytes per object:

| time_chunk | object size | throughput |
|---:|---:|---:|
| 1 | 2.1 MiB | 18.79 MiB/s |
| 4 | 8.3 MiB | 49.56 MiB/s |
| 12 | 24.8 MiB | 90.11 MiB/s |

Throughput tracks *requests per second*, not bytes per second: a single stream moves
1.69 MiB/s regardless, and 12x-bigger objects go 4.8x faster. The default schema
(`time_chunk = commit_every = 1`) is therefore the worst possible shape for a remote
store — it maximises the request count for a given payload.

This is not the same trade-off `ICECHUNK_COMMIT_CADENCE.md` documents. That warns
against `time_chunk > commit_every`, which read-modify-writes a chunk once per step.
Here the two stay **equal** (the writer aligns them), so each chunk is still written
exactly once; only its size changes.

## 3. What it costs at 50 members

A 960 h rollout is 160 steps, so the full N320 corpus is
`50 x 160 x 248 MiB = 2.08 TB`. Pure inference for that is `50 x 160 x 4.4 s = 9.8 h`
— the floor, and unavoidable.

| Store config | Wall clock, 50 members | vs the 9.8 h inference floor |
|---|---:|---:|
| commit every step (the default) | ~120 h | **12.3x** |
| `--commit-every 4` | ~43 h | 4.4x |
| `--commit-every 12` | **~22 h** | 2.2x |

At the default the machine spends 92 % of five days uploading. At `--commit-every 12`
it spends 55 % of one day — the same corpus, the same GPU work, one flag.

## 4. Options, in the order worth trying

1. **`--commit-every 12` (chunk = 72 h).** Measured 5.4x faster end to end than the
   default; no change to what is stored. Costs commit granularity: a crash loses up to
   72 h of that member's steps (it resumes from the last commit, and `--skip-existing`
   still works at member level), and a reader wanting one timestep of one variable
   fetches a 24.8 MiB chunk. **This alone brings the full ensemble to ~22 h.**
2. **`--write-hours 432-792`** — store only the window
   `shared/aifs_n320_grib_1p5defg_nc_cli.py` actually reads (61 of 160 steps, ~790 GB).
   Every step is still computed, so the 9.8 h inference floor is unchanged, but ~62 %
   of the upload disappears. Caveat: with a window, chunks at the window edge are
   partial and fall back to the slower per-step write path, so the gain is slightly
   below 62 %.
3. **`--grid o96`** — 40,320 cells instead of 542,080, 13.4x less data (~155 GB for the
   full corpus), already implemented and bit-identical to earthkit
   (`O96-icechunk-store/README.md`). This contradicts the `_n320` in the target path,
   so it is a different product, not a tuning knob — but if the consumer is the AI-WQ
   1.5 deg pipeline, it is the honest answer, optionally with `--native-store` keeping
   `msl,tp,2t` on native N320 as tier B.
4. **`--float-size f2`** — halves the bytes. Lossy; only for the bulk corpus, never for
   the submitted variables.
5. **More upload concurrency** (`--write-threads 24`) — worth ~10 % on top of the
   object-size win, and the returns are clearly flattening by 24. Not the main lever.

## 5. Two defects found on the way (both fixed, in this branch)

### a. source.coop returns intermittent empty-body errors on chunk PUTs

```
icechunk.StorageError: object store error service error: unhandled error:
  error parsing XML: no root element
  0: icechunk::asset_manager::write_chunk
  1: icechunk::store::set with key="v_850/c/0/0/0"
```

The first full run died on it mid-member. Reproduced with **no AIFS code involved** — a
loop of plain 2.1 MiB chunk writes failed on the 34th — and it is *flaky, not a hard
rejection*: an identical 40-write loop immediately afterwards passed clean, and 60
sequential `aws s3 cp` PUTs of the same size never failed. Icechunk does not retry it
(its own retry policy treats the parse failure as non-retryable, so it surfaces
immediately rather than after a backoff).

**Fix:** `retry()` in `icechunk_output.py`, wrapping each chunk write, each commit, and
the schema init, with exponential backoff. Across the runs here it fired 1-3 times per
run and cleared on the **first** retry every time. Default `write_retries=1` (a plain
call) keeps the local-filesystem path failing fast, where a write error is real.

### b. Per-variable writes were sequential

A step's 120 PUTs are independent — different arrays, never the same chunk — but were
issued one at a time, capping a step at one stream's 1.69 MiB/s.

**Fix:** `write_threads` on `IcechunkMemberWriter`, defaulting to 1 (local disk gains
nothing) and to 16 from the S3 runner.

## 6. What is on this branch

| File | Change |
|---|---|
| `run_s3_icechunk_v2.py` | **New.** S3/source.coop counterpart of `run_local_icechunk_v2.py`. GCS input (`ea_aifs_w1` / `input_v2_pre50r1` / `coiled-data-e4drr_202505.json`) -> FP16 inference -> Icechunk on source.coop. Refreshable STS credentials, credential-life preflight, `--write-threads` / `--write-retries`. |
| `run_local_icechunk_v2.py` | `repo_factory` hook so the S3 runner reuses the rollout loop instead of copying it; `write_retries` / `write_threads` passthrough. Defaults unchanged — the local path behaves exactly as before. |
| `icechunk_output.py` | `retry()` helper; `write_retries` and `write_threads` on `IcechunkMemberWriter`. Both default to the previous behaviour. |
| `.gitignore` (root + v2) | `coiled-data.json` -> `coiled-data*.json`, see §8. |

Credentials are **not** committed: the source.coop STS token lives in
`fp16FahamuAIFSv2/.env` (mode 600, matched by the existing `*.env` rule).

## 7. Reproducing

```bash
# env (Python >= 3.11 — icechunk publishes no cp310 wheel)
./setup_gpu_env_v2.sh --prefix /scratch/venv-aifs-v2-py312 --python 3.12 --with-icechunk
export HF_HOME=/scratch/hf_cache CC=/usr/bin/gcc
source-coop login --duration 12h --port 39427    # see run-pre50r1-dates/SOURCE_COOP_CREDENTIALS.md

# one member, 72 h, into a throwaway prefix -- the 10.0 s/step row of §1
python run_s3_icechunk_v2.py --date 20260212_0000 --members 1 --n-members 1 \
    --lead-time 72 --commit-every 12 \
    --store-prefix forecasts/fp16FahamuAIFSv2_n320/_smoke --no-tag
```

Verified readback (`xarray.open_zarr` on the store): `member=1, time=4, values=542080`,
122 arrays, `2t` 222-310 K, `msl` 947-1042 hPa, `tp` >= 0, wave fields 67 % finite
(NaN over land, as expected from the LSM mask).

**The cycle date is 20260212, not 20260725.** The pkls are at
`gs://ea_aifs_w1/20260212_0000/input_v2_pre50r1/` (50 members, 926 MiB each) and the
bucket holds no 20260725 prefix.

## 8. Unrelated, but found here: an unignored service-account key

`fp16FahamuAIFSv2/coiled-data-e4drr_202505.json` was untracked **and unignored** — one
`git add -A` from being committed. The rules only covered the exact name
`coiled-data.json`, which is precisely the escape `SOURCE_COOP_CREDENTIALS.md` §7 and
commit `5928ec1` both record as having happened before *with this very filename*.
Widened to `coiled-data*.json` in both `.gitignore` files. The key itself is not
committed here.
