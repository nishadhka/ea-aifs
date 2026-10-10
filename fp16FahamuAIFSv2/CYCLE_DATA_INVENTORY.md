# What data exists for each cycle — stores, grids, variables

Surveyed 2026-10-10 by reading every Icechunk store under `/tank/projects/aifs-run`.
This is what is **actually on disk**, not what each cycle produced: most older cycles have
been purged down to their small products.

---

## The two grids, and why there are two

Every rollout is computed on **N320** and then written at one or both of these:

| store | grid | points | resolution | holds |
|---|---|---|---|---|
| `icechunk_o96` | O96 octahedral reduced Gaussian | **40,320** | ~112 km | **120 data variables**, usually all 132 steps (6–792 h) |
| `icechunk_n320_aiwq` | N320 reduced Gaussian | **542,080** | **~28 km** | **10 data variables**, 61 steps (432–792 h) |
| `icechunk_v2` | N320 reduced Gaussian | 542,080 | ~28 km | 120 variables — the pre-tier-B full store, ~583 GB. None left on disk. |

Each store also carries `latitude`, `longitude`, `member`, `time`, so an array count of
124 or 14 in older notes is the same thing as 120 or 10 data variables here.

**N320 is 13.4× more points than O96** (542,080 / 40,320), which is the whole reason for the
split: a full-N320 corpus is ~583 GB per cycle against ~209 GB for O96-plus-sidecar.

### What each target reads, and why

| target | store | grid | reason |
|---|---|---|---|
| `tas` / `mslp` / `pr` quintiles | **N320 sidecar** | ~28 km | submitted product; 3a regrids 432–792 h to 1.5° |
| TS days | **N320 sidecar** | ~28 km | the wind maximum is resolution-sensitive; O96 smooths it and yields ~2× more tracks but fewer storm-days |
| MJO phases | **O96 corpus** | ~112 km | zonal wavenumber 1–3, so 112 km is ample — **and the sidecar has no `u_200`/`v_200`** |

That pair is the mirror image of each other, and it is why no single store serves both.

### The sidecar's 10 variables

```
msl  tp  2t          the three submitted AI-WQ variables
10u  10v             surface wind for the TS tracker
t_200 t_300 t_500    the TS warm-core test (core vs annulus, 200-500 hPa)
u_850 v_850          low-level wind: TS, and the MJO U850 band
```

`u_200`/`v_200` are **not** included. They are available opt-in via `--native-vars-mjo`
(`DOWNSTREAM_VARS_MJO`) at +13.2 GB/cycle, which is deliberately not the default: MJO works
fine from O96, and storage is the binding constraint here.

---

## Per-cycle inventory

`steps` is written steps out of 132; `6-792h` means the complete corpus, `432-792h` the
downstream window only.

| cycle | total | O96 (120 var, ~112 km) | N320 (~28 km) | pkls | products |
|---|---|---|---|---|---|
| 20260212 | 49 M | — | — | — | — |
| 20260514 | 96 G | 103 G · 132 steps | — | — | — |
| 20260604 | 54 G | **9.1 G** · 132 steps | — | 50 | — |
| 20260625 | 0 | — | — | — | — |
| 20260702 | 0 | — | — | — | — |
| 20260709 | 141 G | 104 G · 132 steps | — | 50 | — |
| 20260716 … 20260806 | 13 M each | — | — | — | — |
| 20260813 | 68 G | 45 G · **61 steps** | 12 G · **3 vars** | — | — |
| 20260820 | 99 G | 102 G · 132 steps (+2.2 G `_nwh`) | 1.1 G `_nwh` · 10 var | — | TS ×2 |
| 20260827 | 13 M | — | — | — | TS |
| **20260903** | 256 G | 158 G · 132 steps | 51 G · 10 var · 61 steps | 50 | TS, MJO |
| **20260910** | 260 G | 162 G · 132 steps | 52 G · 10 var · 61 steps | 50 | TS, MJO |
| **20260917** | 259 G | 164 G · 132 steps | 53 G · 10 var · 61 steps | 50 | TS, MJO |
| **20260924** | 256 G | 157 G · 132 steps | 51 G · 10 var · 61 steps | 50 | TS, MJO |
| **20261001** | 256 G | 158 G · 132 steps | 51 G · 10 var · 61 steps | 50 | TS, MJO |
| 20261008 | 46 G | — | — | **50** | — (pkl only; Step 2 awaiting disk) |

Three anomalies worth knowing before trusting a row:

- **`20260604`'s O96 store is 9.1 GB** against ~158 GB for a full 50-member cycle, while
  reporting 132 steps and 120 variables. That is a partial or few-member store, not a
  complete corpus. Do not treat it as a usable cycle without checking its member count.
- **`20260813`'s N320 sidecar has only 3 variables**, so the TS tracker — which needs
  `10u/10v/msl/t_200/t_300/t_500/u_850/v_850` — **can never run on it**. Its O96 store is
  also only 61 steps, so days 8 and 15 are missing and MJO's `(9,4)` cannot be built either.
- **`20260820` has `_nwh` duplicates** (`icechunk_o96_nwh`, `icechunk_n320_nwh`) from the
  `--native-write-hours` test, 3.3 GB together. They are test artefacts, not products.

### A purge deadlock in `20260813`

`cleanup_aifs_run.py` refuses to purge a cycle whose N320 store is present but whose
`ts_days_probs_*.nc` is not — correctly, since the store is the only source of the fields the
tracker reads. But `20260813`'s sidecar has **3 variables**, so that product can never be
produced, and the guard will therefore refuse that cycle **forever**.

68 GB is held by a rule that is right in general and unsatisfiable here. Purging it needs a
deliberate override, not a fix to the guard.

---

## The pkl staging convention — symlinks, not renames

Step 1 writes `proto_input_state_member_NNN.pkl`; `run_local_icechunk_v2.py` hardcodes
`input_state_member_NNN.pkl` (line 125) with no prefix option. The established practice,
visible on `20260604`/`20260709`/`20260903`/`20260910`/`20260917`, is a **symlink**:

```
input_state_member_001.pkl -> proto_input_state_member_001.pkl
```

That is better than renaming, and the reason is `--skip-existing`: it matches on the
`proto_` name, so after a rename a Step 1 resume re-fetches all 50 members (~1 h 50 m of
wasted work), while with a symlink both names resolve and a resume skips correctly.

```bash
cd $BASE/input_states
for i in $(seq 1 50); do
  f=$(printf "input_state_member_%03d.pkl" $i)
  if [ -f "$f" ] && [ ! -L "$f" ]; then mv "$f" "proto_$f" && ln -s "proto_$f" "$f"; fi
done
```

`20260924` and `20261001` were renamed rather than symlinked; harmless there because both
cycles are complete and submitted. `20261008` has been converted back, since its Step 2 is
still pending and a resume is plausible.

---

## Storage arithmetic

| item | size |
|---|---|
| input pkls, 50 members | **42–43 GB** (0.97 GB each) |
| O96 corpus, 120 var × 132 steps × 50 members | **157–164 GB** |
| N320 sidecar, 10 var × 61 steps × 50 members | **51–53 GB** |
| 3a NetCDF, 1.5° | ~2 GB |
| **a full tier-B cycle** | **~256 GB** |
| full-N320 alternative (`icechunk_v2`) | ~583 GB |
| TS product | ~8.5 KB |
| MJO product | ~49 KB |
| MJO band series (per cycle, kept outside `aifs-run`) | ~6 MB |

**The products are ~10⁻⁵ of the store they come from.** That asymmetry is the retention rule:
extract TS and MJO before purging, because the store is the only source and re-creating it
costs a GPU day.

Band series for nine cycles live in `/tank/projects/mjo_model_clim/` (41 MB, 2026-05-14 →
2026-10-20), deliberately **outside `aifs-run`** so `cleanup_aifs_run.py` cannot reach them.

---

## Related

- [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md) — why TS needs 28 km.
- [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md) — why MJO does not, and §5 on the model
  climatology the band series feed.
- [`ts-mjo/ERA5_VPM_CLOUD_BRIEF.md`](ts-mjo/ERA5_VPM_CLOUD_BRIEF.md) — ERA5's native grid is
  this same N320 grid, point for point.
- `cleanup_aifs_run.py` — the retention guard, and `HEAVY_PREFIXES`.
