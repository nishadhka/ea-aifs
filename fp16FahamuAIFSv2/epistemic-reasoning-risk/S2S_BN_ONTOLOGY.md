# The grouped process ontology on the AIFS-ENS v2 Icechunk store

*`qd-1.md.txt`'s process DAG, the bn-ibf Julia/RxInfer Bayesian network, and what the store
actually contains — reconciled.*

Inspiration: [`icpac-igad/bn-ibf@jua-bnet`](https://github.com/icpac-igad/bn-ibf/tree/jua-bnet/flood_ibf)
— `flood_bn_ibf_v1.jl` (RxInfer BN, soft evidence, CRMA cost-loss trigger, storyline
selection) and `flood_data_prep.py` (icechunk → zonal evidence CSV → BN). Both patterns are
kept. What changes is the *substrate* (a 50-member N320 S2S store at days 18–33 instead of a
0.25° 0–7 day TP store) and therefore the *ontology* (a water-balance tendency instead of a
flood probability).

**Files:** `s2s_bn_evidence_prep.py` (store → evidence), `s2s_water_balance_bn.jl` (BN),
`discretization_registry.yaml` v0.2.0 (the state contracts). Composes with the existing
`NETWORK_DIRECTION.md` (generative direction), `CPT_BUILD_CRITIQUE_AND_PLAN.md` (the rules
enforced below) and `cpt_build.py` (Phase-1 counter, whose ESS estimator is reused).

---

## 1. What the store actually holds — verified, not assumed

Probed directly on `cycle-20260730_0000` (`/tank/projects/aifs-run/20260730_0000/icechunk_v2`):

| Property | Verified value |
|---|---|
| Dimensions | `member=50`, `time=132` allocated, `values=542080` |
| Steps written | **61**, indices 71–131 → **432–792 h, 6-hourly** (days 18–33) |
| Data variables | **122** (120 fields + `latitude`/`longitude`) |
| Pressure levels | `t,u,v,w,z` on **14** (10→1000 hPa); `q` on **13** (no 10 hPa) |
| Surface | `msl, sp, skt, tcw, tp, cp, ro, sf, snowc, sd, ssrd, strd, tcc/hcc/mcc/lcc, swh, mwp, cdww, cos_mwd/sin_mwd`, plus the digit-named `2t, 2d, 10u/10v, stl1/2, swvl1/2` |
| `tp, cp, ro, sf, ssrd, strd` | **INTERVAL** (per-6 h), *not* accumulated — field means are flat with lead time; `tp` ≈ 0.69 mm/6 h, `ro` nonzero on 5.8 % of cells |
| Land mask | `lsm` is **absent**; the input-side mask survives as **NaN over sea in `swvl1`** (68.3 % NaN) — `finite(swvl1)` *is* the land mask |
| IGAD box (`reg.IGAD.v1`) | 11 342 N320 cells, **8 780 land** |

### The five things qd-1 assumes that this store does not provide

| qd-1 assumes | Reality | Consequence |
|---|---|---|
| "AIFS-ENS at **0.25°**" | native **N320 reduced Gaussian**, an *unstructured* `values` axis | `regionmask`/2-D zonal statistics do not apply; cells are quasi-equal-area so a plain cell mean *is* area-weighted (no cos φ weights). **Spatial derivatives are undefined** → moisture-flux *convergence* is blocked (needs regrid or a spherical-harmonic step). Every diagnostic here is integral/mean-based by construction. |
| "extending to about **800 hours**" | only **432–792 h** is written | qd-1's own lead ladder collapses: the 0–120 h *event-scale* and 120–360 h *regime* bands **do not exist**. Only the 360–800 h band survives, so the target node must be a **water-balance tendency** — which is exactly what qd-1 recommends for that band. |
| `cape` for the rainfall environment | **absent** | substituted by an explicit index: ascent `-(w_500+w_700)/2`, lapse `t_850−t_500`, mid-level RH from `q_700`,`t_700`. Registered as `loc.rgenv.v1`; it is an *environment* index, not an instability proxy. |
| `tcwv` | the store has **`tcw`** (total column water, includes condensate) | usable as an IWV cross-check only — never as a second moisture parent beside IVT (A5). |
| Turkana jet from terrain channelling | `z`(orography), `slor`, `sdor` are **input constants, not written to the store** | **Turkana-jet node is blocked** until a static N320 constants file is added. Registered in `blocked_loci` rather than silently approximated. |

### One thing the store provides that qd-1 assumes it cannot

> *"AIFS-ENS … cannot independently resolve catchment saturation or river response at 25 km."*

Correct as to *skill*, but the model does output **`ro` (runoff)** and **`swvl1/swvl2`
(soil water)**. That does not make it a hydrological model — there is no routing and no real
catchment — so these are **not** promoted to the target. They enter in the epistemically
honest place:

- `swvl1/2` → the **antecedent-wetness fallback proxy** (grade B, see §3),
- `ro` → an **evidence child of the water-balance node**: `P(RO | W)`, never a parent.

That is the generative direction `NETWORK_DIRECTION.md` fixed: evidence hangs *off* the
latent state. A deliberately flat likelihood means runoff can *corroborate* a surplus but
cannot *establish* one — enforced by a self-test asserting the posterior shift stays < 0.30.

---

## 2. The DAG as implemented

```text
                     [ensemble evidence]                    [external observation]
                              │                                       │
                              ▼                                       ▼
   C  circulation regime ─► M  moisture transport ─┬─► R  rainfall environment
                                                   │            │
                                                   └────────────┤
                                                                ▼
                                                     P  precipitation forcing
                                                                │
                            A  antecedent wetness ──────────────┤
                                                                ▼
                                            W  catchment water-balance pressure
                                                       │        │
                                              ┌────────┘        └────────┐
                                              ▼                          ▼
                                     surplus alert (CRMA)      deficit alert (CRMA)
                                              │
                                              └── RO  runoff  (evidence CHILD: P(RO|W))
```

**Why a chain and not bn-ibf's flat 5-parent CPT.** The flood original hangs `antecedent`,
`exceedance`, `spatial`, `trend`, `agreement` and `tail` directly on one risk node. That works
there because those parents are near-independent *summaries of one field* (TP). Here the
candidate parents are `z500, msl, u/v/q, w, t, tp, tcw…` — all functions of the same model
state. qd-1's "avoid this structure" warning is exactly right, and the critique's A5 makes it
concrete: correlated diagnostics as sibling parents multiply-count one piece of evidence and
sharpen the posterior with repetition. The chain gives each node **one** mechanistic parent
path, so nothing is counted twice.

### The ensemble is a *joint* sample — this is what makes the CPTs legitimate

Each of the 50 members is one internally consistent atmosphere, so member *m* supplies **one
state at every node simultaneously**: `(C, M, R, P, A, RO)_m`. That gives 50 joint draws over
the whole chain, from which `P(M|C)`, `P(R|M)`, `P(P|M,R)` are **counted** — the corrected
Route A: *one state per member per lead window*, `n ≤ 50`, never 50 × 61 correlated steps (A1),
with counts down-weighted to the ESS (B3, participation ratio, reused from `cpt_build.py`).

This also settles where evidence may be attached:

| `--evidence-mode` | Behaviour | Use |
|---|---|---|
| **`chain`** (default) | evidence enters only at **C** (ensemble), **A** (external/proxy) and **RO** (child); the counted CPTs carry it downstream | correct — the same 50 members are used once |
| `all` | soft evidence on every node, as a naive port of the flood script would do | comparison only; **warns** that the posterior is over-sharpened |

### The elicited seam is named, not hidden

`P(W | P, A)` and `P(RO | W)` are **not counted from anything** — no observation in this
system can yet close them (Route E). They are ordinal-kernel CPTs built from stated scores
with two named non-linearities: heavy rain on very dry ground is *damped* (infiltration),
persistent rain on wet ground is *amplified*. Their width `sigma` is the elicitation's stated
uncertainty and is **widened for the later lead window** (W1 = 1.0 → W2 = 1.35), so the day
25–33 posterior is structurally less certain than day 18–24. Upper network counted from the
store, lower network elicited — the seam the plan insisted must never be silent.

---

## 3. Node-by-node: qd-1 → store → implementation

| Node | qd-1 asks for | Store fields used (all verified present) | Diagnostic | State thresholds (absolute) | Grade |
|---|---|---|---|---|---|
| **C** circulation regime | `msl`, `z500`, `u850/v850` grouped into one regime | `z_500`, `msl`, `v_850` | window-mean `z_500` anomaly from the 50-member mean → whiten → k-means K=4; clusters **ordered** by a circulation-only index `z(−msl_anom) + z(v_850)` | unfavourable / neutral / convergent / strongly_convergent | A-relative |
| **M** moisture transport | `tcwv`, `q850/q700`, winds → IVT or MFT | `q_*`, `u_*`, `v_*` (13 shared levels) | full `|IVT| = |(1/g)∫q·V dp|`, region-mean, window-mean per member; **persistence** = step fraction above the *enhanced* threshold | cuts at **150 / 250 / 350** kg m⁻¹ s⁻¹ (0.6×/1.0×/1.4× the AR criterion) | A |
| **R** rainfall environment | `cape`, `t850/t500`, `q700/r700`, `w500/w700` | `w_500`, `w_700`, `t_850`, `t_500`, `q_700`, `t_700` | **no CAPE**: `ascent/0.01 + (lapse−26)/3 + (rh700−0.50)/0.15`, RH via Tetens | index cuts at **−1.0 / 0.5 / 2.0** | A |
| **P** precipitation forcing | `tp` + rolling accumulations, persistence, exceedance | `tp` (interval, m → mm) | rate = total/window-days; peak rolling 24 h; wet-step fraction | **1.0 / 4.0 mm day⁻¹**, heavy split by `peak24/total > 0.5` | A |
| **A** antecedent wetness | **observed** rainfall/soil moisture — *not* the forecast | `swvl1`, `swvl2` (land cells, first step of window) | depth-weighted `(0.07·swvl1 + 0.21·swvl2)/0.28` | **0.12 / 0.20 / 0.28 / 0.35** m³ m⁻³ (wilting point → field capacity) | **B-proxy** |
| **W** water-balance pressure | the ordered surplus/deficit target | — | elicited `P(W|P,A)`, ordinal kernel | strong_deficit … extreme_surplus | elicited |
| **RO** runoff | *(not in qd-1)* | `ro` (interval, land, window total mm) | rate = total/window-days | **0.05 / 0.25 / 0.75 mm day⁻¹** | B-proxy, **child** |

### States are absolute physical thresholds — a correction found by running it

The first draft of this design did the apparently-cautious thing: *"no climatology exists, so
define every state as a within-ensemble quantile."* **That is wrong, and running it proved
it.** A quantile state is a *rank*, so the fraction of members in each state is fixed by the
cut points no matter what the atmosphere is doing — quartiles always return
`[0.25, 0.25, 0.25, 0.25]`. On `cycle-20260730_0000` every quantile-defined node returned
**identical soft evidence for both lead windows** (`A = [0.2]×5`, `R = [0.26,0.24,0.24,0.26]`,
…): the unit-level evidence carried literally zero information about the forecast. Only the
circulation node varied, because k-means is a *pattern classification*, not a rank.

States therefore need a fixed reference. Until a model climatology exists (Phase 4 / Route B),
that reference is the physically-anchored thresholds above — each with its anchor stated in the
registry (the AR magnitude criterion; wilting point and field capacity; a dry/moist mid-level
RH divide), none of them fitted to this sample. Rates are used instead of totals so the two
lead windows are comparable despite their different lengths. With the fix, the evidence moves
with the forecast: the rainfall-generation environment strengthens from `0.72` to `0.86`
*supportive* between week 3 and weeks 4–5, and the circulation occupancy shifts markedly.

This is exactly the discipline `CPT_BUILD_CRITIQUE_AND_PLAN.md` demanded — *write the
thresholds down once, version them, never re-tune them per event* — and the registry enforces
it: changing any number mints a new locus id and counts never pool across ids.

**A scale caveat that comes with absolute thresholds.** In the demo run over the whole IGAD box
the moisture node is degenerate — all 50 members fall in `normal` (region-mean |IVT| 157–230,
entirely inside the 150–250 bin). That is a *scale artifact*, not a signal: the AR criterion
anchors a **local** IVT, while a mean over 11 342 cells is heavily smoothed. The intended unit
is an admin-1 or a catchment (`--adm1`), where the spread is real. A degenerate node is
reported as such rather than rescued by re-tuning the threshold to split the sample.

Every locus is **region-pooled before classification** (Part-0 admissibility): there is no
grid-cell node anywhere, because a single N320 cell at day 25 carries no predictability.

### The antecedent gap — the one structural problem neither document solves

qd-1 is emphatic and right: *"AIFS-ENS forecast precipitation alone should not be used to infer
the initial catchment wetness."* But for **this** store the problem is worse than qd-1 states,
and this is the single most important finding in this note:

- the forecast window starts at **hour 432**, so "antecedent" means the catchment state ~18 days
  *after* initialization;
- observations at initialization (CHIRPS/IMERG, the `R1` node) are therefore **~18 days stale**
  for the day-18 window and ~33 days stale for the far end;
- the rainfall that would bridge that gap — hours **0–432** — is **discarded and not stored**.

So there are exactly three options, and the choice must be recorded at the node:

1. **Store the reduced full-window field set** (`tp`, `swvl1/2`, `u850,u200,z500,tcw` every 12 h
   for 0–792 h, cheap relative to the 583 GB full store) so the 0–432 h forcing exists. *This is
   the real fix* and it is already step 5 of the diagnostics build order.
2. **Use the model's own `swvl1/2` at the window start** — implemented here as the fallback.
   Honest limitation: it is the model's soil water after 18+ days of free running with no
   land-surface data assimilation. Grade B, and the CSV records
   `antecedent_source=MODEL_SOIL_PROXY(swvl1,swvl2)@window_start` in every row.
3. **Observed antecedent at initialization + an explicit climatological gap term** — defensible
   for slow variables (deep soil, groundwater), not for the top layers.

Until (1) exists, the `A` node is the weakest link in the chain, and the BN says so rather than
hiding it.

---

## 4. What the BN produces

Per (spatial unit × lead window):

- the posterior over `W` — five ordered states,
- `p_surplus = P(elevated) + P(extreme)` and `p_deficit = P(strong) + P(mild)`,
- **two CRMA gradings**, not one. The flood original computes a single risk → single
  Monitor/Evaluate/Assess/Actionable ladder. A water-balance node serves both hazards, so the
  cost-loss rule (Murphy 1977; Richardson 2000; Lopez et al. 2020) is applied **independently to
  each tail** — an anticipatory action for flooding and one for drought have different C/L and
  must not be collapsed into one number,
- the implied precipitation-forcing marginal (what the chain thinks the rain does),
- **storylines**: per-member runs → worst-surplus / median / worst-deficit member with its
  exceedance fraction, following the original's `select_storylines`.

Provenance travels with every row: the cycle tag, registry version, ESS, whether the chain CPTs
were counted or elicited, the evidence mode, and the seam note.

---

## 5. Staging (qd-1's own order, with what is blocked)

**Stage 1 — process backbone. Implemented.** C → M → R → P + A → W, plus RO as evidence child.

**Stage 2 — regional mechanisms.**
- *Somali jet*: available now — `loc.somalijet.v1` already exists (`v_850`, 40–55 °E, 5 °S–5 °N).
  Enters as a **classifier of the moisture-transport regime**, per qd-1, not as a separate parent.
- *Moisture corridor*: use qd-1's term, **organised moisture-transport corridor**, not
  "atmospheric river" — classical AR geometry thresholds are extratropical and would mislabel
  monsoon flow. Needs the connected-object step; on an unstructured `values` axis this requires
  a neighbour graph, so it is real work, not a filter.
- *Turkana jet*: **BLOCKED** — needs orography that is not in the store.
- *Moisture-flux convergence*: **BLOCKED** — needs a spatial derivative; see §1.
- *Evaporative demand* (the drought branch qd-1 asks for): `2t`, `2d` (→ VPD), `ssrd`, `10u/10v`
  are all present; a reference-ET-like index is straightforward and is the natural next node.

**Stage 3 — background climate conditioning.** ENSO/IOD/MJO as **prior-conditioning on C**,
per qd-1 — never arrows into the hazard. Note the correction from the critique (A4): the
`skt`-based IOD/Niño index is *not* a good ensemble node, because AIFS has no ocean model, so
the 50 members' SST spread is near-degenerate; these should enter as an **observed
boundary-condition node at initialization** (OISST), grade B, registry id already reserved.

---

## 6. Running it

```bash
# 1. store -> evidence  (unit-level soft evidence + per-member sidecar + counted chain CPTs)
python s2s_bn_evidence_prep.py \
    --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
    --tag cycle-20260730_0000 \
    --out evidence_20260730.csv \
    --member-out evidence_members_20260730.csv \
    --cpt-out process_cpt_20260730.nc --cpt-json process_cpt_20260730.json
# optional admin-1 units — pure-numpy point-in-polygon, no geopandas/regionmask needed:
#   --adm1 icpac_adm1v3.geojson --adm1-id GID_1 --adm1-name NAME_1

# 2. evidence -> water-balance posterior + two CRMA gradings
julia --project s2s_water_balance_bn.jl \
    --input-csv evidence_20260730.csv --output-csv wbp_20260730.csv \
    --cpt-json process_cpt_20260730.json --evidence-mode chain

# 3. storylines (worst-surplus / median / worst-deficit member per unit)
julia --project s2s_water_balance_bn.jl \
    --member-csv evidence_members_20260730.csv --storyline-csv storylines_20260730.csv

julia --project s2s_water_balance_bn.jl --test    # CPT normalisation + monotonicity + width
```

---

## 7. Honest limits

- **`n_cycles = 1`.** Every counted CPT here is within-cycle joint structure, typed as such in
  the artifact. It is *not* the `P(E|H)` calibration — that needs matched forecast–observation
  pairs across many cycles (Phase 3), and no amount of member counting substitutes for it.
- **ESS ≈ 2, not 50.** The Phase-1 run found the regional `z500` spread at days 18–33 collapses
  to ~2 effective degrees of freedom (participation ratio 1.8–1.9, redundancy 0.96). The counted
  chain CPTs inherit that discount, so they are deliberately wide. A single low-DOF forecast
  cannot establish a conditional relationship.
- **The `A` node is a model proxy** until the full-window field set or a CHIRPS/IMERG ingest
  exists (§3).
- **The thresholds are provisional.** They are physically anchored and stated, but they are not
  a climatology. Replacing them with model-climatological percentiles is Phase 4, and doing so
  mints new registry ids rather than editing the current ones.
- **Node degeneracy at box scale is real** (the moisture node in the demo run) and is a reason
  to run with `--adm1`, not a reason to move a threshold.
- **The elicited seam is judgement**, versioned in git and auditable, but judgement.
- **The Julia side is untested on this machine** — Julia is not installed here. It mirrors the
  verified RxInfer idioms of `flood_bn_ibf_v1.jl` (`DiscreteTransition(node, diageye(K))`
  evidence channels, `@initialization`, `missing` on the queried node) and ships a `--test`
  self-check, but it has not been executed. The exact tensor-contraction path (`infer_chain`) is
  the default and does not depend on RxInfer at all.
- **No verification.** Nothing here has been scored against observations. The null model
  (climatology-only) and the skill harness are still Phase 5.

---

*Verified against `cycle-20260730_0000` on 2026-08-05: 122 data vars, 50 members, N320
(542 080 cells), 61 written steps at 432–792 h. Every field named in this document was probed
present or absent; every absence is registered in `blocked_loci` rather than approximated.*
