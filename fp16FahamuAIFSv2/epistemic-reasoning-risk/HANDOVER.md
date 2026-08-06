# Handover — S2S process-BN on the AIFS-ENS store, for second opinion

**Date:** 2026-08-06 · **Author:** this session (Claude Opus 5, driven by @nishadhka)
**Status:** uncommitted working tree · Python side runs on real data · **Julia side unrun**
**Purpose:** hand this to a reviewer. Everything below is stated so it can be *attacked*.
Claims are numbered; if you disagree, reply by number.

---

## 0. TL;DR for the reviewer

We adapted two ICPAC bn-ibf reference implementations (a Julia/RxInfer flood BN and its Python
icechunk data-prep) to a different substrate: the **AIFS-ENS v2 S2S Icechunk store** (50
members, native N320, hours 432–792), following the grouped process ontology proposed in
`qd-1.md.txt`. Three new files, one registry bump.

The three things most worth a second opinion, in order:

1. **Is the target node right?** We replaced "flood risk" with an ordered **catchment
   water-balance pressure** (deficit ↔ surplus) because the store starts at day 18. §4.1
2. **Is the evidence wiring honest, or subtly circular?** The chain CPTs are *counted from the
   same 50 members* that also supply the evidence. We argue that is correct; it is also the
   most attackable decision in the design. §4.3
3. **Are the thresholds defensible?** All state definitions are now absolute physical
   thresholds. Every number is a judgement call and they are listed for challenge in §6.

---

## 1. What existed before this session

Committed already (`730a52c`, `e4bfe45`, `0ed0521`):

| File | What it is |
|---|---|
| `epistemic-reasoning-summary.md` | the CRMA/epistemic framing — loci, CPTs, evidence typing |
| `ICECHUNK_S2S_DIAGNOSTICS.md` | what the store enables beyond skill scores |
| `CPT_BUILD_CRITIQUE_AND_PLAN.md` | critique of the two above + 5-phase plan *(written earlier in this same session, commit `e4bfe45`)* |
| `NETWORK_DIRECTION.md` | Phase-0 contract: generative direction, latent `H`, evidence as children `P(E\|H)` |
| `discretization_registry.yaml` | Phase-0 contract: versioned state definitions |
| `cpt_build.py`, `cpt_artifact_20260730.nc`, `IMPLEMENTATION.md` | Phase-1 counter (z500 regime → IVT), ESS ≈ 1.85 finding |

The new work is meant to sit *on top of* those contracts, not replace them. **Check that it
actually does** — particularly that the generative direction in `NETWORK_DIRECTION.md` is
respected (we believe it is: `RO` hangs off `W` as a child, never as a parent).

## 2. What this session added

| File | Lines | Status |
|---|---|---|
| `S2S_BN_ONTOLOGY.md` | 296 | the reasoning; the document to review first |
| `s2s_bn_evidence_prep.py` | 545 | **runs**, validated on `cycle-20260730_0000` |
| `s2s_water_balance_bn.jl` | 592 | **never executed** (no Julia on this box) |
| `discretization_registry.yaml` | +137 | bumped 0.1.0 → 0.2.0; adds `process_loci`, `blocked_loci` |

Untracked/uncommitted; `qd-1.md.txt` (the user-supplied brief) is also untracked.
Inspiration: `icpac-igad/bn-ibf@jua-bnet` — `flood_bn_ibf_v1.jl`, `flood_data_prep.py`.

## 3. Verified facts — please re-verify, do not take on trust

Everything below was probed directly against
`/tank/projects/aifs-run/20260730_0000/icechunk_v2`, tag `cycle-20260730_0000`, with
`/tank/projects/micromamba/envs/aifs-gpu/bin/python`:

| Fact | Value |
|---|---|
| Shape | `member=50`, `time=132` allocated, **61 written** (idx 71–131 → 432–792 h, 6-hourly), `values=542080` |
| Vars | 122 data vars; `t,u,v,w,z` on 14 levels, `q` on 13 (no 10 hPa) |
| **Absent** | `cape`, `tcwv`, `lsm`, `z`/`slor`/`sdor` (surface orography constants) |
| **Present, unexpectedly useful** | `ro` (runoff), `sf`, `snowc`, `swvl1/2`, `cp`, `w` at all levels |
| Flux semantics | `tp, cp, ro, sf, ssrd, strd` are **interval per-6h**, not accumulated — established by flat field-means across lead time (the naive "fraction of cells decreasing" test is *inconclusive* for `ro`/`sf` because only 5.8 %/8.2 % of cells are nonzero) |
| Land mask | `finite(swvl1)` — 68.3 % NaN globally; IGAD box 11 342 cells, 8 780 land |

**Re-verification is cheap** — the probe scripts are short; ask if you want them replayed.
If any of these are wrong, most of §4 collapses, so this is the first thing to check.

## 4. Design decisions — rationale *and* the case against

### 4.1 Target node = catchment water-balance pressure, not flood risk

*For:* the store's first written step is hour 432. Event-scale flood reasoning at day 18–33 is
not defensible; qd-1 says the same ("broad water-balance tendency, not catchment-specific flood
prediction" for 360–800 h). One ordered node (strong_deficit … extreme_surplus) serves both
flood and drought reasoning, which matches ICPAC's actual dual mandate.

*Against (argue this if you want to):* it makes the output harder to act on than a flood
probability, and it is not what the bn-ibf pipeline downstream expects. If the consumer needs a
flood trigger, this design forces a further, unmodelled step.

### 4.2 Chain DAG instead of the original's flat multi-parent CPT

*For:* bn-ibf hangs 5–6 parents on one risk node; those parents are near-independent summaries
of a single field (TP). Here the candidates (`z500, msl, q/u/v, w, t, tp, tcw`) are all
functions of the same model state, so as siblings they multiply-count one piece of evidence.
qd-1 warns against exactly this ("avoid this structure"), and the critique doc's A5 makes it
concrete.

*Against:* a chain assumes the mechanism is genuinely sequential. Real convection is not — a
supportive environment and moisture supply interact, and our `P(P|M,R)` canonical form imposes
a particular interaction shape (noisy-MAX-like) that nobody has validated.

### 4.3 ⚠ The decision most in need of a second opinion: where evidence attaches

The 50 members each give **one state at every node simultaneously**, i.e. a joint draw over the
whole chain. We use that twice:

- to **count** `P(M|C)`, `P(R|M)`, `P(P|M,R)` (one state per member per window, n ≤ 50,
  ESS-discounted), and
- to supply the **evidence** at `C`.

*The obvious objection:* applying a CPT counted from sample *X* to evidence drawn from sample
*X* is circular — the posterior just reproduces the ensemble's own joint marginal, and the BN
adds nothing.

*Our answer, which you should stress-test:* that is not a bug, it is the *correct null*. The
forecast-side chain is only meant to transport the ensemble's joint structure faithfully; the
BN's added value is where **non-forecast information enters** — the antecedent node `A` and the
elicited `P(W|P,A)`, `P(RO|W)`. If a reviewer concludes the forecast side should instead be
collapsed to a single node (since it carries one sample's information), that is a legitimate
simplification and we would like to hear it argued.

*Mitigation already in place:* `--evidence-mode chain` (default) attaches ensemble evidence
only at `C`, `A`, `RO`; `--evidence-mode all` reproduces the naive port and **warns** that the
posterior is over-sharpened.

### 4.4 Runoff as an evidence child, not a target

`ro` exists in the store. We deliberately did **not** promote it to the endpoint (no routing, no
real catchment in the model), and instead hang it off `W` as `P(RO|W)` with a deliberately flat
likelihood, enforced by a self-test asserting the posterior shift stays < 0.30.

*Against:* one could argue `ro` is the most direct catchment signal available and deserves more
weight — or conversely that an unrouted bucket-scheme runoff should not be in the network at all.

### 4.5 Absolute thresholds instead of within-ensemble quantiles — a bug found by running it

The first draft defined states as within-ensemble quantiles, reasoning "no climatology exists,
so make everything relative". **That is wrong.** A quantile state is a *rank*: the fraction of
members per state is fixed by the cut points regardless of the forecast. Empirically, on
`cycle-20260730_0000`, every quantile-defined node returned identical soft evidence for both
lead windows (`A = [0.2]×5`, `R = [0.26,0.24,0.24,0.26]`) — zero information. Only the k-means
circulation node varied, because it classifies a *pattern*, not a rank.

Fixed by absolute, physically-anchored thresholds (§6). After the fix the evidence moves with
the forecast (R: 0.72 → 0.86 *supportive* between W1 and W2).

*Reviewer check:* is the reasoning above right, and is the replacement sound, or did we trade an
informationless definition for an arbitrary one?

## 5. Known issues we did not fix — please confirm or dispute

1. **`tp` is region-meaned over all cells, `ro`/`swvl` over land cells only.** Inconsistent. For
   the whole-IGAD demo box this dilutes precipitation with ocean cells. Moot for admin-1 units
   (all land), but it should probably be land-masked for consistency. *We consider this a real
   defect.*
2. **The antecedent node is not actually independent of the forecast.** The fallback `A` uses
   the model's own `swvl1/2` from the *same run* that produces `P`. So `P(W|P,A)` treats as
   separate parents two quantities from one model integration. The staleness problem is
   documented in the ontology note; **this dependence problem is arguably worse** and is not.
3. **The circulation ordering index is season-dependent.** Clusters are ordered by
   `z(−msl_anom) + z(v_850)`; a southerly 850 hPa anomaly is moisture-bearing for East Africa in
   some seasons and not others (the Somali jet reverses). The ordering may invert between MAM
   and OND. Needs a seasonal check.
4. **Evidence is not ESS-discounted, only the counts are.** ESS ≈ 1.85 is applied when counting
   CPTs, but the `C` state fractions enter at full strength as if 50 independent members.
5. **`kmeans2` can return empty clusters** (`missing="warn"`); the ordering step then sorts an
   `inf` mean. Not observed in the runs, but not guarded.
6. **Quasi-equal-area assumption.** We take unweighted cell means on the reduced Gaussian grid
   on the grounds that N320 cells are near-equal-area. Worth a sanity check by someone who knows
   the grid better than we do.
7. **Moisture node degenerate at box scale** — all 50 members in `normal` (region-mean IVT
   157–230, inside the 150–250 bin). We call this a scale artifact and refuse to re-tune the
   threshold to split the sample. A reviewer may reasonably say the threshold set is simply
   wrong for regional means.

## 6. Every judgement number, in one place (attack these)

**Python — `s2s_bn_evidence_prep.py` `THRESH`:**

| Quantity | Cuts | Stated anchor |
|---|---|---|
| region-mean \|IVT\| (kg m⁻¹ s⁻¹) | 150 / 250 / 350 | AR criterion (Guan & Waliser 250) at 0.6×/1.0×/1.4× |
| IVT persistence for `extreme` | ≥ 0.5 of steps above 250 | arbitrary-but-round |
| rainfall-env index | −1.0 / 0.5 / 2.0 | index = `ascent/0.01 + (lapse−26)/3 + (rh700−0.50)/0.15` |
| precip rate (mm day⁻¹) | 1.0 / 4.0 | round numbers, *not* climatological |
| episodic vs persistent | `peak24/total > 0.5` | scale-free shape rule |
| soil water (m³ m⁻³) | 0.12 / 0.20 / 0.28 / 0.35 | loam wilting point / field capacity |
| runoff rate (mm day⁻¹) | 0.05 / 0.25 / 0.75 | round numbers |

**Julia — `s2s_water_balance_bn.jl` (the elicited lower network, Route E):**
`M_SCORE`, `R_SCORE`, `P_SCORE`, `A_SCORE`, `W_CENTRE`; kernel widths `sigma` (1.0 for W1,
**1.35 for W2** — the deliberate widening with lead); the two named non-linearities in
`P(W|P,A)` (**−0.45** heavy-rain-on-dry-ground damping, **+0.45** persistent-rain-on-wet
amplification); the `w_pers` episodic/persistent split; the `P(RO|W)` matrix; and
`cost_loss_ratio = 0.2` (pre-positioned stockpiles, Weingärtner & Wilkinson 2019) in the
two-sided CRMA rule.

**Structural choices:** `k = 4` regimes; lead windows W1 = 432–576 h, W2 = 582–792 h;
region box lat [−12, 18], lon [22, 52].

## 7. What is tested, and what is not

**Tested (real data, `cycle-20260730_0000`):** the whole Python path — store open, masks, all
six diagnostics, state assignment, soft evidence, per-member sidecar, counted CPTs, netCDF and
JSON artifacts. Sanity checks that passed: counts sum to 50 (one state per member — the A1
rule); C occupancy `[8,11,15,16]` matches the reported fractions; ESS 1.85/1.87 reproduces the
earlier `cpt_build.py` finding; unobserved parent rows fall back to the prior, observed rows
carry real structure (`M→R` = `[0.05, 0.274, 0.626, 0.05]`).

**Not tested:** *anything in Julia.* No Julia on this machine. The BN mirrors the reference
file's verified RxInfer idioms (`DiscreteTransition(node, diageye(K))` evidence channels,
`@initialization`, `missing` on the queried node) and ships a `--test` self-check covering CPT
normalisation, wet/dry monotonicity, the runoff-corroboration bound and the lead-window
widening — **but none of it has been run.** The default inference path (`infer_chain`) is exact
tensor contraction and does not need RxInfer at all, so a reviewer with Julia can start there.
Also untested: the `--adm1` GeoJSON path (pure-numpy point-in-polygon, written because
geopandas/regionmask are not installed) — no admin-1 file was available.

**No verification of any kind against observations.** No skill, no null model, no calibration.

## 8. Suggested review order

1. §3 — re-verify the store facts; if `tp`/`ro` interval semantics or the absent-field list are
   wrong, stop and tell us.
2. §4.3 — the circularity question. This is the one we most want challenged.
3. §5.1 and §5.2 — the two defects we have named but not fixed.
4. §6 — the thresholds and elicited scores, ideally by someone with East-African forecasting
   experience rather than by a modeller.
5. Run `julia s2s_water_balance_bn.jl --test` if you have Julia, and tell us what breaks.

## 9. Open questions for a human decision

- Is a **water-balance tendency** the product ICPAC wants from day-18–33 lead, or should this
  target a seasonal-outlook framing instead?
- Should the forecast side collapse to a single node (§4.3)?
- Which spatial unit is canonical — admin-1, or hydrological catchments? The code supports
  arbitrary polygons; nobody has decided.
- Who owns the elicited numbers in §6? They are currently *ours*, which is the wrong provenance
  for an operational network — the plan calls for a named-expert elicitation record.
- Do we accept the model soil-moisture proxy for `A` in the interim, or block the whole chain
  until the CHIRPS/IMERG ingest (Phase 2) or the reduced full-window field set exists?

## 10. Reproduce

```bash
python s2s_bn_evidence_prep.py \
    --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
    --tag cycle-20260730_0000 \
    --out evidence_20260730.csv --member-out evidence_members_20260730.csv \
    --cpt-out process_cpt_20260730.nc --cpt-json process_cpt_20260730.json
# ~4 min, single unit; the IVT vertical integral over 13 levels dominates the runtime

julia --project s2s_water_balance_bn.jl --test
julia --project s2s_water_balance_bn.jl \
    --input-csv evidence_20260730.csv --output-csv wbp_20260730.csv \
    --cpt-json process_cpt_20260730.json --evidence-mode chain
```

Python env: `/tank/projects/micromamba/envs/aifs-gpu` (icechunk, xarray, zarr≥3, scipy, yaml,
pandas — **no** geopandas/regionmask, hence the hand-rolled point-in-polygon).
Julia deps required but never installed here: `RxInfer, CSV, DataFrames, JSON3`.

---

*Nothing in this handover is committed. Working tree: `discretization_registry.yaml` modified;
`S2S_BN_ONTOLOGY.md`, `s2s_bn_evidence_prep.py`, `s2s_water_balance_bn.jl`, `qd-1.md.txt`
untracked; the `ICECHUNK_S2S_DIAGNOSTICS.md` move into this folder is staged from earlier.*
