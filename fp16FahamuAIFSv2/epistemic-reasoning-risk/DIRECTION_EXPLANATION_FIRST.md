# Direction note — explanation first, risk downstream

**Date:** 2026-08-06 · **Status:** adopted; implemented in code as of commit following
`a988c44` · **Supersedes** the implicit scope of `S2S_BN_ONTOLOGY.md` (admin-1 risk as the goal)

---

## The question

> *Are we too focused on risk at admin-1 level? Should we dislodge from that and focus on
> explaining the forecast — the global circulation opportunity that AIFS-ENS v2's full 3-D
> output in the Icechunk store creates — with risk as a prelude rather than the goal?*

## The decision

**Yes. The circulation explanation record becomes the primary product; the Bayesian risk
network becomes a downstream consumer of the same loci.** Demoted, not deleted.

This is not a change of technique — the loci, the registry, the counting rule and the ESS
discipline all survive. It is a change of *what the pipeline is for*, and therefore of what
counts as done.

## Why — the argument from what we actually found

**1. Every weak link is on the risk side; every solid one is on the explanation side.**
Building the admin-1 BN surfaced four blocking dependencies that do not exist yet: no
CHIRPS/IMERG ingest, no EM-DAT, no exposure or vulnerability layer, no elicitation panel. And
three defects that are structural, not fixable by more code: the antecedent node is ~18 days
stale *and* not independent of the forecast (same model run); `P(W|P,A)` and `P(RO|W)` are
elicited by us with nothing able to close them; the moisture node degenerates once pooled to a
polygon. The explanation layer needs none of that. It needs the store, which exists, and ERA5
for verification, which is a well-posed problem in a way impact verification is not.

**2. The scale argument, applied consistently.** Part 0 of `CPT_BUILD_CRITIQUE_AND_PLAN.md`
says predictability at days 18–33 lives at planetary and synoptic scale, and that a locus is
admissible only where its scale retains predictability. Admin-1 pooling takes that
planetary-scale signal, projects it onto polygons where predictability is weakest, and then
multiplies it by elicited numbers nobody has validated. We wrote the rule and then partly broke
it by chasing admin-1 units.

**3. An asymmetry worth naming.** The state-definition bug (registry header; `S2S_BN_ONTOLOGY.md`
§3) was that within-ensemble quantiles are informationless *as risk states*. But *"n of 50
members place the atmosphere in regime B"* is exactly the right form for an explanation product.
**Ensemble-relative framing is native to explanation and fatal to calibrated risk.** The same
property that broke the risk states is the substrate of the explanation product.

**4. It is what the store is uniquely for.** The submission pipeline keeps 3 of 120 variables;
the store keeps the full 3-D state at native N320. The acceptance argument this whole folder
opens with — that MLWP needs *understanding, translating, communicating*, not only skill — is
an argument for the explanation product specifically. Risk framing routes the store's advantage
through a bottleneck that discards it.

## What changes

| | Before | After |
|---|---|---|
| Unit of analysis | admin-1 polygons | circulation regions and modes (`IGAD_EA`, `EQ_INDIAN`, `WIO`, `CONGO`); admin-1 optional |
| Primary output | risk grade per admin-1 | **circulation record**: ensemble counts over circulation states + a generated narrative, ESS attached to every count |
| Verification | impact verification (needs data we lack) | circulation diagnostics vs **ERA5** for the same valid window |
| Customer | anticipatory-action trigger | duty forecaster; and the *"why did the model favour this quintile"* loop back to the AI-WQ submission |
| Risk network | the goal | a consumer, switched on when the observation side exists |

## What this costs — the new first-class blocker

Under risk framing, the derivative problem was deferrable: velocity potential χ200, the
divergent/Walker circulation, moisture-flux convergence and vorticity all need spatial
derivatives, which are undefined on the store's unstructured reduced-Gaussian `values` axis, so
they sat in `blocked_loci` and we moved on.

**Under explanation framing those diagnostics *are* the product.** A regrid step
(`earthkit-regrid` to a regular grid) or a spherical-harmonic transform is now required work,
not a registered absence. The registry records this priority change.

Second cost: the reduced full-window field set (0–792 h, 12-hourly, a handful of fields) moves
from "nice later" to necessary — MJO propagation and moisture-corridor life-cycle cannot be
tracked from hour 432, and those are core explanation content rather than optional extras.

## What is kept

The registry and its provenance discipline (substrate-independent, and the real asset); the
lead-window stratification; the ESS discount; the one-state-per-member joint-draw counting; the
generative direction of `NETWORK_DIRECTION.md`; and the whole BN, unchanged in mechanism.

## Revised work order

**E1 — Circulation record.** *Done in this commit.* Regime occupancy, |IVT| distribution and
corridor-threshold exceedance counts, Somali-jet index, rainfall-environment components,
precipitation-rate distribution, ESS, and a narrative sentence — per region per lead window.
**E2 — Regrid or spherical-harmonic step** → χ200, divergent circulation, MFC, vorticity. The
blocker that now matters most.
**E3 — Reduced full-window field set** → MJO propagation, corridor life-cycle from genesis.
**E4 — Model climatology accumulation** → grade-C loci become true anomalies within a season.
**E5 — ERA5 verification** of the circulation diagnostics over the day-18–33 valid window.
**E6 — Quintile lineage** — join the record to the AI-WQ submission and answer *why* the model
favoured a quintile from the ingredients.
**R-track** — the water-balance BN resumes when Phase-2 observations land.

## What would show this decision to be wrong

Stated up front, so the pivot is falsifiable rather than a preference:

1. **Forecasters find counts-without-calibration unusable.** If the duty forecaster wants a
   probability they can act on and not "31 of 50 members, ESS ≈ 2", the explanation product is
   an academic artefact and risk framing was right.
2. **ERA5 verification shows the regime assignment is unstable** — if the same cycle's regimes
   don't correspond to anything verifiable, the narrative is confident noise.
3. **ESS ≈ 1.85 undercuts the explanation product too.** This is the sharpest one, and it cuts
   against us: with ~2 effective degrees of freedom, partitioning into **k = 4** regimes is
   probably over-partitioning, and the regime occupancy counts may be an artefact of k-means on
   a rank-2 spread. **Concrete test: rerun at k = 2 and check whether the occupancy structure
   and the narrative survive.** If they don't, the honest product is much thinner than E1
   currently implies — closer to "the ensemble carries two circulation stories this cycle".

Item 3 should be run before E2 is started.

---

*Implemented by `s2s_bn_evidence_prep.py --explain-out` (primary path) with
`--regions IGAD_EA,EQ_INDIAN,WIO,CONGO`. The BN (`s2s_water_balance_bn.jl`) is unchanged in
mechanism and re-typed as downstream. Registry v0.2.0 carries the new regions, the
`explanation_products` entry, and the blocked-loci priority change.*
