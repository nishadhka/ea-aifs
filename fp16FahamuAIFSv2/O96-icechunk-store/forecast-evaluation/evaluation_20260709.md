# Forecast evaluation — cycle 20260709 (O96, operational inputs)

Local AI-WQ RPSS, scored with `AI_WQ_package.forecast_evaluation` — the competition's own
code. **Nothing uploaded.**

| | |
|---|---|
| init | **2026-07-09 00z** |
| verification weeks | 2026-07-27 (week1), 2026-08-03 (week2) |
| source | `20260709_0000/icechunk_o96`, **compacted**, tag `cycle-20260709_0000-c` |
| store | 104 GB, 50/50 members, 132/132 steps |
| evaluated | 2026-08-22 |

## Result

| week | valid | tas | mslp | pr |
|---|---|---|---|---|
| week1 | 2026-07-27 | −0.035 | +0.023 | **+0.060** |
| week2 | 2026-08-03 | −0.196 | +0.098 | +0.004 |
| **mean** | | **−0.115** | **+0.060** | **+0.032** |

**Overall mean RPSS −0.008** — the first cycle to come out net *negative*.

### Diagnostics

| var | week | sharpness | hit-rate | reliability | bias | rmse | pattern-r |
|---|---|---|---|---|---|---|---|
| tas | 1 | 0.512 | 0.396 | −0.116 | +0.060 K | 1.96 K | 0.994 |
| tas | 2 | 0.538 | 0.377 | −0.160 | −0.019 K | 1.83 K | 0.995 |
| mslp | 1 | 0.442 | 0.269 | **−0.173** | +15.3 Pa | 439 Pa | 0.850 |
| mslp | 2 | 0.474 | 0.312 | −0.162 | +31.1 Pa | 502 Pa | 0.846 |
| pr | 1 | 0.389 | 0.317 | −0.073 | +0.50 mm | 22.7 mm | 0.684 |
| pr | 2 | 0.397 | 0.295 | −0.102 | −0.48 mm | 26.3 mm | 0.629 |

Biases stay negligible, so again this is calibration rather than a broken field.

**What is different here is `mslp`.** In the two earlier cycles `mslp` was the reliable
variable — reliability −0.001 to −0.05, RPSS +0.13 to +0.23. Here it is **−0.17**, the
worst reliability of any variable in any cycle scored so far, and its RPSS collapses to
+0.06. Its sharpness rose (0.44–0.47 vs 0.41) while its hit-rate fell hard (0.27–0.31 vs
0.37). So the ensemble became *more* confident about mid-latitude circulation in a week it
was *less* right — the signature of a flow-dependent bad week, not a systematic defect.

`pr` week1 (+0.060) is the best precipitation score across all three cycles, which points
the same way: skill at days 18–33 is dominated by what the atmosphere happened to be doing,
not by the configuration.

## All scored cycles

| cycle | inputs | tas | mslp | pr | overall |
|---|---|---|---|---|---|
| [20260212](evaluation_20260212.md) | donor-sourced (pre-50r1) | −0.074 | +0.132 | +0.032 | **+0.030** |
| [20260514](evaluation_20260514.md) | operational | −0.167 | +0.233 | +0.005 | **+0.024** |
| **20260709** | operational | −0.115 | +0.060 | +0.032 | **−0.008** |
| 20260820 | operational | — | — | — | verifies 2026-09-14 |

Consistent across all three: **`tas` is always negative** (−0.07 to −0.17) and always the
most over-confident variable (reliability −0.09 to −0.22). That is the one finding stable
enough to act on, and it is a dispersion problem, not an accuracy one — `tas` bias never
exceeds 0.06 K and pattern-r is 0.994+ everywhere.

`mslp` and `pr` swing widely between cycles (+0.06→+0.23 and +0.005→+0.032), which three
cycles cannot separate from weather. **Do not read a trend into three points.**

## Commands

```bash
cd .../O96-icechunk-store/forecast-evaluation
./evaluate_cycle.sh 20260709
```

Note this ran against a **compacted** store whose original tag was tombstoned. The
`--icechunk-tag cycle-20260709_0000` in the driver no longer resolves; the reader fell back
to branch `main` — which after compaction is the same snapshot — and reported doing so.
That fallback is what made this evaluation work unchanged.

## Caveats

1. Scores the O96 route (`N320 → O96 → 1.5°`), not a native-N320 submission. The regrid
   contributes ~3 % of the `tas` miscalibration — see
   [regrid_impact_n320_vs_o96.md](regrid_impact_n320_vs_o96.md).
2. Never submitted, so no leaderboard figure to check against.
3. Three cycles, three different seasons. Enough to see the `tas` pattern repeat; not
   enough for any statement about mean skill.
