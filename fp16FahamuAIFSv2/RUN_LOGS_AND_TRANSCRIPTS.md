# Run logs and submission transcripts

Where each step's console output goes, and how the per-cycle
`fp16FahamuAIFSv2_<DATE>.txt` submission transcript is assembled from it.

---

## Where the logs live

**`/tank/projects/`** — flat, outside the repo and outside the run directory.

That location is deliberate: `cleanup_aifs_run.py` only walks
`RUN_ROOT = /tank/projects/aifs-run`, so reclaiming ~630 GB from a finished cycle
never touches the logs. They are the only surviving record of a run once its
store is deleted, and several of them (the O96/N320 timings, the SIGTERM at
member 40 on 20260820) have already been the source for a `run_commands_*.md`
after the data was gone.

Nothing prunes them either — plan on that rather than assuming rotation.

## Naming

`run_<MMDD>_<step>.log`, plus `proto_gcs_<DATE>_all50.log` for Step 1:

| Step | What | Log |
|------|------|-----|
| 1 | pkl creation | `proto_gcs_20260820_all50.log` |
| 2 / 2N | GPU inference → Icechunk | `run_0820_n320.log` |
| 3a | regrid → 1.5° NetCDF | `run_0820_3a.log` |
| 3b | quintiles → AI-WQ NetCDF | `run_0820_3b.log` |
| 3c | submit via ECBox | `run_0820_3c.log` |

**The convention has drifted, so check before trusting a name.** Earlier cycles
used `run_<MMDD>_all.log` for everything, `run_<MMDD>_downstream.log` for 3a+3b,
and `run_0813_3ab.log` — which despite its name holds **only 3a**
(`aifs_n320_grib_1p5defg_nc_cli.py`), with no 3b or 3c output in it at all. Grep
for a step's own banner rather than going by the filename:

```bash
grep -c "Step 6: Ready for AI Weather Quest" <log>   # 3b ran into this log
grep -c "Submitting mslp_week1"              <log>   # 3c ran into this log
```

## How they are written

Every long step is launched detached with stdout and stderr redirected:

```bash
setsid nohup bash -c "cd $PWD; $PY -u ../shared/forecast_submission_cli.py \
    --date 20260820 --v2 --output-dir $BASE/aiwq" \
    > /tank/projects/run_0820_3c.log 2>&1 < /dev/null &
```

Three parts of that are load-bearing:

- **`-u`** — unbuffered. Without it Python flushes in blocks and the log tells
  you nothing while the step runs; you cannot distinguish a slow ECBox file from
  a hung one, which is exactly when you need to know.
- **`setsid` + `< /dev/null`** — survives the launching shell. At ~12 min/file,
  3c exceeds any foreground timeout, and a SIGTERM mid-step is what left the
  20260820 O96 rollout at 39/50 members.
- **`2>&1`** — the AI-WQ and icechunk warnings go to stderr, and they are part of
  the record (the FTP-rename failure on 20260813 showed up only there).

---

## Building `fp16FahamuAIFSv2_<DATE>.txt`

The transcript is **3b followed by 3c**, verbatim, with the two shell prompt
lines that would have separated them in an interactive session. Steps 1, 2 and
3a are not part of it.

```bash
DATE=20260820; MMDD=0820
BASE=/tank/projects/aifs-run/${DATE}_0000
P='(aifs-gpu) ezra@e4drr-crafd-icpac:~/ea-aifs/fp16FahamuAIFSv2$ '

{
  printf '%spython ../shared/ensemble_quintile_analysis_cli.py --date %s --v2 \\\n' "$P" "$DATE"
  printf '    --local-nc-dir %s/nc_1p5deg --work-dir %s/aiwq\n' "$BASE" "$BASE"
  cat /tank/projects/run_${MMDD}_3b.log

  printf '%spython ../shared/forecast_submission_cli.py --date %s --v2 \\\n' "$P" "$DATE"
  printf '    --output-dir %s/aiwq          # AI-WQ 3.29 routes to ECBox\n' "$BASE"
  cat /tank/projects/run_${MMDD}_3c.log

  printf '%s\n' "$P"
} > fp16FahamuAIFSv2_${DATE}.txt

# logs keep raw colour codes; terminal-captured transcripts do not
sed -i 's/\x1b\[[0-9;]*m//g' fp16FahamuAIFSv2_${DATE}.txt
```

### Two things to do before committing

**1. Strip ANSI escapes.** icechunk's Rust logger emits colour codes, so a log
carries them and a copy-pasted terminal scrollback does not. `20260813` has none
(it was captured from the terminal), `20260806` and the first cut of `20260820`
had two each. The `sed` above normalises it; without it the file looks fine in a
terminal and is full of `^[[33m` in an editor or a diff.

**2. Scan for credentials.** 3c authenticates **twice per file**, so the log is a
verbatim auth transcript. It has never leaked a secret, but that is worth
confirming rather than assuming:

```bash
grep -cF "$(grep '^ecbox=' .env | cut -d= -f2-)"          fp16FahamuAIFSv2_${DATE}.txt   # want 0
grep -cF "$(grep '^AIWQ_PASSWORD=' .env | cut -d= -f2-)"  fp16FahamuAIFSv2_${DATE}.txt   # want 0
grep -niE "password|token|secret|Bearer|api[_-]?key"      fp16FahamuAIFSv2_${DATE}.txt   # want nothing
```

### Expected shape

`20260820` against its `20260813` predecessor — the step banners should line up
within a few lines:

| | 20260813 | 20260820 |
|---|---|---|
| 3b command | 1 | 1 |
| Steps 1–6 | 12, 212, 227, 305, 308, 338 | 12, 214, 229, 307, 310, 340 |
| 3c command | 342 | 344 |
| `SUBMISSION SUMMARY` | 627 | 952 |
| total lines | 635 | **960** |

**The file grows every cycle for a reason unrelated to the forecast.** The AI-WQ
package prints the full registered-teams CSV on every call — twice per file, six
files — and that table has gone from 104 rows to 118. Roughly 325 of the 960
lines are that table. A jump in length is not a signal that anything changed in
the run.

### Cycles before 20260820

`20260709` … `20260813` were captured from **terminal scrollback**, not from
logs — 3b and 3c were run in the foreground, so no `run_<MMDD>_3b.log` or
`run_<MMDD>_3c.log` exists for them and the recipe above cannot reproduce them.
From `20260820` on, both steps run detached and the transcript is assembled from
the logs.

---

## Related

- [`run_commands_20260820.md`](run_commands_20260820.md) — the per-cycle record
  the logs feed, including the pre-flight that works offline and the note that
  `--dry-run` does **not** exercise the ECBox transport.
- [`LOCAL_GPU_RUN.md`](LOCAL_GPU_RUN.md) — Step 2 on the local box.
- `cleanup_aifs_run.py` — reclaims a finished cycle; leaves `/tank/projects/*.log`
  alone.
