#!/usr/bin/env python3
"""
Fully-local AIFS-ENS-2.0 FP16 runner -> local Icechunk store (no GCS, no GRIB).

On-box counterpart of ``fp16_automate_aifs_gpu_pipeline_v2.py``. Where that
orchestrator does download(GCS) -> GRIB inference -> upload(GCS) -> cleanup, this
script collapses the loop to **read local pkl -> infer -> write straight into a local
Icechunk store -> commit** (one ACID snapshot per member). Nothing touches GCS, so no
``coiled-data.json`` is needed. The existing GRIB+GCS pipeline is left untouched as the
fallback.

Output is the ``(member, time, values)`` Icechunk schema from ``ICECHUNK_PATH_A.md``
(native N320 cells, ALL fields kept incl. ``cos_mwd``/``sin_mwd``), written via
``icechunk_output.IcechunkMemberWriter`` to a local-filesystem store, e.g.
``/tank/projects/aifs-run/<date>/icechunk_v2``.

Input pkls are the 112-field member states from
``ecmwf_opendata_pkl_input_aifsens_v2.py`` (run it with ``--no-upload --keep-local``
to produce them locally).

All ensemble members land in **one store**: the schema's ``member`` axis is sized by
``--n-members`` (default 50) and each member writes its own slice, owning disjoint chunks
(member-chunk size 1), so members never conflict. ``--skip-existing`` makes a multi-member
run idempotent — an interrupted 50-member run resumes without redoing finished members.

Usage::

    # single member, short lead
    python run_local_icechunk_v2.py --date 20260625_0000 --members 1 --lead-time 72 \
        --input-dir /tank/projects/aifs-run/20260625_0000/input_states \
        --store     /tank/projects/aifs-run/20260625_0000/icechunk_v2

    # full ensemble, all 50 members into the SAME store, resumable
    python run_local_icechunk_v2.py --date 20260625_0000 --members 1-50 --lead-time 960 \
        --input-dir .../input_states --store .../icechunk_v2 \
        --commit-every 1 --skip-existing
"""

import os
import sys
import time
import argparse

# This script lives in the model subfolder; import the sibling modules directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LEAD_TIME = 960          # hours (40 days); pass --lead-time 72 for a smoke test
TIME_STEP_HOURS = 6              # aifs-ens-2.0 timestep
INFERENCE_PRECISION = "16"       # FP16
INFERENCE_NUM_CHUNKS = 16        # higher -> less VRAM, slower
DEFAULT_N_MEMBERS = 50           # size of the member axis in the store


def parse_member_range(member_str):
    """'1-50' or '1,2,3' or '7' -> list[int]."""
    member_str = member_str.strip()
    if "-" in member_str:
        a, b = map(int, member_str.split("-"))
        return list(range(a, b + 1))
    if "," in member_str:
        return [int(m) for m in member_str.split(",")]
    return [int(member_str)]


def run(date_str, members, input_dir, store_path, lead_time,
        n_members=DEFAULT_N_MEMBERS, precision=INFERENCE_PRECISION,
        num_chunks=INFERENCE_NUM_CHUNKS, float_size="f4", tag=True,
        commit_every=1, skip_existing=False):

    # VRAM knobs must be set before the model / CUDA context is created.
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(num_chunks)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import fp16_multi_run_AIFS_ENS_v2 as runmod
    from icechunk_output import (open_repo_local, init_schema, schema_exists,
                                 member_written, IcechunkMemberWriter)
    from anemoi.inference.runners.simple import SimpleRunner

    bad = [m for m in members if m < 1 or m > n_members]
    if bad:
        raise ValueError(f"members {bad} outside 1..{n_members} (raise --n-members)")

    n_steps = lead_time // TIME_STEP_HOURS

    print("=" * 70)
    print("AIFS-ENS-2.0 FP16  ->  LOCAL Icechunk store (no GCS, no GRIB)")
    print("=" * 70)
    print(f"Date:        {date_str}")
    print(f"Members:     {members}")
    print(f"Input pkls:  {input_dir}")
    print(f"Store:       {store_path}")
    print(f"Precision:   FP16 | chunks={num_chunks} | dtype={float_size}")
    print(f"Lead time:   {lead_time}h ({lead_time // 24}d) -> {n_steps} steps @ {TIME_STEP_HOURS}h")
    print(f"Commit:      every {commit_every} step(s) = {commit_every * TIME_STEP_HOURS}h "
          f"| time_chunk={commit_every} (aligned, no amplification)")
    print(f"Member axis: 1..{n_members} in ONE store"
          + (" | skip already-written members (resume)" if skip_existing else ""))
    print("=" * 70)

    repo = open_repo_local(store_path)

    print("\nLoading model ecmwf/aifs-ens-2.0 (FP16)...")
    t0 = time.time()
    runner = SimpleRunner({"huggingface": "ecmwf/aifs-ens-2.0"},
                          device="cuda", precision=precision)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    ok, failed, skipped, snap = [], [], [], None
    for member in members:
        m_t0 = time.time()
        print(f"\n--- Member {member:03d} ---")

        # Idempotent resume: a member whose final step is already present is done.
        # Checked before the pkl load / inference so a resumed run does no wasted work.
        if skip_existing and schema_exists(repo) and member_written(repo, member - 1, n_steps):
            print(f"    [SKIP] member {member:03d} already complete in store")
            skipped.append(member)
            continue

        try:
            input_state = runmod.load_input_state_from_pickle(member, input_dir)
        except FileNotFoundError as e:
            print(f"    [SKIP] {e}")
            failed.append(member)
            continue

        ref_date = input_state["date"]
        runner.time_step = TIME_STEP_HOURS
        runner.lead_time = lead_time
        runner.reference_date = ref_date

        gen = runner.run(input_state=input_state, lead_time=lead_time)
        try:
            first = next(gen)
        except StopIteration:
            print("    [FAIL] inference yielded no states")
            failed.append(member)
            continue

        # One-time schema init, from the first real state (var names/coords exact).
        if not schema_exists(repo):
            lats = first["latitudes"]
            lons = first["longitudes"]
            n_values = len(lats)
            var_names = list(first["fields"].keys())
            print(f"    [SCHEMA] init {n_members}x{n_steps}x{n_values}, "
                  f"{len(var_names)} vars ({float_size}), time_chunk={commit_every}")
            init_schema(repo, n_members=n_members, n_steps=n_steps,
                        n_values=n_values, var_names=var_names,
                        latitudes=lats, longitudes=lons,
                        ref_date=ref_date, timestep_s=TIME_STEP_HOURS * 3600,
                        float_size=float_size, time_chunk=commit_every)

        writer = IcechunkMemberWriter(repo, member_index=member - 1,
                                      member_number=member, commit_every=commit_every)
        writer.write_step(first)                 # keep ALL fields (no wave drop); commits on cadence
        for state in gen:
            writer.write_step(state)
            if writer.n % 4 == 0:
                print(f"    {writer.n * TIME_STEP_HOURS}h / {lead_time}h "
                      f"({len(writer.snapshots)} commits)")

        if writer.n != n_steps:
            print(f"    [WARN] wrote {writer.n} steps, expected {n_steps}")
        snap = writer.finalize()                 # flush any uncommitted tail steps
        ok.append(member)
        print(f"    [OK] member {member:03d}: {writer.n} steps, "
              f"{len(writer.snapshots)} commits (last {snap}) in {time.time() - m_t0:.1f}s")

    if tag and snap is not None:
        try:
            repo.create_tag(f"cycle-{date_str}", snapshot_id=snap)
            print(f"\n[TAG] cycle-{date_str} -> {snap}")
        except Exception as e:
            print(f"\n[TAG] skipped ({e})")

    print("\n" + "=" * 70)
    print(f"DONE: {len(ok)} written, {len(skipped)} skipped, {len(failed)} failed "
          f"of {len(members)} members -> {store_path}")
    if skipped:
        print(f"Skipped (already in store): {skipped}")
    if failed:
        print(f"Failed: {failed}")
    print("=" * 70)
    return len(failed) == 0


def main():
    ap = argparse.ArgumentParser(
        description="Local AIFS-ENS-2.0 FP16 inference into a local Icechunk store")
    ap.add_argument("--date", required=True, help="cycle date YYYYMMDD_0000")
    ap.add_argument("--members", default="1", help="'1-50' or '1,2,3' or '7'")
    ap.add_argument("--lead-time", type=int, default=DEFAULT_LEAD_TIME,
                    help="forecast lead time (hours), default 960")
    ap.add_argument("--input-dir", required=True,
                    help="dir holding input_state_member_NNN.pkl")
    ap.add_argument("--store", required=True,
                    help="local Icechunk store path (created if missing)")
    ap.add_argument("--n-members", type=int, default=DEFAULT_N_MEMBERS,
                    help="size of the member axis in the store (default 50)")
    ap.add_argument("--precision", default=INFERENCE_PRECISION, choices=["16", "32"])
    ap.add_argument("--chunks", type=int, default=INFERENCE_NUM_CHUNKS,
                    help="ANEMOI_INFERENCE_NUM_CHUNKS (default 16)")
    ap.add_argument("--float-size", default="f4", choices=["f4", "f2"],
                    help="stored dtype (f4 exact / f2 bulk)")
    ap.add_argument("--commit-every", type=int, default=1,
                    help="commit every N model steps (1 = every 6h = per step, "
                         "default; 2 = 12h, 12 = 72h). time_chunk is aligned to this "
                         "so writes never amplify.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip members whose final step is already in the store "
                         "(idempotent resume of an interrupted multi-member run)")
    ap.add_argument("--no-tag", action="store_true", help="do not tag the cycle")
    args = ap.parse_args()

    members = parse_member_range(args.members)
    ok = run(date_str=args.date, members=members, input_dir=args.input_dir,
             store_path=args.store, lead_time=args.lead_time,
             n_members=args.n_members, precision=args.precision,
             num_chunks=args.chunks, float_size=args.float_size,
             tag=not args.no_tag, commit_every=args.commit_every,
             skip_existing=args.skip_existing)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
