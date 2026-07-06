#!/usr/bin/env python3
"""
Icechunk ensemble output writer for AIFS-ENS-2.0 (Path A — **local** variant).

Implements the writer specified in ``ICECHUNK_PATH_A.md`` §2.1, but targets a
**local filesystem** Icechunk store (``icechunk.local_filesystem_storage``) instead
of GCS. Everything else — the ``(member, time, values)`` schema, the ``(1, n_steps,
n_values)`` chunking that lets members own disjoint chunks, the one-commit-per-member
ACID model — is identical, so this store is a drop-in for the GCS one and reads back
the same way with ``xarray.open_zarr``.

Use for on-box runs where the GPU inference writes straight into a store on a local
disk (e.g. ``/tank/projects/<date>/icechunk_v2``) — no GRIB staging, no upload.

Schema (native N320 reduced-Gaussian cells, no regrid at write time)::

    coords: latitude(values) f4, longitude(values) f4,
            time(time) i4 (seconds since reference_date), member(member) i4
    data:   one (member, time, values) f4/f2 array per model output field
            (ALL fields kept, incl. cos_mwd/sin_mwd — a gain over the GRIB path)

Typical use (single process, sequential members)::

    repo = open_repo_local("/tank/projects/20260625_0000/icechunk_v2")
    init_schema(repo, n_members=50, n_steps=12, n_values=542080,
                var_names=first_state_field_names,
                latitudes=lats, longitudes=lons,
                ref_date=ref_date, timestep_s=6*3600)      # once, before any writer
    w = IcechunkMemberWriter(repo, member_index=member-1)  # member 1 -> index 0
    for state in runner.run(...):
        w.write_step(state)
    w.commit(member)
"""

import numpy as np
import zarr
import icechunk


# ---------------------------------------------------------------------------
# Repository handles
# ---------------------------------------------------------------------------
def open_repo_local(path):
    """Open (or create) a local-filesystem Icechunk repository at ``path``.

    ``path`` is a directory on a local disk; it is created if missing.
    """
    storage = icechunk.local_filesystem_storage(str(path))
    return icechunk.Repository.open_or_create(storage)


def open_repo_gcs(bucket, prefix, service_account_file):
    """Open (or create) a GCS-backed Icechunk repository (parity with Path A doc)."""
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix,
                                   service_account_file=service_account_file)
    return icechunk.Repository.open_or_create(storage)


# ---------------------------------------------------------------------------
# Schema (create once, before any member is written)
# ---------------------------------------------------------------------------
def init_schema(repo, n_members, n_steps, n_values, var_names,
                latitudes, longitudes, ref_date, timestep_s,
                float_size="f4", time_chunk=1, commit_msg=None):
    """Create + commit the (member, time, values) schema on branch ``main``.

    Must run once, committed BEFORE any :class:`IcechunkMemberWriter` opens a
    session. ``var_names`` are the model output field names (derive them from the
    first yielded state so the set always matches the checkpoint exactly).
    ``latitudes``/``longitudes`` are the native N320 cell coordinates (length
    ``n_values``). Returns the commit snapshot id.

    ``time_chunk`` sets how many timesteps live in one chunk along the time axis.
    **Set it equal to the commit cadence** (``1`` = per model step / 6 h, the
    default): a step-by-step writer that commits every ``k`` steps into a chunk of
    ``time_chunk = k`` steps writes each chunk once (~no write amplification). A big
    ``time_chunk`` (e.g. ``n_steps``) written step-by-step read-modify-writes the
    whole chunk every step → severe amplification — see ``ICECHUNK_COMMIT_CADENCE.md``.
    """
    s = repo.writable_session("main")
    root = zarr.group(s.store, overwrite=True)

    root.create_array("latitude", shape=(n_values,), chunks=(n_values,),
                      dtype="f4", dimension_names=("values",))[:] = np.asarray(latitudes, "f4")
    root.create_array("longitude", shape=(n_values,), chunks=(n_values,),
                      dtype="f4", dimension_names=("values",))[:] = np.asarray(longitudes, "f4")

    t = root.create_array("time", shape=(n_steps,), chunks=(n_steps,),
                          dtype="i4", dimension_names=("time",))
    t[:] = (np.arange(1, n_steps + 1) * int(timestep_s)).astype("i4")
    t.attrs["units"] = f"seconds since {ref_date}"
    t.attrs["calendar"] = "gregorian"

    root.create_array("member", shape=(n_members,), chunks=(n_members,),
                      dtype="i4", dimension_names=("member",))[:] = \
        np.arange(1, n_members + 1, dtype="i4")

    tc = max(1, min(int(time_chunk), n_steps))
    for v in var_names:
        # fill_value=NaN so members/steps not yet written read as missing, not 0
        # (a 0 fill would masquerade as a real value, e.g. 0 K temperature).
        root.create_array(v, shape=(n_members, n_steps, n_values),
                          chunks=(1, tc, n_values), dtype=float_size,
                          fill_value=float("nan"),
                          dimension_names=("member", "time", "values"))

    msg = commit_msg or (f"init schema: {n_members}x{n_steps}x{n_values}, "
                         f"{len(var_names)} vars ({float_size}), time_chunk={tc}")
    return s.commit(msg)


def schema_exists(repo, sentinel="time"):
    """True if the schema group is already present on ``main`` (idempotent init)."""
    try:
        root = zarr.open_group(repo.readonly_session("main").store, mode="r")
        return sentinel in root
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-member writer (one ACID commit per member)
# ---------------------------------------------------------------------------
class IcechunkMemberWriter:
    """Writes ONE member's rollout into the (member, time, values) arrays, committing
    at a configurable step cadence.

    ``member_index`` is 0-based (ensemble member N -> index N-1). With
    ``commit_every=1`` (the default) it commits **after every model step** — i.e.
    every 6 h for aifs-ens-2.0, the natural inference cadence — so each step becomes a
    durable ACID snapshot as the rollout streams. Pair with ``time_chunk=commit_every``
    in :func:`init_schema` so each committed window is a whole chunk written once
    (no write amplification; see ``ICECHUNK_COMMIT_CADENCE.md``).

    Because each member owns disjoint chunks (member-chunk size 1), sequential members
    never share a chunk, so commits stay conflict-free.
    """

    def __init__(self, repo, member_index, member_number=None,
                 commit_every=1, branch="main"):
        self.repo = repo
        self.branch = branch
        self.m = int(member_index)
        self.member_number = int(member_number) if member_number is not None else self.m + 1
        self.commit_every = max(1, int(commit_every))
        self.n = 0                       # total steps written this member
        self._since_commit = 0           # steps written since last commit
        self._var_names = None
        self.snapshots = []              # snapshot id per committed window
        self._open()

    def _open(self):
        """Start a fresh writable session (called after each commit)."""
        self.session = self.repo.writable_session(self.branch)
        self.root = zarr.open_group(self.session.store, mode="r+")
        if self._var_names is None:
            self._var_names = set(self.root.array_keys())

    def write_step(self, state):
        """Write one yielded inference state (all fields), committing on cadence."""
        fields = state["fields"]
        for name, value in fields.items():
            if name in self._var_names:
                self.root[name][self.m, self.n, :] = np.asarray(value).reshape(-1)
        self.n += 1
        self._since_commit += 1
        if self._since_commit >= self.commit_every:
            self._commit_window()

    def _commit_window(self):
        """Commit the steps written since the last commit; reopen a fresh session."""
        first = self.n - self._since_commit + 1
        snap = self.session.commit(
            f"member {self.member_number:03d} steps {first:03d}-{self.n:03d}")
        self.snapshots.append(snap)
        self._since_commit = 0
        self._open()
        return snap

    def finalize(self):
        """Flush any steps not yet committed (when n_steps % commit_every != 0).

        Returns the last snapshot id for this member.
        """
        if self._since_commit > 0:
            self._commit_window()
        return self.snapshots[-1] if self.snapshots else None

    # Backwards-compatible alias: previously commit() sealed the whole member.
    def commit(self, member_number=None):
        if member_number is not None:
            self.member_number = int(member_number)
        return self.finalize()
